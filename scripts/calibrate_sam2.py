"""Reproducible COCO val2017 calibration of the published SAM2.1 Tiny exports.

The full 5,000-image manifest is mandatory for production output. Memory inputs
come from deterministic translated-image sequences, not unrelated COCO frames.
These are calibration samples, not a video-accuracy validation dataset.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from convert_qdq_matmul import convert
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.calibrate import CalibrationMethod, TensorsData
from onnxruntime.quantization.qdq_quantizer import QDQQuantizer
from prepare_sam2 import SOURCES, prepare

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sam2_video_predictor import SAM2VideoPredictor  # noqa: E402
from settings import preprocess_image  # noqa: E402

PROTOCOL = "coco-val2017-sam21-tiny-512-minmax-v1"
OFFSETS = ((0, 0), (4, 0), (4, 4), (0, 4), (-4, 0))
EXCLUDED = {f"/layers.{i}/MatMul{suffix}" for i in range(4) for suffix in ("", "_1", "_4", "_5")}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def download_sources(source):
    source.mkdir(parents=True, exist_ok=True)
    for name in SOURCES.values():
        path = source / name
        if not path.exists():
            url = "https://storage.googleapis.com/ailia-models/segment-anything-2.1/" + name
            temporary = path.with_suffix(".download")
            print(f"Downloading {url}", flush=True)
            with (
                urllib.request.urlopen(url, timeout=120) as response,
                temporary.open("wb") as target,
            ):
                shutil.copyfileobj(response, target)
            onnx.checker.check_model(str(temporary))
            temporary.replace(path)
    if not (source / "pretrained_weights.npz").is_file():
        raise FileNotFoundError("Copy pretrained_weights.npz from ailia-models/segment-anything-2")


def dataset_manifest(images, annotations, source, smoke_images=0):
    data = json.loads(annotations.read_text())
    entries = sorted(data["images"], key=lambda x: x["id"])
    if len(entries) != 5000 or len({x["id"] for x in entries}) != 5000:
        raise ValueError("Expected the complete COCO val2017 annotation: 5,000 unique images")
    if {p.name for p in images.glob("*.jpg")} != {e["file_name"] for e in entries}:
        raise ValueError("COCO image directory must match all 5,000 annotated image filenames")
    records = []
    for entry in entries:
        path = images / entry["file_name"]
        image = cv2.imread(str(path))
        if image is None or image.shape[:2] != (entry["height"], entry["width"]):
            raise ValueError(f"Missing, corrupt or incorrectly sized image: {path}")
        records.append(dict(id=entry["id"], file_name=entry["file_name"], sha256=digest(path)))
    return dict(
        protocol=PROTOCOL,
        production=not smoke_images,
        images=records,
        smoke_images=smoke_images,
        annotations_sha256=digest(annotations),
        source_sha256={
            name: digest(source / name) for name in [*SOURCES.values(), "pretrained_weights.npz"]
        },
        source_url="https://storage.googleapis.com/ailia-models/segment-anything-2.1/",
        versions=dict(
            onnx=onnx.__version__,
            onnxruntime=ort.__version__,
            numpy=np.__version__,
            opencv=cv2.__version__,
            pycocotools=importlib.metadata.version("pycocotools"),
        ),
        implementation_sha256={
            name: digest(ROOT / name)
            for name in (
                "scripts/calibrate_sam2.py",
                "scripts/prepare_sam2.py",
                "scripts/convert_qdq_matmul.py",
                "sam2_video_predictor.py",
                "settings.py",
            )
        },
        offsets_512=OFFSETS,
        calibration="global MinMax; no percentile clipping",
        memory_protocol="5 translated views per image; FP32 tracking; zero-padded memory",
        holdout="COCO val2017 is calibration data, not an independent accuracy test",
    )


def calibration_tensors(model, key):
    constants = {x.name for x in model.graph.initializer}
    types = {
        v.name: v.type.tensor_type.elem_type
        for v in [*model.graph.input, *model.graph.value_info, *model.graph.output]
    }
    names = set()
    for node in model.graph.node:
        if key == "memory_attention" and node.name in EXCLUDED:
            continue
        ops = (
            {"MatMul", "Conv", "Softmax", "LayerNormalization"}
            if key == "image_encoder"
            else {"MatMul"}
        )
        if node.op_type not in ops:
            continue
        inputs = (
            node.input[:1] if node.op_type in {"Softmax", "LayerNormalization"} else node.input[:2]
        )
        for name in [*inputs, *node.output[:1]]:
            if name and name not in constants:
                if types.get(name) != TensorProto.FLOAT:
                    raise ValueError(f"Missing FP32 shape/type information for {name}")
                names.add(name)
    return sorted(names)


class RangeSession:
    """Reduce in the graph; retain only min/max scalars, never activation datasets."""

    def __init__(self, path, key, threads, collect=False):
        model = onnx.load(path)
        self.output_count = len(model.graph.output)
        self.names = calibration_tensors(model, key) if collect else []
        self.ranges = {}
        self.calls = 0
        self.key = key
        for index, name in enumerate(self.names):
            for op in ("ReduceMin", "ReduceMax"):
                output = f"__calibration_{index}_{op}"
                model.graph.node.append(helper.make_node(op, [name], [output], keepdims=0))
                model.graph.output.append(
                    helper.make_tensor_value_info(output, TensorProto.FLOAT, [])
                )
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.enable_profiling = False
        options.log_severity_level = 3
        self.net = ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )

    def run(self, feed):
        if self.key == "memory_attention":
            feed = dict(feed)
            for key, value in feed.items():
                if key.endswith(("_1", "_2")):
                    capacity = 4096 if key.endswith("_1") else 16
                    if len(value) > capacity:
                        raise ValueError("Memory bank capacity exceeded")
                    feed[key] = np.pad(
                        value, [(0, capacity - len(value))] + [(0, 0)] * (value.ndim - 1)
                    )
        outputs = self.net.run(None, feed)
        for index, name in enumerate(self.names):
            lo, hi = (
                float(v)
                for v in outputs[self.output_count + 2 * index : self.output_count + 2 * index + 2]
            )
            if not np.isfinite([lo, hi]).all():
                raise ValueError(f"Nonfinite calibration values: {name}")
            old = self.ranges.get(name, [0.0, 0.0])
            self.ranges[name] = [min(old[0], lo), max(old[1], hi)]
        self.calls += 1
        return outputs[: self.output_count]


class Predictor(SAM2VideoPredictor):
    def __init__(self, weights):
        super().__init__(onnx=False, benchmark=False, legacy=False)
        self.weights_path = weights

    def _load_pretrained_weights(self, version, model_type, num_maskmem, mem_dim, hidden_dim):
        prefix = f"v{version}_{model_type}"
        with np.load(self.weights_path) as data:
            return tuple(
                data[f"{prefix}_{name}"].astype(np.float32)
                for name in ("maskmem_tpos_enc", "no_mem_embed", "no_mem_pos_enc", "no_obj_ptr")
            )


def foreground_point(coco, image_id, height, width):
    annotations = coco.loadAnns(coco.getAnnIds(imgIds=[image_id], iscrowd=False))
    for annotation in sorted(annotations, key=lambda a: (-a["area"], a["id"])):
        mask = coco.annToMask(annotation).astype(np.uint8)
        if mask.any():
            distances = cv2.distanceTransform(np.pad(mask, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
            y, x = np.unravel_index(np.argmax(distances), distances.shape)
            return np.array([[x * 512 / width, y * 512 / height]], np.float32), np.array(
                [1], np.int32
            )
    # Retain all images, including those with no non-crowd instance. These
    # receive an explicit background prompt; no image is silently discarded.
    return np.array([[256, 256]], np.float32), np.array([0], np.int32)


def track_image(image, points, labels, nets, predictor):
    image = cv2.resize(image, (512, 512))
    state = predictor.init_state(version="2.1", model_type="hiera_t", image_size=512)
    for index, (dx, dy) in enumerate(OFFSETS):
        frame = cv2.warpAffine(
            image,
            np.array([[1, 0, dx], [0, 1, dy]], np.float32),
            (512, 512),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        predictor.append_image(state, preprocess_image(frame), 512, 512, nets["image_encoder"])
        if index == 0:
            predictor.add_new_points_or_box(
                state, frame_idx=0, obj_id=1, points=points, labels=labels, **nets
            )
            predictor.propagate_in_video_preflight(state, **nets)
        _, _, logits = predictor.propagate_in_video(state, frame_idx=index, **nets)
        if not np.isfinite(logits).all():
            raise ValueError("Nonfinite FP32 tracking mask")


def calibrate(args, manifest):
    from pycocotools.coco import COCO

    cv2.setNumThreads(1)
    coco = COCO(str(args.coco_annotations))
    checkpoint = args.output / f"ranges_{args.shard_index:03d}.json"
    identity = digest(args.output / "manifest.json")
    state = dict(
        manifest_sha256=identity,
        shard_index=args.shard_index,
        shards=args.workers,
        completed=[],
        ranges={},
        calls={},
    )
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        if state["manifest_sha256"] != identity or state["shards"] != args.workers:
            raise ValueError("Checkpoint does not match this calibration configuration")
    nets = {
        key: RangeSession(
            args.output / "fp32" / f"{key}.onnx",
            key,
            args.threads,
            collect=key in ("image_encoder", "memory_attention"),
        )
        for key in SOURCES
    }
    for key, net in nets.items():
        net.ranges = state["ranges"].get(key, {})
        net.calls = state["calls"].get(key, 0)
    predictor = Predictor(args.source_dir / "pretrained_weights.npz")
    images = manifest["images"][: args.smoke_images or None][args.shard_index :: args.workers]
    completed = set(state["completed"])
    start = time.perf_counter()
    for entry in images:
        if entry["id"] in completed:
            continue
        path = args.coco_images / entry["file_name"]
        if digest(path) != entry["sha256"]:
            raise ValueError(f"Image changed after manifest creation: {path}")
        image = cv2.imread(str(path))
        points, labels = foreground_point(coco, entry["id"], *image.shape[:2])
        track_image(image, points, labels, nets, predictor)
        state["completed"].append(entry["id"])
        state["ranges"] = {key: net.ranges for key, net in nets.items()}
        state["calls"] = {key: net.calls for key, net in nets.items()}
        save_json(checkpoint, state)
        if len(state["completed"]) % 10 == 0 or len(state["completed"]) == len(images):
            print(
                f"shard {args.shard_index}: {len(state['completed'])}/{len(images)} images, "
                f"{time.perf_counter() - start:.1f}s",
                flush=True,
            )


def add_float_boundaries(model, ranges):
    """Keep Softmax/LayerNorm computation and parameters FP32, quantize activation I/O."""
    producers = {o: n for n in model.graph.node for o in n.output}
    consumers = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    def boundary(source, target, name, original):
        lo, hi = ranges[original]
        scale = np.array(max((hi - lo) / 255, 1e-10), np.float32)
        zero = np.array(np.clip(np.rint(-lo / scale), 0, 255), np.uint8)
        model.graph.initializer.extend(
            [
                numpy_helper.from_array(scale, name + "_s"),
                numpy_helper.from_array(zero, name + "_z"),
            ]
        )
        return [
            helper.make_node(
                "QuantizeLinear", [source, name + "_s", name + "_z"], [name + "_q"], axis=0
            ),
            helper.make_node(
                "DequantizeLinear", [name + "_q", name + "_s", name + "_z"], [target], axis=0
            ),
        ]

    nodes = []
    for index, node in enumerate(model.graph.node):
        if node.op_type in ("Softmax", "LayerNormalization"):
            source = node.input[0]
            if source not in producers or producers[source].op_type != "DequantizeLinear":
                target = f"__float_boundary_{index}_in"
                nodes.extend(boundary(source, target, target, source))
                node.input[0] = target
            output = node.output[0]
            if not consumers.get(output) or any(
                n.op_type != "QuantizeLinear" for n in consumers[output]
            ):
                node.output[0] = f"__float_boundary_{index}_out"
                nodes.append(node)
                nodes.extend(boundary(node.output[0], output, node.output[0], output))
                continue
        nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)


def finalize(args, manifest):
    ranges = {"image_encoder": {}, "memory_attention": {}}
    completed = []
    calls = {key: 0 for key in SOURCES}
    for shard in range(args.workers):
        state = json.loads((args.output / f"ranges_{shard:03d}.json").read_text())
        if (
            state["manifest_sha256"] != digest(args.output / "manifest.json")
            or state["shards"] != args.workers
        ):
            raise ValueError("Stale or incompatible calibration shard")
        completed.extend(state["completed"])
        for key in calls:
            calls[key] += state["calls"][key]
        for key in ranges:
            for name, (lo, hi) in state["ranges"][key].items():
                old = ranges[key].get(name, [0, 0])
                ranges[key][name] = [min(lo, old[0]), max(hi, old[1])]
    expected = [e["id"] for e in manifest["images"][: args.smoke_images or None]]
    if sorted(completed) != expected:
        raise ValueError("Calibration incomplete or duplicate images; refusing to generate models")
    if calls["image_encoder"] != 5 * len(expected) or calls["memory_attention"] != 4 * len(
        expected
    ):
        raise ValueError("Unexpected calibration call counts")
    destination = args.output / ("smoke_models" if args.smoke_images else "models")
    destination.mkdir(exist_ok=True)
    statistics = {}
    for key in ranges:
        model = onnx.load(args.output / "fp32" / f"{key}.onnx")
        data = TensorsData(
            CalibrationMethod.MinMax,
            {
                name: (np.array(lo, np.float32), np.array(hi, np.float32))
                for name, (lo, hi) in ranges[key].items()
            },
        )
        quantizer = QDQQuantizer(
            copy.deepcopy(model),
            True,
            False,
            QuantType.QInt8,
            QuantType.QUInt8,
            data,
            [],
            sorted(EXCLUDED) if key == "memory_attention" else [],
            ["MatMul", "Conv"] if key == "image_encoder" else ["MatMul"],
            dict(ActivationSymmetric=False, WeightSymmetric=True, MatMulConstBOnly=False),
        )
        quantizer.quantize_model()
        quantized = quantizer.model.model
        if key == "image_encoder":
            add_float_boundaries(quantized, ranges[key])
        quantizer.model.topological_sort()
        tensors = {t.name: t for t in quantized.graph.initializer}
        for node in quantized.graph.node:
            if node.op_type in ("QuantizeLinear", "DequantizeLinear") and node.input[1] in tensors:
                if numpy_helper.to_array(tensors[node.input[1]]).size == 1:
                    for name in node.input[1:]:
                        if name in tensors and numpy_helper.to_array(tensors[name]).size == 1:
                            del tensors[name].dims[:]
                    attrs = [a for a in node.attribute if a.name != "axis"]
                    del node.attribute[:]
                    node.attribute.extend(attrs)
                    node.attribute.append(helper.make_attribute("axis", 0))
        lowered, statistics[key] = convert(
            quantized, include_shared=True, nxp_ep_benchmark=key == "image_encoder"
        )
        onnx.checker.check_model(lowered)
        onnx.save(lowered, destination / f"{key}_512_integer.onnx")
    for key, name in {
        "prompt_encoder": "prompt_512.onnx",
        "mask_decoder": "decoder_512.onnx",
        "memory_encoder": "memory_encoder_512.onnx",
        "mlp": "mlp_original.onnx",
        "obj_ptr_tpos_proj": "obj_ptr_tpos_proj_original.onnx",
    }.items():
        shutil.copyfile(args.output / "fp32" / f"{key}.onnx", destination / name)
    shutil.copyfile(
        args.source_dir / "pretrained_weights.npz", destination / "pretrained_weights.npz"
    )
    save_json(
        args.output / "completion.json",
        dict(
            production=not args.smoke_images,
            images=len(completed),
            calls=calls,
            conversion=statistics,
            manifest_sha256=digest(args.output / "manifest.json"),
            model_sha256={p.name: digest(p) for p in sorted(destination.iterdir())},
            status="calibrated; deployment requires independent accuracy and device validation",
        ),
    )
    print(f"Completed: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--download-sources", action="store_true", help="Download missing published ONNX files"
    )
    parser.add_argument("--coco-images", type=Path, required=True)
    parser.add_argument("--coco-annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--smoke-images", type=int, default=0, help="Non-production pipeline test only"
    )
    parser.add_argument(
        "--stage", choices=("all", "prepare", "calibrate", "finalize"), default="all"
    )
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if (
        args.threads < 1
        or args.workers < 1
        or not 0 <= args.shard_index < args.workers
        or not 0 <= args.smoke_images <= 5000
    ):
        parser.error("Invalid thread, worker, shard or smoke count")
    if ort.__version__ != "1.23.2":
        parser.error("Use onnxruntime==1.23.2; quantizer API is version-pinned")
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "manifest.json"
    if args.stage in ("all", "prepare"):
        if args.download_sources:
            download_sources(args.source_dir)
        manifest = dataset_manifest(
            args.coco_images, args.coco_annotations, args.source_dir, args.smoke_images
        )
        manifest["execution"] = dict(threads=args.threads, workers=args.workers)
        # JSON roundtrip makes the tuple/list representation canonical for resume.
        manifest = json.loads(json.dumps(manifest))
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("Output directory belongs to a different dataset or protocol")
        save_json(path, manifest)
        prepare(args.source_dir, args.output / "fp32")
    else:
        manifest = json.loads(path.read_text())
        if manifest["smoke_images"] != args.smoke_images:
            raise ValueError("Smoke mode does not match manifest")
        if manifest["execution"] != dict(threads=args.threads, workers=args.workers):
            raise ValueError("Thread/worker configuration does not match manifest")
        for name, expected in manifest["implementation_sha256"].items():
            if digest(ROOT / name) != expected:
                raise ValueError(f"Calibration implementation changed: {name}")
        for name, expected in manifest["source_sha256"].items():
            if digest(args.source_dir / name) != expected:
                raise ValueError(f"Source changed: {name}")
        if digest(args.coco_annotations) != manifest["annotations_sha256"]:
            raise ValueError("COCO annotations changed")
    if args.stage == "all":
        processes = []
        try:
            for index in range(args.workers):
                log = (args.output / f"worker_{index:03d}.log").open("a")
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    *sys.argv[1:],
                    "--stage",
                    "calibrate",
                    "--shard-index",
                    str(index),
                ]
                processes.append(
                    (subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log)
                )
            while processes and any(process.poll() is None for process, _ in processes):
                if any(process.poll() not in (None, 0) for process, _ in processes):
                    raise RuntimeError("Calibration worker failed; see worker logs")
                time.sleep(1)
            if any(process.returncode != 0 for process, _ in processes):
                raise RuntimeError("Calibration worker failed; no final models generated")
        finally:
            for process, log in processes:
                if process.poll() is None:
                    process.terminate()
                process.wait()
                log.close()
        finalize(args, manifest)
    elif args.stage == "calibrate":
        calibrate(args, manifest)
    elif args.stage == "finalize":
        finalize(args, manifest)


if __name__ == "__main__":
    main()
