"""Full COCO val2017 MinMax calibration, resumable with verified image coverage."""

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as T
from onnx import helper as h

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from depth_convert import prepare, quantize, targets

from depth_inputs import MODEL, REVISION, preprocess


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(data)
    return result.hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "coco-images", "coco-annotations", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    entries = sorted(json.loads(args.coco_annotations.read_text())["images"], key=lambda x: x["id"])
    if len(entries) != 5000 or len({e["id"] for e in entries}) != 5000:
        raise ValueError("Exactly 5000 unique COCO val2017 images are required")
    if {p.name for p in args.coco_images.glob("*.jpg")} != {e["file_name"] for e in entries}:
        raise ValueError("Image directory does not match all COCO val2017 annotations")
    manifest = dict(
        revision=REVISION,
        protocol="depth-coco-val2017-minmax-v1",
        source_sha256=digest(args.source),
        annotations_sha256=digest(args.coco_annotations),
        workers=args.workers,
        input_shape=[1, 3, 518, 518],
        preprocessing="RGB cubic square resize, ImageNet normalization",
        versions=dict(
            numpy=np.__version__, onnx=onnx.__version__, ort=ort.__version__, cv2=cv2.__version__
        ),
        code_sha256={
            p.name: digest(p)
            for p in [
                Path(__file__),
                Path(__file__).with_name("depth_convert.py"),
                Path(__file__).parents[1] / "depth_inputs.py",
            ]
        },
        images=[
            dict(
                id=e["id"],
                file_name=e["file_name"],
                sha256=digest(args.coco_images / e["file_name"]),
            )
            for e in entries
        ],
    )
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    mp = args.output / "manifest.json"
    if mp.exists() and json.loads(mp.read_text()) != manifest:
        raise ValueError("Calibration changed; use a new output directory")
    save(mp, manifest)
    model = prepare(onnx.load(args.source))
    _, selected = targets(model)
    activations = list(dict.fromkeys(n.input[0] for n in selected))
    del model.graph.output[:]
    for index, name in enumerate(activations):
        for op in ("ReduceMin", "ReduceMax"):
            output = f"depth_calibration_{index}_{op}"
            model.graph.node.append(h.make_node(op, [name], [output], keepdims=0))
            model.graph.output.append(h.make_tensor_value_info(output, T.FLOAT, []))
    onnx.save(model, args.output / "instrument.onnx")
    del model

    def worker(index):
        path = args.output / f"worker_{index}.json"
        checkpoint = (
            json.loads(path.read_text())
            if path.exists()
            else dict(fingerprint=fingerprint, done=[], ranges={})
        )
        if checkpoint["fingerprint"] != fingerprint:
            raise ValueError("Checkpoint fingerprint mismatch")
        done, ranges = set(checkpoint["done"]), checkpoint["ranges"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.log_severity_level = 3
        session = ort.InferenceSession(
            str(args.output / "instrument.onnx"), options, providers=["CPUExecutionProvider"]
        )
        shard = entries[index :: args.workers]
        for entry in shard:
            if entry["id"] in done:
                continue
            image = cv2.imread(str(args.coco_images / entry["file_name"]))
            if image is None or image.shape[:2] != (entry["height"], entry["width"]):
                raise ValueError(f"Invalid image: {entry['file_name']}")
            values = session.run(None, {"image": preprocess(image)})
            for i, name in enumerate(activations):
                lo, hi = float(values[2 * i]), float(values[2 * i + 1])
                if not np.isfinite([lo, hi]).all():
                    raise ValueError("Non-finite calibration activation")
                old = ranges.get(name, [lo, hi])
                ranges[name] = [min(old[0], lo), max(old[1], hi)]
            done.add(entry["id"])
            if len(done) % 25 == 0 or entry is shard[-1]:
                save(path, dict(fingerprint=fingerprint, done=sorted(done), ranges=ranges))
                print(f"worker {index}: {len(done)}/{len(shard)}", flush=True)
        return sorted(done), ranges

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(worker, range(args.workers)))
    completed = sorted(x for done, _ in results for x in done)
    if completed != [e["id"] for e in entries]:
        raise ValueError("Incomplete or duplicate calibration coverage")
    ranges = {}
    for _, part in results:
        for name, (lo, hi) in part.items():
            old = ranges.get(name, [lo, hi])
            ranges[name] = [min(old[0], lo), max(old[1], hi)]
    save(args.output / "ranges.json", ranges)
    model = quantize(prepare(onnx.load(args.source)), ranges)
    onnx.save(model, args.output / MODEL)
    save(
        args.output / "completion.json",
        dict(
            production=True,
            calibrated_images=5000,
            revision=REVISION,
            method="global MinMax",
            fixed_matmuls=48,
            activation="static UINT8 tensor-wise",
            weight="INT8 channel-wise",
            dynamic_attention="24 floating MatMuls; ailia runtime dynamic INT8 NPU enabled",
            source_sha256=digest(args.source),
            manifest_sha256=digest(mp),
            ranges_sha256=digest(args.output / "ranges.json"),
            files={MODEL: digest(args.output / MODEL)},
        ),
    )
    print("Complete: all 5000 images calibrated", flush=True)


if __name__ == "__main__":
    main()
