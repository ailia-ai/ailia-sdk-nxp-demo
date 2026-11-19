"""COCO val2017 global MinMax calibration and standard-ONNX Grounding DINO export."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as T
from onnx import helper as h

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grounding_convert import promote_fp32, quantize, save_sharded, targets

from grounding_inputs import MODEL, REVISION, Tokenizer, preprocess

PROTOCOL = "grounding-coco-val2017-all80-minmax-v1"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def save_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def worker(args):
    cv2.setNumThreads(1)
    manifest = json.loads((args.output / "manifest.json").read_text())
    config = json.loads((args.output / "instrument.json").read_text())
    path = args.output / f"worker_{args.worker}.json"
    checkpoint = (
        json.loads(path.read_text())
        if path.exists()
        else dict(done=[], ranges={}, fingerprint=manifest["fingerprint"])
    )
    if checkpoint["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Checkpoint belongs to a different calibration")
    done = set(checkpoint["done"])
    ranges = checkpoint["ranges"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    options.enable_profiling = False
    session = ort.InferenceSession(
        str(args.output / "instrument.onnx"), options, providers=["CPUExecutionProvider"]
    )
    text, _ = Tokenizer(args.vocab).encode(manifest["caption"])
    entries = manifest["images"][: args.smoke_images or None][args.worker :: args.workers]
    start = time.perf_counter()
    for entry in entries:
        if entry["id"] in done:
            continue
        image = cv2.imread(str(args.coco_images / entry["file_name"]))
        if image is None:
            raise ValueError(f"Unreadable image: {entry['file_name']}")
        samples, _ = preprocess(image)
        values = session.run(None, dict(samples=samples, **text))
        for i, name in enumerate(config["activations"]):
            lo, hi = float(values[2 * i]), float(values[2 * i + 1])
            if not np.isfinite([lo, hi]).all():
                raise ValueError(f"Non-finite calibration value for {name}")
            old = ranges.get(name, [lo, hi])
            ranges[name] = [min(old[0], lo), max(old[1], hi)]
        done.add(entry["id"])
        if len(done) % 10 == 0 or entry is entries[-1]:
            save_json(
                path, dict(done=sorted(done), ranges=ranges, fingerprint=manifest["fingerprint"])
            )
            print(
                f"worker {args.worker}: {len(done)}/{len(entries)} images, {time.perf_counter() - start:.1f}s",
                flush=True,
            )


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    annotations = json.loads(args.coco_annotations.read_text())
    entries = sorted(annotations["images"], key=lambda x: x["id"])
    if len(entries) != 5000 or len({e["id"] for e in entries}) != 5000:
        raise ValueError("Exactly 5,000 unique COCO val2017 images are required")
    if {p.name for p in args.coco_images.glob("*.jpg")} != {e["file_name"] for e in entries}:
        raise ValueError("Image files do not match the complete COCO val2017 annotations")
    caption = (
        ". ".join(c["name"] for c in sorted(annotations["categories"], key=lambda c: c["id"])) + "."
    )
    _, tokens = Tokenizer(args.vocab).encode(caption)
    records = []
    for entry in entries:
        path = args.coco_images / entry["file_name"]
        image = cv2.imread(str(path))
        if image is None or image.shape[:2] != (entry["height"], entry["width"]):
            raise ValueError(f"Invalid image: {path}")
        records.append(dict(id=entry["id"], file_name=entry["file_name"], sha256=digest(path)))
    manifest = dict(
        protocol=PROTOCOL,
        revision=REVISION,
        production=not args.smoke_images,
        images=records,
        caption=caption,
        tokens=len(tokens),
        workers=args.workers,
        smoke_images=args.smoke_images,
        input_shape=[1, 3, 400, 599],
        preprocessing="RGB letterbox, ImageNet normalization; grounding_inputs.preprocess",
        source_sha256=digest(args.source),
        vocabulary_sha256=digest(args.vocab),
        annotations_sha256=digest(args.coco_annotations),
        converter_sha256={
            p.name: digest(p)
            for p in [
                Path(__file__),
                Path(__file__).with_name("grounding_convert.py"),
                Path(__file__).with_name("grounding_msda.py"),
                Path(__file__).parents[1] / "grounding_inputs.py",
            ]
        },
        versions=dict(
            numpy=np.__version__, onnx=onnx.__version__, ort=ort.__version__, cv2=cv2.__version__
        ),
    )
    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Calibration configuration changed; use a new output directory")
    save_json(manifest_path, manifest)
    if not (args.output / "instrument.json").exists():
        model = promote_fp32(onnx.load(args.source))
        _, selected = targets(model)
        activations = list(dict.fromkeys(n.input[0] for n in selected))
        del model.graph.output[:]
        for i, name in enumerate(activations):
            for op in ["ReduceMin", "ReduceMax"]:
                output = f"gdino_calibration_{i}_{op}"
                model.graph.node.append(h.make_node(op, [name], [output], keepdims=0))
                model.graph.output.append(h.make_tensor_value_info(output, T.FLOAT, []))
        onnx.save(model, args.output / "instrument.onnx")
        save_json(args.output / "instrument.json", dict(activations=activations))
        del model
    processes = []
    for index in range(args.workers):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *sys.argv[1:],
            "--worker",
            str(index),
        ]
        processes.append(subprocess.Popen(command))
    codes = [process.wait() for process in processes]
    if any(codes):
        raise RuntimeError(f"Calibration workers failed: {codes}; rerun to resume")
    completed, ranges = [], {}
    for index in range(args.workers):
        checkpoint = json.loads((args.output / f"worker_{index}.json").read_text())
        completed.extend(checkpoint["done"])
        for name, (lo, hi) in checkpoint["ranges"].items():
            old = ranges.get(name, [lo, hi])
            ranges[name] = [min(lo, old[0]), max(hi, old[1])]
    expected = [e["id"] for e in entries[: args.smoke_images or None]]
    if sorted(completed) != expected:
        raise ValueError("Incomplete or duplicate calibration image coverage")
    save_json(args.output / "ranges.json", ranges)
    model = quantize(promote_fp32(onnx.load(args.source)), ranges)
    files = save_sharded(model, args.output / "models" / MODEL)
    save_json(
        args.output / "completion.json",
        dict(
            production=not args.smoke_images,
            calibrated_images=len(completed),
            caption=caption,
            method="global MinMax",
            activation="UINT8 tensor-wise",
            weight="INT8 channel-wise",
            fixed_matmuls=347,
            msda_branches=48,
            manifest_sha256=digest(manifest_path),
            ranges_sha256=digest(args.output / "ranges.json"),
            files={p.name: digest(p) for p in files},
        ),
    )
    print("Calibration and conversion complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path(__file__).parents[1] / "models/grounding_tokenizer/vocab.txt",
    )
    parser.add_argument("--coco-images", type=Path, required=True)
    parser.add_argument("--coco-annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument(
        "--smoke-images", type=int, default=0, help="Non-production smoke test only"
    )
    parser.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.workers < 1 or args.threads < 1 or args.smoke_images < 0:
        parser.error("Worker/thread counts must be positive; smoke count must be nonnegative")
    worker(args) if args.worker is not None else main(args)
