"""Calibrate explicit attention quantizers on all 5000 COCO val2017 images.

Input is the existing COCO-calibrated integer model with DynamicQuantizeLinear
attention. Fixed-weight quantization is preserved. Checkpoints are resumable.
"""

import argparse
import hashlib
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scenes import depth_anything_v2_inputs as depth
from scenes import grounding_dino_inputs as grounding


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def freeze_quantizers(model, ranges):
    nodes = []
    for node in model.graph.node:
        if node.op_type != "DynamicQuantizeLinear":
            nodes.append(node)
            continue
        lo, hi = ranges[node.input[0]]
        if not np.isfinite([lo, hi]).all() or not lo <= 0 <= hi:
            raise ValueError(f"Invalid range for {node.input[0]}")
        scale = np.float32((hi - lo) / 255 if hi > lo else 1)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid scale for {node.input[0]}")
        zero = np.uint8(np.clip(np.rint(-lo / scale), 0, 255))
        model.graph.initializer.extend([
            numpy_helper.from_array(np.asarray(scale), node.output[1]),
            numpy_helper.from_array(np.asarray(zero), node.output[2]),
        ])
        nodes.append(helper.make_node(
            "QuantizeLinear", [node.input[0], node.output[1], node.output[2]],
            [node.output[0]], name=node.name,
        ))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("depth", "grounding"), required=True)
    for name in ("source", "coco-images", "coco-annotations", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--vocab", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.workers, args.threads) < 1:
        parser.error("workers and threads must be positive")
    if args.model == "grounding" and args.vocab is None:
        parser.error("--vocab is required for Grounding DINO")
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    annotation = json.loads(args.coco_annotations.read_text())
    entries = sorted(annotation["images"], key=lambda e: e["id"])
    if len(entries) != 5000 or len({e["id"] for e in entries}) != 5000:
        raise ValueError("Exactly 5000 unique COCO val2017 images are required")
    captions = [None]
    if args.model == "grounding":
        categories = sorted(annotation["categories"], key=lambda c: c["id"])
        captions = [". ".join(c["name"] for c in categories) + ".", grounding.DEFAULT_PROMPT]
    model = onnx.load(args.source, load_external_data=False)
    quantizers = [n for n in model.graph.node if n.op_type == "DynamicQuantizeLinear"]
    expected = 48 if args.model == "depth" else 218
    if len(quantizers) != expected:
        raise ValueError(f"Expected {expected} dynamic quantizers, got {len(quantizers)}")
    sources = list(dict.fromkeys(n.input[0] for n in quantizers))
    external = {}
    for tensor in model.graph.initializer:
        for item in tensor.external_data:
            if item.key == "location":
                name = item.value
                if Path(name).name != name:
                    raise ValueError("External data must use a simple filename")
                if name not in external:
                    external[name] = digest(args.source.parent / name)
                    target = args.output / name
                    if not target.exists() or digest(target) != external[name]:
                        shutil.copyfile(args.source.parent / name, target)
    manifest = dict(
        protocol="attention-coco-val2017-minmax-v1", model=args.model,
        source_sha256=digest(args.source), external_data=external,
        annotations_sha256=digest(args.coco_annotations),
        code_sha256=digest(Path(__file__)),
        versions=dict(onnx=onnx.__version__, ort=ort.__version__, numpy=np.__version__, cv2=cv2.__version__),
        workers=args.workers, threads=args.threads, captions=captions,
        vocab_sha256=digest(args.vocab) if args.vocab else None,
        images=[dict(id=e["id"], file_name=e["file_name"],
                     sha256=digest(args.coco_images / e["file_name"])) for e in entries],
    )
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    path = args.output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Calibration inputs changed; use a new output directory")
    save(path, manifest)
    instrument = onnx.ModelProto()
    instrument.CopyFrom(model)
    del instrument.graph.output[:]
    for index, name in enumerate(sources):
        for op in ("ReduceMin", "ReduceMax"):
            output = f"attention_calibration_{index}_{op}"
            instrument.graph.node.append(helper.make_node(op, [name], [output], keepdims=0))
            instrument.graph.output.append(helper.make_tensor_value_info(output, TensorProto.FLOAT, []))
    instrument_path = args.output / "instrument.onnx"
    onnx.save(instrument, instrument_path)
    onnx.checker.check_model(str(instrument_path))
    del instrument

    def worker(index):
        checkpoint_path = args.output / f"worker_{index}.json"
        checkpoint = (json.loads(checkpoint_path.read_text()) if checkpoint_path.exists()
                      else dict(fingerprint=fingerprint, done=[], ranges={}))
        if checkpoint["fingerprint"] != fingerprint:
            raise ValueError("Checkpoint fingerprint mismatch")
        done, ranges = set(checkpoint["done"]), checkpoint["ranges"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        session = ort.InferenceSession(str(instrument_path), options, providers=["CPUExecutionProvider"])
        text_feeds = [None]
        if args.model == "grounding":
            tokenizer = grounding.Tokenizer(args.vocab)
            text_feeds = [tokenizer.encode(caption)[0] for caption in captions]
        for entry in entries[index::args.workers]:
            if entry["id"] in done:
                continue
            image = cv2.imread(str(args.coco_images / entry["file_name"]))
            if image is None:
                raise ValueError(entry["file_name"])
            for text in text_feeds:
                if text is None:
                    feed = {session.get_inputs()[0].name: depth.preprocess(image)}
                else:
                    samples, _ = grounding.preprocess(image)
                    feed = dict(samples=samples, **text)
                values = session.run(None, feed)
                for j, name in enumerate(sources):
                    lo, hi = float(values[2*j]), float(values[2*j+1])
                    if not np.isfinite([lo, hi]).all():
                        raise ValueError(f"Nonfinite range for {name}")
                    previous = ranges.get(name, [0., 0.])
                    ranges[name] = [min(previous[0], lo), max(previous[1], hi)]
            done.add(entry["id"])
            if len(done) % 10 == 0:
                save(checkpoint_path, dict(fingerprint=fingerprint, done=sorted(done), ranges=ranges))
                print(args.model, index, len(done), "/", len(entries[index::args.workers]), flush=True)
        save(checkpoint_path, dict(fingerprint=fingerprint, done=sorted(done), ranges=ranges))
        return done, ranges

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(worker, range(args.workers)))
    coverage, ranges = set(), {name: [0., 0.] for name in sources}
    for done, values in results:
        if coverage & done:
            raise ValueError("Duplicate calibration coverage")
        coverage.update(done)
        for name in sources:
            ranges[name] = [min(ranges[name][0], values[name][0]), max(ranges[name][1], values[name][1])]
    if coverage != {e["id"] for e in entries}:
        raise ValueError("Incomplete calibration coverage")
    save(args.output / "ranges.json", dict(fingerprint=fingerprint, ranges=ranges))
    output = args.output / args.source.name
    onnx.save(freeze_quantizers(model, ranges), output)
    onnx.checker.check_model(str(output))
    save(args.output / "completion.json", dict(
        production=True, images=len(coverage), captions_per_image=len(captions),
        source_sha256=manifest["source_sha256"], model_sha256=digest(output),
        manifest_sha256=digest(path), ranges_sha256=digest(args.output / "ranges.json"),
        quantizers=expected,
    ))
    print("Completed", output, flush=True)


if __name__ == "__main__":
    main()
