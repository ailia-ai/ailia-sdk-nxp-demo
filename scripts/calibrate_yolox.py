"""Calibrate YOLOX-S 320 using all COCO val2017 images and MinMax ranges.

Standard ONNX only. Conv scale/bias/Swish epilogues remain FLOAT expressions
that the ailia Neutron backend fuses before writing UINT8. This is full feature
map quantization, not a claim that every runtime kernel uses integer arithmetic.
Swish uses a quantized lookup table. Source hashes, calibration image coverage
and activation parameters are saved alongside the converted model.
"""

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper as nh


def preprocess(path, size):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read {path}")
    ratio = min(size / image.shape[0], size / image.shape[1])
    resized = cv2.resize(image, (int(image.shape[1] * ratio), int(image.shape[0] * ratio)))
    padded = np.full((size, size, 3), 114, np.uint8)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    return np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32)


def convert(source, coco_images, coco_annotations, output, report=None):
    size, image_count = 320, 5000
    expected = "bcb994c44386898f26617a273d98cd678ce6bb8afcc296dc825a0dd29e05a1da"
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError("Source differs from the verified ailia-models YOLOX-S export")
    model = onnx.load(source)
    model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = size
    model.graph.input[0].type.tensor_type.shape.dim[3].dim_value = size
    model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 2100
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model)
    original = list(model.graph.node)
    producer = {o: n for n in original for o in n.output}
    arrays = {v.name: nh.to_array(v) for v in model.graph.initializer}
    # Keep only detection formatting in FLOAT (box coordinates and scores have
    # intentionally different scales). Backbone and neck boundaries use UINT8.
    tail = set()

    def mark_tail(name):
        n = producer.get(name)
        if n is not None and n.op_type in ("Concat", "Reshape", "Transpose"):
            tail.add(name)
            for x in n.input:
                mark_tail(x)

    mark_tail(model.graph.output[0].name)
    swish = {}
    skipped = set()
    for n in original:
        if n.op_type != "Mul":
            continue
        for a, b in (n.input, list(reversed(n.input))):
            sig = producer.get(b)
            conv = producer.get(a)
            if (
                sig
                and sig.op_type == "Sigmoid"
                and sig.input[0] == a
                and conv
                and conv.op_type == "Conv"
            ):
                swish[a] = n.output[0]
                skipped.update((sig.output[0], n.output[0]))
                break
    names = {"images"}
    for n in original:
        if n.output[0] not in skipped and n.output[0] not in tail:
            names.add(swish.get(n.output[0], n.output[0]))
    names.update(swish)
    names = sorted(names)
    instrument = copy.deepcopy(model)
    del instrument.graph.output[:]
    for i, name in enumerate(names):
        for op in ("Min", "Max"):
            out = f"full_range_{i}_{op}"
            instrument.graph.node.append(h.make_node("Reduce" + op, [name], [out], keepdims=0))
            instrument.graph.output.append(h.make_tensor_value_info(out, T.FLOAT, []))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        instrument.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    available = sorted(coco_images.glob("*.jpg"))
    images = available[:image_count]
    if len(available) != image_count:
        raise ValueError("Exactly 5000 COCO val2017 images are required")
    annotation = json.loads(coco_annotations.read_text())
    if len(annotation["images"]) != 5000 or {p.name for p in images} != {
        v["file_name"] for v in annotation["images"]
    }:
        raise ValueError("COCO validation manifest mismatch")
    ranges = np.tile([np.inf, -np.inf], (len(names), 1))
    for image in images:
        values = np.asarray(session.run(None, {"images": preprocess(image, size)})).reshape(-1, 2)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite calibration values: {image}")
        ranges[:, 0] = np.minimum(ranges[:, 0], values[:, 0])
        ranges[:, 1] = np.maximum(ranges[:, 1], values[:, 1])
        print(size, image.name, flush=True)
    del session
    # Layout/pooling/concat preserve the quantization scale. Residual branches
    # share one scale too, permitting exact integer addition before saturation.
    parent = {name: name for name in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for n in original:
        if n.output[0] in tail or n.output[0] in skipped:
            continue
        if n.op_type in ("Concat", "Add", "Slice", "MaxPool", "Resize", "Reshape", "Transpose"):
            operands = list(n.input) if n.op_type in ("Concat", "Add") else [n.input[0]]
            for x in operands:
                parent[find(x)] = find(n.output[0])
    grouped = {}
    for name, (lo, hi) in zip(names, ranges):
        key = find(name)
        a, b = grouped.get(key, (0.0, 0.0))
        grouped[key] = min(a, float(lo)), max(b, float(hi))
    params = {}
    for name in names:
        lo, hi = grouped[find(name)]
        scale = np.float32(max((hi - lo) / 255, 1e-12))
        zero = np.uint8(np.clip(np.rint(-lo / scale), 0, 255))
        params[name] = scale, zero
    nodes = []
    constants = list(model.graph.initializer)
    serial = 0

    def const(value):
        nonlocal serial
        serial += 1
        name = f"full_const_{serial}"
        constants.append(nh.from_array(np.asarray(value), name))
        return name

    def op(kind, inputs, output=None, **attrs):
        nonlocal serial
        serial += 1
        out = output or f"full_value_{serial}"
        nodes.append(h.make_node(kind, inputs, [out], name=out, **attrs))
        return out

    def quant(value, name):
        scale, zero = params[name]
        return op("QuantizeLinear", [value, const(scale), const(zero)], name + "_u8")

    dequantized = {}

    def dequant(name):
        if name not in dequantized:
            scale, zero = params[name]
            dequantized[name] = op("DequantizeLinear", [name + "_u8", const(scale), const(zero)])
        return dequantized[name]

    quant("images", "images")
    for n in original:
        out = n.output[0]
        if out in skipped:
            continue
        attrs = {a.name: h.get_attribute_value(a) for a in n.attribute}
        if out in tail:
            inputs = [dequant(x) if x in params else x for x in n.input]
            op(n.op_type, inputs, out, **attrs)
        elif n.op_type == "Conv":
            w = arrays[n.input[1]]
            ws = np.maximum(np.max(np.abs(w), axis=(1, 2, 3)) / 127, 1e-12).astype(np.float32)
            qw = np.clip(np.rint(w / ws[:, None, None, None]), -127, 127).astype(np.int8)
            xs, xz = params[n.input[0]]
            acc = op("ConvInteger", [n.input[0] + "_u8", const(qw), const(xz)], **attrs)
            value = op("Cast", [acc], to=T.FLOAT)
            value = op("Mul", [value, const((xs * ws).reshape(1, -1, 1, 1))])
            if len(n.input) > 2:
                value = op("Add", [value, const(arrays[n.input[2]].reshape(1, -1, 1, 1))])
            if out in swish:
                quant(value, out)
                scale, zero = params[out]
                x = (np.arange(256, dtype=np.float32) - int(zero)) * scale
                y = x / (1 + np.exp(-np.clip(x, -80, 80)))
                ys, yz = params[swish[out]]
                table = np.clip(np.rint(y / ys) + int(yz), 0, 255).astype(np.uint8)
                index = op("Cast", [out + "_u8"], to=T.INT32)
                op("Gather", [const(table), index], swish[out] + "_u8", axis=0)
                continue
            quant(value, out)
        elif n.op_type == "Add":
            # qY = saturate(qA + qB - zero), because all three share scale/zero.
            # A 511-entry lookup avoids float Clip fallbacks and uint8 overflow.
            a, b = [op("Cast", [x + "_u8"], to=T.INT32) for x in n.input]
            index = op("Add", [a, b])
            zero = int(params[out][1])
            table = np.clip(np.arange(511) - zero, 0, 255).astype(np.uint8)
            op("Gather", [const(table), index], out + "_u8", axis=0)
        elif n.op_type == "Sigmoid":
            scale, zero = params[n.input[0]]
            x = (np.arange(256, dtype=np.float32) - int(zero)) * scale
            y = 1 / (1 + np.exp(-np.clip(x, -80, 80)))
            ys, yz = params[out]
            table = np.clip(np.rint(y / ys) + int(yz), 0, 255).astype(np.uint8)
            index = op("Cast", [n.input[0] + "_u8"], to=T.INT32)
            op("Gather", [const(table), index], out + "_u8", axis=0)
        elif n.op_type in ("Concat", "Slice", "MaxPool", "Resize", "Reshape", "Transpose"):
            inputs = [x + "_u8" if x in params else x for x in n.input]
            op(n.op_type, inputs, out + "_u8", **attrs)
        else:
            raise ValueError(f"Unsupported operation: {n.op_type}")
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    used = {x for n in nodes for x in n.input}
    del model.graph.initializer[:]
    model.graph.initializer.extend(v for v in constants if v.name in used)
    del model.graph.value_info[:]
    model = onnx.version_converter.convert_version(model, 13)
    onnx.checker.check_model(model, full_check=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    record = {
        "calibration": "COCO val2017 (all 5000 images)",
        "method": "MinMax",
        "model_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "annotations_sha256": hashlib.sha256(coco_annotations.read_bytes()).hexdigest(),
        "versions": {
            "numpy": np.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
        "swish_lut": True,
        "images": [p.name for p in images],
        "operators": dict(Counter(n.op_type for n in model.graph.node)),
        "parameters": {k: [float(s), int(z)] for k, (s, z) in params.items()},
        "note": "UINT8 feature maps; Conv FLOAT epilogues fuse on NPU. Kernel fallbacks require profiling.",
    }
    report = report or output.with_suffix(".calibration.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(record, indent=2))
    print(output, record["operators"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("source", "coco-images", "coco-annotations", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--report", type=Path, help="Calibration record destination")
    args = parser.parse_args()
    convert(args.source, args.coco_images, args.coco_annotations, args.output, args.report)
