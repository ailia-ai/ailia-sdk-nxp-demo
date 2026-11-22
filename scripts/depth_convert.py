"""Explicit ONNX quantization: calibrated fixed weights and dynamic attention."""

import numpy as np
import onnx
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper as nh


def prepare(model):
    for dim, size in zip(model.graph.input[0].type.tensor_type.shape.dim, (1, 3, 518, 518)):
        dim.ClearField("dim_param")
        dim.dim_value = size
    return model


def convolution_targets(model):
    """Keep the narrow high-resolution output head in FLOAT; quantize 28 Conv."""
    weights = {v.name: nh.to_array(v) for v in model.graph.initializer}
    selected = []
    for node in model.graph.node:
        if node.op_type != "Conv" or node.domain:
            continue
        if node.name.startswith("/depth_head/output_conv"):
            continue
        weight = weights.get(node.input[1])
        if (weight is None or weight.ndim != 4 or not np.isfinite(weight).all()
                or any(a.i != 1 for a in node.attribute if a.name == "group")):
            raise ValueError(f"Unsupported Depth convolution: {node.name}")
        selected.append(node)
    if len(selected) != 28:
        raise ValueError(f"Expected 28 Depth convolution targets, got {len(selected)}")
    return weights, selected


def quantize_convolutions(model, ranges):
    """Apply calibrated UINT8 activations and channel-wise 8-bit ConvInteger.

    Store signed weights with a +128 UINT8 offset for ORT ConvInteger support.
    The explicit zero point preserves exactly the same centered integer values.
    """
    weights, selected = convolution_targets(model)
    outputs = {n.output[0] for n in selected}
    nodes, constants = [], list(model.graph.initializer)
    used_names = {s for n in model.graph.node for s in (*n.input, *n.output)}
    prefix = "depth_conv_int8_"
    while any(s.startswith(prefix) for s in used_names):
        prefix += "_"

    def constant(name, value):
        constants.append(nh.from_array(np.asarray(value), name))
        return name

    for i, node in enumerate(model.graph.node):
        if node.output[0] not in outputs:
            nodes.append(node)
            continue
        tag = f"{prefix}{i}_"
        lo, hi = ranges[node.input[0]]
        if not np.isfinite([lo, hi]).all() or lo > hi:
            raise ValueError(f"Invalid calibration range: {node.input[0]}")
        lo, hi = min(lo, 0), max(hi, 0)
        xs = np.float32(max((hi - lo) / 255, 1e-12))
        xz = np.uint8(np.clip(np.rint(-lo / xs), 0, 255))
        w = weights[node.input[1]]
        ws = np.maximum(np.max(np.abs(w), axis=(1, 2, 3)) / 127, 1e-12).astype(np.float32)
        qw = (
            np.clip(np.rint(w / ws[:, None, None, None]), -127, 127) + 128
        ).astype(np.uint8)
        attrs = {a.name: h.get_attribute_value(a) for a in node.attribute}
        scaled = tag + "scaled" if len(node.input) > 2 else node.output[0]
        nodes.extend([
            h.make_node(
                "QuantizeLinear",
                [node.input[0], constant(tag + "xs", xs), constant(tag + "xz", xz)],
                [tag + "x"], name=tag + "quantize",
            ),
            h.make_node(
                "ConvInteger",
                [tag + "x", constant(tag + "w", qw), tag + "xz",
                 constant(tag + "wz", np.uint8(128))],
                [tag + "acc"], name=tag + "conv", **attrs,
            ),
            h.make_node("Cast", [tag + "acc"], [tag + "float"], to=T.FLOAT, name=tag + "cast"),
            h.make_node(
                "Mul", [tag + "float", constant(tag + "scale", (xs * ws).reshape(1, -1, 1, 1))],
                [scaled], name=tag + "scale_output",
            ),
        ])
        if len(node.input) > 2:
            nodes.append(h.make_node(
                "Add", [scaled, constant(tag + "bias", weights[node.input[2]].reshape(1, -1, 1, 1))],
                list(node.output), name=tag + "bias_output",
            ))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    used = {s for n in nodes for s in n.input} | {v.name for v in model.graph.output}
    del model.graph.initializer[:]
    model.graph.initializer.extend(v for v in constants if v.name in used)
    del model.graph.value_info[:]
    onnx.checker.check_model(model)
    return model


def targets(model):
    weights = {t.name: nh.to_array(t) for t in model.graph.initializer}
    nodes = [n for n in model.graph.node if n.op_type == "MatMul" and n.input[1] in weights]
    if len(nodes) != 48 or any(weights[n.input[1]].ndim != 2 for n in nodes):
        raise ValueError("Expected 48 fixed matrix weights in Metric Depth Small")
    return weights, nodes


def quantize(model, ranges):
    weights, selected = targets(model)
    names = {n.output[0] for n in selected}
    nodes = []
    for index, node in enumerate(model.graph.node):
        if node.output[0] not in names:
            nodes.append(node)
            continue
        tag = f"depth_int8_{index}_"
        w = weights[node.input[1]]
        ws = np.maximum(np.abs(w).max(axis=0) / 127, 1e-8).astype(np.float32)
        qw = np.clip(np.rint(w / ws), -127, 127).astype(np.int8)
        lo, hi = ranges[node.input[0]]
        if not np.isfinite([lo, hi]).all() or lo > hi:
            raise ValueError("Invalid calibration range")
        lo, hi = min(lo, 0), max(hi, 0)
        scale = np.float32(max((hi - lo) / 255, 1e-8))
        zero = np.uint8(np.clip(np.rint(-lo / scale), 0, 255))
        for key, value in (("w", qw), ("ws", ws), ("wz", np.int8(0)), ("as", scale), ("az", zero)):
            model.graph.initializer.append(nh.from_array(np.asarray(value), tag + key))
        nodes.extend(
            [
                h.make_node(
                    "QuantizeLinear",
                    [node.input[0], tag + "as", tag + "az"],
                    [tag + "a"],
                    name=tag + "quantize",
                ),
                h.make_node(
                    "MatMulInteger",
                    [tag + "a", tag + "w", tag + "az", tag + "wz"],
                    [tag + "i"],
                    name=tag + "matmul",
                ),
                h.make_node("Cast", [tag + "i"], [tag + "f"], to=T.FLOAT, name=tag + "cast"),
                h.make_node("Mul", [tag + "as", tag + "ws"], [tag + "s"], name=tag + "scale"),
                h.make_node("Mul", [tag + "f", tag + "s"], list(node.output), name=tag + "output"),
            ]
        )
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    used = {x for n in nodes for x in n.input} | {v.name for v in model.graph.output}
    initializers = [t for t in model.graph.initializer if t.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    return quantize_dynamic_matmuls(model)


def quantize_dynamic_matmuls(model):
    """Quantize both floating attention operands tensor-wise at inference time.

    No runtime-specific operators or implicit runtime quantization are needed.
    Preserve each original output name so attention scaling and Softmax remain
    unchanged. The existing 48 calibrated fixed-weight operators are untouched.
    """
    weights = {value.name for value in model.graph.initializer}
    selected = [node for node in model.graph.node if node.domain == "" and node.op_type == "MatMul"]
    if len(selected) != 24 or any(any(name in weights for name in node.input) for node in selected):
        raise ValueError("Expected exactly 24 dynamic MatMuls after fixed-weight conversion")
    if next((op.version for op in model.opset_import if op.domain == ""), 0) < 11:
        raise ValueError("DynamicQuantizeLinear requires ONNX opset 11 or later")
    used = {name for node in model.graph.node for name in (*node.input, *node.output)}
    prefix = "depth_dynamic_int8_"
    while any(name.startswith(prefix) for name in used):
        prefix += "_"
    nodes = []
    for index, node in enumerate(model.graph.node):
        if node.domain != "" or node.op_type != "MatMul":
            nodes.append(node)
            continue
        tag = f"{prefix}{index}_"
        for operand, source in zip(("a", "b"), node.input):
            nodes.append(
                h.make_node(
                    "DynamicQuantizeLinear",
                    [source],
                    [tag + operand, tag + operand + "s", tag + operand + "z"],
                    name=tag + operand + "_quantize",
                )
            )
        nodes.extend(
            [
                h.make_node(
                    "MatMulInteger",
                    [tag + "a", tag + "b", tag + "az", tag + "bz"],
                    [tag + "i"],
                    name=tag + "matmul",
                ),
                h.make_node("Cast", [tag + "i"], [tag + "f"], to=T.FLOAT, name=tag + "cast"),
                h.make_node("Mul", [tag + "as", tag + "bs"], [tag + "s"], name=tag + "scale"),
                h.make_node("Mul", [tag + "f", tag + "s"], list(node.output), name=tag + "output"),
            ]
        )
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    return model


def main():
    """Reuse completed calibration ranges without revisiting the dataset."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ranges", type=Path, required=True)
    parser.add_argument("--attention-ranges", type=Path,
                        help="Completed fixed-attention calibration ranges")
    parser.add_argument("--conv-ranges", type=Path,
                        help="Completed 28-convolution COCO calibration ranges")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = quantize(prepare(onnx.load(args.source)), json.loads(args.ranges.read_text()))
    if args.attention_ranges:
        from calibrate_attention import freeze_quantizers
        model = freeze_quantizers(model, json.loads(args.attention_ranges.read_text())["ranges"])
    if args.conv_ranges:
        if not args.attention_ranges:
            parser.error("--conv-ranges requires --attention-ranges")
        model = quantize_convolutions(model, json.loads(args.conv_ranges.read_text())["ranges"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)


if __name__ == "__main__":
    main()
