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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = quantize(prepare(onnx.load(args.source)), json.loads(args.ranges.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)


if __name__ == "__main__":
    main()
