"""Offline FP32/INT8 conversion using standard ONNX operators only."""

from pathlib import Path

import numpy as np
import onnx
from grounding_msda import rewrite
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper as nh


def promote_fp32(model):
    def tensor(value):
        if value.data_type == T.FLOAT16:
            value.CopyFrom(nh.from_array(nh.to_array(value).astype(np.float32), value.name))

    for value in model.graph.initializer:
        tensor(value)
    for value in [*model.graph.input, *model.graph.output, *model.graph.value_info]:
        if value.type.tensor_type.elem_type == T.FLOAT16:
            value.type.tensor_type.elem_type = T.FLOAT
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                tensor(attr.t)
            if node.op_type == "Cast" and attr.name == "to" and attr.i == T.FLOAT16:
                attr.i = T.FLOAT
    return model


def targets(model):
    constants = {value.name: nh.to_array(value) for value in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Identity" and node.input[0] in constants:
            constants[node.output[0]] = constants[node.input[0]]
    selected = [n for n in model.graph.node if n.op_type == "MatMul" and n.input[1] in constants]
    if len(selected) != 347:
        raise ValueError(f"Expected 347 fixed-weight MatMuls, found {len(selected)}")
    return constants, selected


def quantize(model, ranges):
    constants, selected = targets(model)
    outputs = {n.output[0] for n in selected}
    quantized, nodes = {}, []

    def init(name, value):
        model.graph.initializer.append(nh.from_array(np.asarray(value), name))
        return name

    def add(op, name, inputs, output=None, **attrs):
        result = output or name + "_out"
        nodes.append(h.make_node(op, inputs, [result], name=name, **attrs))
        return result

    for index, node in enumerate(model.graph.node):
        if node.output[0] not in outputs:
            nodes.append(node)
            continue
        tag = f"gdino_int8_{index}"
        name = node.input[0]
        if name not in quantized:
            lo, hi = ranges[name]
            if not np.isfinite([lo, hi]).all():
                raise ValueError(f"Non-finite range: {name}")
            lo, hi = min(lo, 0), max(hi, 0)
            scale = np.float32(max((hi - lo) / 255, 1e-8))
            sa = init(tag + "_sa", scale)
            za = init(tag + "_za", np.uint8(np.clip(np.rint(-lo / scale), 0, 255)))
            qa = add("QuantizeLinear", tag + "_quantize", [name, sa, za])
            quantized[name] = qa, sa, za
        qa, sa, za = quantized[name]
        weight = constants[node.input[1]].astype(np.float32)
        if weight.ndim != 2:
            raise ValueError(f"Non-matrix constant weight: {node.name}")
        k, n = weight.shape
        scales = np.maximum(np.abs(weight).max(axis=0) / 127, np.float32(1e-8)).astype(np.float32)
        qw = np.clip(np.rint(weight / scales), -127, 127).astype(np.int8)
        pk, pn = max(128, (k + 63) // 64 * 64), max(128, n)
        qw = init(tag + "_weight", np.pad(qw, ((0, pk - k), (0, pn - n))))
        sb = init(tag + "_sb", scales)
        zb = init(tag + "_zb", np.int8(0))
        zero = init(tag + "_zero", np.array([0], np.int64))
        last = init(tag + "_last", np.array([-1], np.int64))
        width = init(tag + "_width", np.array([n], np.int64))
        shape = add("Shape", tag + "_shape", [qa])
        prefix = add("Slice", tag + "_prefix", [shape, zero, last, zero])
        restored = add("Concat", tag + "_restored_shape", [prefix, width], axis=0)
        flat = add(
            "Reshape",
            tag + "_flatten",
            [qa, init(tag + "_flat_shape", np.array([1, -1, k], np.int64))],
        )
        if pk != k:
            pads = init(tag + "_pads", np.array([0, 0, 0, 0, 0, pk - k], np.int64))
            flat = add("Pad", tag + "_pad", [flat, pads, za])
        value = add("MatMulInteger", tag + "_integer", [flat, qw, za, zb])
        if pn != n:
            value = add(
                "Slice",
                tag + "_columns",
                [value, zero, width, init(tag + "_axis", np.array([2], np.int64))],
            )
        value = add("Cast", tag + "_float", [value], to=T.FLOAT)
        value = add("Mul", tag + "_scale_a", [value, sa])
        value = add("Mul", tag + "_scale_b", [value, sb])
        add("Reshape", tag + "_restore", [value, restored], output=node.output[0])
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    rewrite(model)  # Also removes unused original weights and dead shape nodes.
    return quantize_dynamic(model)


def quantize_dynamic(model):
    """Make all 109 remaining floating MatMuls explicitly tensor-wise UINT8."""
    selected = [n for n in model.graph.node if n.op_type == "MatMul" and n.domain == ""]
    if len(selected) != 109:
        raise ValueError(f"Expected 109 dynamic MatMuls, found {len(selected)}")
    if next((op.version for op in model.opset_import if op.domain == ""), 0) < 11:
        raise ValueError("DynamicQuantizeLinear requires opset 11 or later")
    used = {s for n in model.graph.node for s in (*n.input, *n.output)}
    prefix = "gdino_dynamic_int8_"
    while any(s.startswith(prefix) for s in used):
        prefix += "_"
    nodes = []
    for index, node in enumerate(model.graph.node):
        if node.op_type != "MatMul" or node.domain != "":
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
    return model


def save_sharded(model, path, limit=96 * 1024 * 1024):
    """Standard ONNX external data, keeping every Git artifact below 100 MB."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shards, stream, offset = [], None, 0
    try:
        for tensor in model.graph.initializer:
            if len(tensor.raw_data) < 1024:
                continue
            data = tensor.raw_data
            if len(data) > limit:
                raise ValueError(f"Initializer exceeds shard limit: {tensor.name}")
            if stream is None or offset + len(data) > limit:
                if stream:
                    stream.close()
                shard = path.with_name(f"{path.stem}.data{len(shards):02d}")
                shards.append(shard)
                stream, offset = shard.open("wb"), 0
            onnx.external_data_helper.set_external_data(tensor, shards[-1].name, offset, len(data))
            stream.write(data)
            offset += len(data)
            tensor.ClearField("raw_data")
    finally:
        if stream:
            stream.close()
    onnx.save_model(model, path)
    onnx.checker.check_model(str(path))
    return [path, *shards]


def main():
    """Convert the upstream model with completed COCO calibration ranges."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ranges", type=Path, required=True)
    parser.add_argument("--attention-ranges", type=Path,
                        help="Completed fixed-attention calibration ranges")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = quantize(promote_fp32(onnx.load(args.source)), json.loads(args.ranges.read_text()))
    if args.attention_ranges:
        from calibrate_attention import freeze_quantizers
        model = freeze_quantizers(model, json.loads(args.attention_ranges.read_text())["ranges"])
    save_sharded(model, args.output)


if __name__ == "__main__":
    main()
