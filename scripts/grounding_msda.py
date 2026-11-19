"""Rewrite multi-level sampling into standard ONNX single-level reductions.

Keep attention weights normalized over all levels. Only move the final sum
inside each level, then add the per-level outputs. No quantization is changed.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as h
from onnx import numpy_helper as nh


def rewrite(model):
    # Removing the post-sampling Cast is only valid for this FP32 variant.
    if any(x.data_type == onnx.TensorProto.FLOAT16 for x in model.graph.initializer):
        raise ValueError("Expected the FP32 model, not the FP16 variant")
    if any(
        n.op_type == "Cast"
        and any(a.name == "to" and a.i == onnx.TensorProto.FLOAT16 for a in n.attribute)
        for n in model.graph.node
    ):
        raise ValueError("FP16 Cast nodes are not supported by this rewrite")
    producers = {o: n for n in model.graph.node for o in n.output}
    consumers = defaultdict(list)
    for node in model.graph.node:
        for name in node.input:
            consumers[name].append(node)
    constants = {i.name: nh.to_array(i) for i in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    constants[node.output[0]] = nh.to_array(attr.t)

    def attrs(node):
        return {a.name: h.get_attribute_value(a) for a in node.attribute}

    added, replaced = [], set()
    blocks = []
    for concat in model.graph.node:
        if concat.op_type != "Concat" or attrs(concat).get("axis") != -2 or len(concat.input) != 4:
            continue
        samplers = []
        for name in concat.input:
            node = producers.get(name)
            if node is None or node.op_type != "Unsqueeze":
                break
            if list(constants.get(node.input[1], [])) not in [[-2], [3]]:
                break
            node = producers.get(node.input[0])
            if node is not None and node.op_type == "Cast":
                if attrs(node)["to"] != onnx.TensorProto.FLOAT:
                    break
                node = producers.get(node.input[0])
            if node is None or node.op_type != "GridSample":
                break
            a = attrs(node)
            if (
                a.get("mode", b"bilinear") != b"bilinear"
                or a.get("padding_mode", b"zeros") != b"zeros"
                or a.get("align_corners", 0)
            ):
                break
            samplers.append(node)
        if len(samplers) != 4:
            continue
        reshape = [n for n in consumers[concat.output[0]] if n.op_type == "Reshape"]
        if len(reshape) != 1:
            continue
        mul = consumers[reshape[0].output[0]]
        if len(mul) != 1 or mul[0].op_type != "Mul":
            continue
        mul = mul[0]
        reduce = consumers[mul.output[0]]
        if len(reduce) != 1 or reduce[0].op_type != "ReduceSum":
            continue
        reduce = reduce[0]
        if attrs(reduce).get("keepdims", 1) != 0 or list(
            constants.get(reduce.input[1], [])
        ) not in [[-1], [3]]:
            continue
        weights = next(x for x in mul.input if x != reshape[0].output[0])
        prefix = reduce.name + "_per_level"

        def const(suffix, values):
            name = prefix + suffix
            model.graph.initializer.append(nh.from_array(np.asarray(values, dtype=np.int64), name))
            return name

        def op(kind, suffix, inputs, output=None, **attributes):
            name = prefix + suffix
            result = output or name + "_output"
            added.append(h.make_node(kind, inputs, [result], name=name, **attributes))
            return result

        split_shape = const("_weights_shape", [0, 0, 0, 4, -1])
        split_weights = op("Reshape", "_weights", [weights, split_shape])
        expanded_shape = const("_expanded_shape", [0, 0, 0, 1, -1])
        flat_shape = const("_flat_shape", [0, 0, 0, -1])
        axes = const("_sum_axis", [-1])
        results = []
        for level, sampler in enumerate(samplers):
            index = const(f"_index_{level}", level)
            w = op("Gather", f"_weight_{level}", [split_weights, index], axis=3)
            sample = op("GridSample", f"_sample_{level}", list(sampler.input), **attrs(sampler))
            expanded = op("Reshape", f"_expand_{level}", [sample, expanded_shape])
            flat = op("Reshape", f"_flatten_{level}", [expanded, flat_shape])
            weighted = op("Mul", f"_multiply_{level}", [flat, w])
            results.append(op("ReduceSum", f"_reduce_{level}", [weighted, axes], keepdims=0))
        total = results[0]
        for level in range(1, 4):
            total = op(
                "Add",
                f"_add_{level}",
                [total, results[level]],
                output=reduce.output[0] if level == 3 else None,
            )
        replaced.add(reduce.name)
        blocks.append(reduce.name)

    if len(blocks) != 12:
        raise ValueError(f"Expected 12 Grounding DINO MSDA blocks, found {len(blocks)}")
    nodes = [n for n in model.graph.node if n.name not in replaced] + added
    producers = {o: i for i, n in enumerate(nodes) for o in n.output}
    live, ordered = set(), []

    def visit(name):
        index = producers.get(name)
        if index is None or index in live:
            return
        live.add(index)
        node = nodes[index]
        for x in node.input:
            visit(x)
        ordered.append(node)

    for output in model.graph.output:
        visit(output.name)
    del model.graph.node[:]
    model.graph.node.extend(ordered)
    used = {x for n in ordered for x in n.input} | {x.name for x in model.graph.output}
    initializers = [x for x in model.graph.initializer if x.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    # Drop stale annotations for removed tensors; shape inference may regenerate them.
    del model.graph.value_info[:]
    onnx.checker.check_model(model)
    return blocks


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("destination", type=Path)
    args = p.parse_args()
    model = onnx.load(args.source)
    blocks = rewrite(model)
    onnx.save(model, args.destination)
    print(
        f"Rewritten {len(blocks)} MSDA blocks / {4 * len(blocks)} sampling branches: {args.destination}"
    )
