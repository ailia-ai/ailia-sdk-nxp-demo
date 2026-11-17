"""Lower private QDQ MatMul pairs to standard ONNX integer arithmetic.

Scales and quantized tensors are preserved (no calibration or re-quantization).
Only scalar activation scales and scalar/last-axis weight scales are accepted.
Other operators, including QDQ Conv and activation Q/DQ, remain unchanged.
By default shared dequantized tensors retain their original floating-point
MatMul arithmetic. --include-shared also lowers those MatMuls while retaining
any dequantization still needed by other consumers.
Integer accumulation changes floating-point rounding: validate model outputs
before deploying the converted model. The source is never overwritten.
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def convert(model, *, include_shared=False, nxp_ep_benchmark=False):
    model = onnx.ModelProto.FromString(model.SerializeToString())
    graph = model.graph
    producers = {name: node for node in graph.node for name in node.output}

    def subgraph_references(node):
        for attr in node.attribute:
            graphs = (
                [attr.g]
                if attr.type == onnx.AttributeProto.GRAPH
                else attr.graphs
                if attr.type == onnx.AttributeProto.GRAPHS
                else []
            )
            for child in graphs:
                yield from (item.name for item in child.output)
                for member in child.node:
                    yield from member.input
                    yield from subgraph_references(member)

    consumers = Counter(name for node in graph.node for name in node.input)
    consumers.update(name for node in graph.node for name in subgraph_references(node))
    exposed = {item.name for item in graph.output}
    constants = {item.name: numpy_helper.to_array(item) for item in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant" and not node.domain:
            for attr in node.attribute:
                if attr.name == "value":
                    constants[node.output[0]] = numpy_helper.to_array(attr.t)
    names = {name for node in graph.node for name in (*node.input, *node.output)}
    names.update(item.name for item in graph.initializer)

    def unique(base):
        name = base
        while name in names:
            name += "_"
        names.add(name)
        return name

    nodes, stats = [], Counter()
    for node in graph.node:
        pair = [producers.get(name) for name in node.input]
        if (
            node.op_type != "MatMul"
            or node.domain
            or len(pair) != 2
            or any(
                d is None or d.op_type != "DequantizeLinear" or d.domain or len(d.input) != 3
                for d in pair
            )
        ):
            nodes.append(node)
            continue
        a, b = pair
        if not include_shared and any(
            consumers[name] != 1 or name in exposed for name in node.input
        ):
            nodes.append(node)
            stats["preserved_shared_dq"] += 1
            continue
        if any(name not in constants for d in pair for name in d.input[1:]):
            raise ValueError(f"Nonconstant quantization parameters: {node.name}")
        sa, za = (constants[x] for x in a.input[1:])
        sb, zb = (constants[x] for x in b.input[1:])
        if (
            sa.size != 1
            or za.size != 1
            or not sb.size
            or not zb.size
            or sb.ndim > 1
            or sa.dtype != np.float32
            or sb.dtype != np.float32
            or za.dtype != np.uint8
            or zb.dtype not in (np.int8, np.uint8)
            or any(not np.isfinite(s).all() or np.any(s <= 0) for s in (sa, sb))
            or any(any(attr.name == "block_size" and attr.i for attr in d.attribute) for d in pair)
        ):
            raise ValueError(f"Unsupported quantization parameters: {node.name}")
        if sb.size > 1:
            axis = next((attr.i for attr in b.attribute if attr.name == "axis"), 1)
            weight = constants.get(b.input[0])
            if weight is None or axis not in (-1, weight.ndim - 1) or weight.shape[-1] != sb.size:
                raise ValueError(f"Expected constant last-axis weights: {node.name}")
        if zb.size != 1 and not np.all(zb == zb.flat[0]):
            raise ValueError(f"Nonuniform weight zero point: {node.name}")
        combined_scale = np.asarray(sa * sb, dtype=np.float32)
        if not np.isfinite(combined_scale).all() or np.any(combined_scale <= 0):
            raise ValueError(f"Unrepresentable combined scale: {node.name}")
        # Scalarize uniform zero points for runtimes that accept only scalars.
        bz = unique(node.output[0] + "_integer_b_zero")
        graph.initializer.append(
            numpy_helper.from_array(np.asarray(zb.flat[0], dtype=zb.dtype), bz)
        )
        integer = unique(node.output[0] + "_integer")
        floating = unique(node.output[0] + "_integer_float")
        activation_scaled = unique(node.output[0] + "_activation_scaled")
        activation = a.input[0]
        result = node.output[0]
        restore_shape = None
        if nxp_ep_benchmark and b.input[0] in constants:
            weight = constants[b.input[0]]
            if weight.ndim != 2:
                raise ValueError(f"Expected rank-two constant weights: {node.name}")

            def shape_constant(label, values):
                name = unique(result + label)
                graph.initializer.append(
                    numpy_helper.from_array(np.asarray(values, np.int64), name)
                )
                return name

            source_shape = unique(result + "_source_shape")
            leading_shape = unique(result + "_leading_shape")
            restore_shape = unique(result + "_restore_shape")
            flattened = unique(result + "_nxp3d_input")
            flat_shape = shape_constant("_flat_shape", [1, -1, weight.shape[0]])
            begin = shape_constant("_shape_begin", [0])
            end = shape_constant("_shape_end", [-1])
            axis = shape_constant("_shape_axis", [0])
            width = shape_constant("_result_width", [weight.shape[1]])
            nodes.extend(
                [
                    helper.make_node("Shape", [activation], [source_shape], name=source_shape),
                    helper.make_node(
                        "Slice",
                        [source_shape, begin, end, axis],
                        [leading_shape],
                        name=leading_shape,
                    ),
                    helper.make_node(
                        "Concat",
                        [leading_shape, width],
                        [restore_shape],
                        name=restore_shape,
                        axis=0,
                    ),
                    helper.make_node(
                        "Reshape", [activation, flat_shape], [flattened], name=flattened
                    ),
                ]
            )
            activation = flattened
        scaled_result = unique(result + "_nxp3d_scaled") if restore_shape else result
        nodes.extend(
            [
                helper.make_node(
                    "MatMulInteger",
                    [activation, b.input[0], a.input[2], bz],
                    [integer],
                    name=integer,
                ),
                helper.make_node(
                    "Cast", [integer], [floating], name=floating, to=TensorProto.FLOAT
                ),
                helper.make_node(
                    "Mul", [floating, a.input[1]], [activation_scaled], name=activation_scaled
                ),
                helper.make_node(
                    "Mul", [activation_scaled, b.input[1]], [scaled_result], name=node.name
                ),
            ]
        )
        if restore_shape:
            nodes.append(
                helper.make_node(
                    "Reshape",
                    [scaled_result, restore_shape],
                    [result],
                    name=unique(result + "_restore"),
                )
            )
        stats["constant_rhs" if b.input[0] in constants else "dynamic_rhs"] += 1
    # Remove newly dead DQ nodes and initializers, including shared chains,
    # without stripping live Q/DQ around non-MatMul operators.
    used = {out.name for out in graph.output}
    kept = []
    for node in reversed(nodes):
        if any(name in used for name in node.output):
            kept.append(node)
            used.update(node.input)
            used.update(subgraph_references(node))
    del graph.node[:]
    graph.node.extend(reversed(kept))
    initializers = [item for item in graph.initializer if item.name in used]
    del graph.initializer[:]
    graph.initializer.extend(initializers)
    onnx.checker.check_model(model)
    return model, dict(stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--include-shared",
        action="store_true",
        help="Also lower MatMul with shared DQ inputs; retain DQ for other consumers. Validate end-to-end accuracy.",
    )
    parser.add_argument(
        "--nxp-ep-benchmark",
        action="store_true",
        help="Flatten fixed-weight inputs to [1,M,K] for NXP EP timing. Does not fix NXP numerical errors; not an accuracy-qualified export.",
    )
    args = parser.parse_args()
    if args.destination.exists() or args.source.resolve() == args.destination.resolve():
        parser.error("Destination must be a new file")
    model, stats = convert(
        onnx.load(args.source),
        include_shared=args.include_shared,
        nxp_ep_benchmark=args.nxp_ep_benchmark,
    )
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.destination)
    print(stats)


if __name__ == "__main__":
    main()
