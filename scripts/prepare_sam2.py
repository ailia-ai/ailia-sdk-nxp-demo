"""Adapt the published SAM2.1 Tiny ONNX exports to the demo's 512px grid."""

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper as nh

SOURCES = {
    "image_encoder": "image_encoder_hiera_t_2.1.onnx",
    "prompt_encoder": "prompt_encoder_with_mask_hiera_t_2.1.onnx",
    "mask_decoder": "mask_decoder_hiera_t_2.1.onnx",
    "memory_attention": "memory_attention_6d_hiera_t_2.1.onnx",
    "memory_encoder": "memory_encoder_hiera_t_2.1.onnx",
    "mlp": "mlp_hiera_t_2.1.onnx",
    "obj_ptr_tpos_proj": "obj_ptr_tpos_proj_hiera_t_2.1.onnx",
}


def replace_constants(model, replacements):
    missing = set(replacements)
    for node in model.graph.node:
        if node.name in replacements:
            attr = next(a for a in node.attribute if a.name == "value")
            attr.t.CopyFrom(nh.from_array(np.asarray(replacements[node.name])))
            missing.remove(node.name)
    if missing:
        raise ValueError(f"Unsupported source export; missing constants: {sorted(missing)}")


def prepare(source, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for key, filename in SOURCES.items():
        model = onnx.load(Path(source) / filename)
        if key == "image_encoder":
            dims = model.graph.input[0].type.tensor_type.shape.dim
            dims[2].dim_value = dims[3].dim_value = 512
            resize = next(n for n in model.graph.node if n.name == "/image_encoder/trunk/Resize")
            shape = next(t for t in model.graph.initializer if t.name == resize.input[3])
            shape.CopyFrom(nh.from_array(np.array([1, 96, 128, 128], np.int64), shape.name))
        elif key == "prompt_encoder":
            coords = (np.arange(32, dtype=np.float32) + 0.5) / 32 * 2 - 1
            xx, yy = np.meshgrid(coords, coords)
            replace_constants(
                model,
                {
                    "/Constant_11": np.array(512, np.float32),
                    "/Constant_31": np.array(512, np.float32),
                    "/Constant_66": np.array([32], np.int64),
                    "/Constant_67": np.array([32], np.int64),
                    "/pe_layer/Constant": np.stack([xx, yy], -1),
                },
            )
            for value in model.graph.output:
                if value.name == "dense_pe":
                    value.type.tensor_type.shape.dim[2].dim_value = 32
                    value.type.tensor_type.shape.dim[3].dim_value = 32
        elif key == "mask_decoder":
            replacements = {
                (1, 4096, 8, 16): (1, 1024, 8, 16),
                (1, 256, 64, 64): (1, 256, 32, 32),
                (1, 32, 65536): (1, 32, 16384),
                (1, -1, 256, 256): (1, -1, 128, 128),
            }
            for node in model.graph.node:
                if node.op_type == "Constant":
                    for attr in node.attribute:
                        if attr.name == "value":
                            value = nh.to_array(attr.t)
                            if value.ndim == 1 and tuple(value.tolist()) in replacements:
                                attr.t.CopyFrom(
                                    nh.from_array(
                                        np.array(
                                            replacements[tuple(value.tolist())], dtype=value.dtype
                                        )
                                    )
                                )
            for value in model.graph.input:
                dims = value.type.tensor_type.shape.dim
                if len(dims) == 4:
                    for dim in dims[2:]:
                        dim.dim_value //= 2
        elif key == "memory_encoder":
            for value in [*model.graph.input, *model.graph.output]:
                for dim in value.type.tensor_type.shape.dim[2:]:
                    dim.dim_value //= 2
            replace_constants(
                model,
                {
                    "/position_encoding/Constant": np.arange(1, 33, dtype=np.float32).reshape(
                        1, 32, 1
                    ),
                    "/position_encoding/Constant_3": np.arange(1, 33, dtype=np.float32).reshape(
                        1, 1, 32
                    ),
                    "/position_encoding/Constant_1": np.array([1, 1, 32], np.int64),
                    "/position_encoding/Constant_4": np.array([1, 32, 1], np.int64),
                },
            )
        elif key == "memory_attention":
            for value in model.graph.input:
                dims = value.type.tensor_type.shape.dim
                dims[0].dim_value = (
                    1024
                    if value.name in ("curr", "curr_pos")
                    else 4096
                    if value.name.endswith("_1")
                    else 16
                )
                dims[1].dim_value = 1
            count = 0
            for node in model.graph.node:
                if node.op_type == "Constant":
                    attr = next((a for a in node.attribute if a.name == "value"), None)
                    if attr is not None:
                        value = nh.to_array(attr.t)
                        if value.shape == (1, 1, 4096, 128, 2, 2):
                            value = value.reshape(1, 1, 64, 64, 128, 2, 2)
                            value = value[:, :, :32, :32].copy().reshape(1, 1, 1024, 128, 2, 2)
                            attr.t.CopyFrom(nh.from_array(value))
                            count += 1
            if count != 13:
                raise ValueError(f"Unsupported RoPE export: expected 13 grids, found {count}")
        # Tensor-valued constants become initializers so per-channel weight
        # quantization does not mistake fixed weights for dynamic activations.
        nodes = []
        for node in model.graph.node:
            if (
                node.op_type == "Constant"
                and len(node.attribute) == 1
                and node.attribute[0].name == "value"
            ):
                tensor = onnx.TensorProto.FromString(node.attribute[0].t.SerializeToString())
                tensor.name = node.output[0]
                model.graph.initializer.append(tensor)
            else:
                nodes.append(node)
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        del model.graph.value_info[:]
        onnx.checker.check_model(model)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.save(model, destination / f"{key}.onnx")
