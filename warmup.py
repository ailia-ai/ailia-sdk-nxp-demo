"""Valid synthetic inputs for one inference immediately after model Open.

These fixed shapes match the bundled exports. Dynamic click counts can still
require shape-dependent preparation later; no extra inference is run on click.
"""

import numpy as np

from settings import ENCODER, MEMORY_ATTENTION


def inputs_for_model(name):
    def zeros(*shape):
        return np.zeros(shape, np.float32)

    if name == ENCODER:
        return {"input_image": zeros(1, 3, 512, 512)}
    if name == "prompt_512.onnx":
        return dict(
            coords=zeros(1, 1, 2),
            labels=np.ones((1, 1), np.int32),
            masks=zeros(1, 1, 128, 128),
            masks_enable=np.zeros(1, np.int32),
        )
    if name == "decoder_512.onnx":
        return dict(
            image_embeddings=zeros(1, 256, 32, 32),
            image_pe=zeros(1, 256, 32, 32),
            sparse_prompt_embeddings=zeros(1, 2, 256),
            dense_prompt_embeddings=zeros(1, 256, 32, 32),
            high_res_features1=zeros(1, 32, 128, 128),
            high_res_features2=zeros(1, 64, 64, 64),
        )
    if name == MEMORY_ATTENTION:
        return dict(
            curr=zeros(1024, 1, 256),
            curr_pos=zeros(1024, 1, 256),
            memory_1=zeros(4096, 1, 64),
            memory_pos_1=zeros(4096, 1, 64),
            memory_2=zeros(16, 1, 64),
            memory_pos_2=zeros(16, 1, 64),
            attention_mask_1=np.ones((4096, 1), bool),
            attention_mask_2=np.ones((16, 1), bool),
        )
    if name == "memory_encoder_512.onnx":
        return dict(pix_feat=zeros(1, 256, 32, 32), masks=zeros(1, 1, 512, 512))
    if name in ("mlp_original.onnx", "obj_ptr_tpos_proj_original.onnx"):
        return {"x": zeros(1, 256)}
    raise ValueError(f"No Open-time warm-up inputs defined for {name}")
