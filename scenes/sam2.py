"""SAM 2 image segmentation scene."""

import time
from pathlib import Path

import cv2
import numpy as np

from runtime import SessionCache
from settings import ENCODER, preprocess_image


class Engine:
    def __init__(self, directory, backend, cache=None):
        self.backend = backend
        self.directory = Path(directory)
        # Shared settings from the measured SAM512 QDQ configuration.
        start = time.perf_counter()
        cache = cache if cache is not None else SessionCache(backend)
        self.encoder = cache.get(self.directory / ENCODER, backend, backend.endswith("_npu"))
        # Explicitly CPU on both runtimes: dynamic point prompts do not trigger
        # NPU recompilation. UI labels identify this hybrid execution policy.
        self.prompt = cache.get(
            self.directory / "sam2.1_tiny_prompt_encoder_512_fp32.onnx", backend
        )
        self.decoder = cache.get(self.directory / "sam2.1_tiny_mask_decoder_512_fp32.onnx", backend)
        self.no_mem = (
            np.load(self.directory / "sam2.1_tiny_weights.npz")["v2.1_hiera_t_no_mem_embed"]
            .astype(np.float32)
            .reshape(1, 256, 1, 1)
        )
        self.load_ms = (time.perf_counter() - start) * 1000
        self.path = None
        self.features = None

    def prepare(self, path):
        self.path = None if isinstance(path, np.ndarray) else str(path)
        image = path if isinstance(path, np.ndarray) else cv2.imread(self.path)
        if image is None:
            raise ValueError("Cannot read image: " + self.path)
        start = time.perf_counter()
        self.input = preprocess_image(image)[None]
        self.preprocess_ms = (time.perf_counter() - start) * 1000
        return self.encode()

    def encode(self):
        start = time.perf_counter()
        values = self.encoder.run({"input_image": self.input})
        self.encoder_ms = (time.perf_counter() - start) * 1000
        if len(values) != 7 or not all(np.isfinite(v).all() for v in values):
            raise RuntimeError("Unexpected/non-finite encoder outputs")
        self.features = dict(
            image_embeddings=values[6] + self.no_mem,
            high_res_features1=values[4].copy(),
            high_res_features2=values[5].copy(),
        )
        return self.encoder_ms

    def segment(self, points):
        if not points or len(points) > 64:
            raise ValueError("Use 1 to 64 foreground/background clicks")
        points = np.asarray(points, np.float32)
        if not np.isfinite(points).all() or (points[:, :2] < 0).any() or (points[:, :2] >= 1).any():
            raise ValueError("Click outside the image")
        start = time.perf_counter()
        p0 = time.perf_counter()
        sparse, dense, pe = self.prompt.run(
            dict(
                coords=points[None, :, :2] * 512,
                labels=points[None, :, 2].astype(np.int32),
                masks=np.zeros((1, 1, 128, 128), np.float32),
                masks_enable=np.array([0], np.int32),
            )
        )
        prompt_ms = (time.perf_counter() - p0) * 1000
        p0 = time.perf_counter()
        masks, scores, *_ = self.decoder.run(
            dict(
                self.features,
                image_pe=pe,
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
            )
        )
        decoder_ms = (time.perf_counter() - p0) * 1000
        if not np.isfinite(masks).all() or not np.isfinite(scores).all():
            raise RuntimeError("Non-finite decoder output")
        index = int(np.argmax(scores[0, 1:])) + 1
        mask = (masks[0, index] > 0).astype(np.uint8)
        return dict(
            mask=mask,
            score=float(scores[0, index]),
            prompt_ms=prompt_ms,
            decoder_ms=decoder_ms,
            click_ms=(time.perf_counter() - start) * 1000,
        )
