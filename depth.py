"""Resident Metric Depth Small runtime with dynamic MatMul NPU execution."""

import time
from pathlib import Path

import cv2
import numpy as np

from depth_inputs import MODEL, SIZE, preprocess
from engine import Session


class DepthEngine:
    def __init__(self, directory, backend):
        self.directory, self.backend = Path(directory), backend
        self.session = None

    def run(self, path, compare=False):
        image = path if isinstance(path, np.ndarray) else cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read image: {path}")
        feed = {"image": preprocess(image)}
        if self.session is None:
            start = time.perf_counter()
            self.session = Session(
                self.directory / MODEL,
                self.backend,
                npu=self.backend.endswith("_npu"),
                warmup_inputs={"image": np.zeros((1, 3, SIZE, SIZE), np.float32)},
            )
            self.load_ms = (time.perf_counter() - start) * 1000
        times = []
        for _ in range(3 if compare else 1):
            start = time.perf_counter()
            outputs = self.session.run(feed)
            times.append((time.perf_counter() - start) * 1000)
        depth = np.asarray(outputs[0]).reshape(SIZE, SIZE)
        if not np.isfinite(depth).all():
            raise ValueError("Non-finite depth output")
        return dict(
            encoder_ms=float(np.median(times)),
            click_ms=0.0,
            inference_runs_ms=times,
            depth=depth,
            load_ms=self.load_ms,
            depth_min_m=float(depth.min()),
            depth_max_m=float(depth.max()),
            version=self.session.version,
            mode="depth median / 3 runs" if compare else "depth / latest",
        )
