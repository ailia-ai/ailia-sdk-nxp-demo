"""ViT-B/16 INT4 classification shared by the GUI and benchmark."""

import time
from pathlib import Path

import cv2
import numpy as np

from scenes.vit_labels import imagenet_category

MODEL = "vit_b16_224_int4.onnx"
REVISION = "vit-b16-224-matmulnbits-int4-v1"


def preprocess(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = cv2.resize(rgb, (224, 224)).astype(np.float32) / 127.5 - 1.0
    return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])


def classify(logits):
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    if logits.size != 1000 or not np.isfinite(logits).all():
        raise ValueError("Invalid ViT classification output")
    scores = np.exp(logits - logits.max())
    scores /= scores.sum()
    return [
        dict(index=int(i), label=imagenet_category[i], score=float(scores[i]))
        for i in np.argsort(scores)[-5:][::-1]
    ]


class VitEngine:
    def __init__(self, directory, backend):
        from runtime import Session

        self.backend = backend
        start = time.perf_counter()
        self.session = Session(
            Path(directory) / MODEL,
            backend,
            npu=backend.endswith("_npu"),
            warmup_inputs={"input": np.zeros((1, 3, 224, 224), np.float32)},
        )
        self.load_ms = (time.perf_counter() - start) * 1000

    def run(self, path, compare=False):
        image = path if isinstance(path, np.ndarray) else cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read {path}")
        feed = {"input": preprocess(image)}
        times = []
        for _ in range(3 if compare else 1):
            start = time.perf_counter()
            outputs = self.session.run(feed)
            times.append((time.perf_counter() - start) * 1000)
        return dict(
            encoder_ms=float(np.median(times)),
            click_ms=0.0,
            inference_runs_ms=times,
            classifications=classify(outputs[0]),
            load_ms=self.load_ms,
            version=self.session.version,
            mode="classification median / 3 runs" if compare else "classification / latest",
        )
