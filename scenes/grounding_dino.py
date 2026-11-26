"""Resident Grounding DINO runtime, separate from SAM's tracking state."""

import time
from pathlib import Path

import cv2
import numpy as np

from runtime import Session
from scenes.grounding_dino_inputs import HEIGHT, MODEL, WIDTH, Tokenizer, detections, preprocess


class GroundingEngine:
    def __init__(self, directory, backend):
        self.directory, self.backend = Path(directory), backend
        self.tokenizer = Tokenizer(self.directory / "grounding_tokenizer" / "vocab.txt")
        self.session = None
        self.text_tokens = None

    def run(self, path, caption, compare=False):
        image = path if isinstance(path, np.ndarray) else cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read image: {path}")
        samples, geometry = preprocess(image)
        text, tokens = self.tokenizer.encode(caption, pad_to=256)
        if self.session is None:
            start = time.perf_counter()
            warmup = dict(samples=np.zeros((1, 3, HEIGHT, WIDTH), np.float32), **text)
            self.session = Session(
                self.directory / MODEL,
                self.backend,
                npu=self.backend.endswith("_npu"),
                warmup_inputs=warmup,
                validate_outputs=lambda values: (
                    len(values) == 2
                    and np.isfinite(values[0][..., : len(tokens)]).all()
                    and np.isfinite(values[1]).all()
                ),
            )
            self.load_ms = (time.perf_counter() - start) * 1000
            self.text_tokens = tuple(tokens)
        times = []
        for _ in range(3 if compare else 1):
            # ailia retains omitted inputs. Updating only the image preserves
            # unchanged text branches; ORT requires every input on every run.
            update_text = not self.backend.startswith("ailia") or self.text_tokens != tuple(tokens)
            feed = dict(samples=samples, **(text if update_text else {}))
            start = time.perf_counter()
            outputs = self.session.run(feed)
            times.append((time.perf_counter() - start) * 1000)
            self.text_tokens = tuple(tokens)
        return dict(
            encoder_ms=float(np.median(times)),
            click_ms=0.0,
            inference_runs_ms=times,
            detections=detections(outputs, tokens, geometry),
            caption=caption,
            load_ms=self.load_ms,
            version=self.session.version,
            mode="detection median / 3 runs" if compare else "detection / latest",
        )
