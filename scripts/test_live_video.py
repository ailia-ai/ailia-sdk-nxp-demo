#!/usr/bin/env python3
"""Device smoke test: real video capture through the GUI's inference worker."""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from live_input import LiveInput  # noqa: E402
from runtime_worker import RuntimeWorker  # noqa: E402
from settings import BACKENDS  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=["ailia_npu", "ort_cpu"])
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("sam", "grounding", "depth", "yolox"),
        default=["sam", "grounding", "depth", "yolox"],
    )
    parser.add_argument(
        "--timeout", type=float, default=600, help="Per-frame timeout, including first model Open"
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="ailia-live-") as directory:
        video = str(Path(directory) / "test.avi")
        image = cv2.imread(str(ROOT / "images/truck.jpg"))
        image = cv2.resize(image, (640, 480))
        writer = cv2.VideoWriter(
            video, cv2.CAP_OPENCV_MJPEG, cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 480)
        )
        if not writer.isOpened():
            raise RuntimeError("MJPG video writer unavailable")
        for offset in range(20):
            writer.write(np.roll(image, offset, axis=1))
        writer.release()
        capture, runner = LiveInput(video), RuntimeWorker(ROOT / "models")
        try:
            for model in args.models:
                for backend in args.backends:
                    process = None
                    for index in range(2):
                        deadline = time.monotonic() + 10
                        item = None
                        while item is None and time.monotonic() < deadline:
                            item = capture.read()
                            time.sleep(0.02)
                        if item is None or item[0] != "frame":
                            raise RuntimeError(f"Capture failed: {item}")
                        runner.submit(
                            dict(
                                model=model,
                                backend=backend,
                                action="live",
                                image=item[1],
                                points=[[0.5, 0.5, 1]],
                                caption="truck. person.",
                                video_mode=False,
                            )
                        )
                        if index and runner.process.pid != process:
                            raise RuntimeError("Session was not retained between frames")
                        process = runner.process.pid
                        seen = False
                        while True:
                            result = runner.responses.get(timeout=args.timeout)
                            if result["kind"] == "error":
                                raise RuntimeError(result["text"])
                            if result["kind"] == "result":
                                seen = True
                                key = {"sam": "mask", "depth": "depth"}.get(model, "detections")
                                if key not in result:
                                    raise RuntimeError(f"Missing output: {key}")
                                if key != "detections" and not np.isfinite(result[key]).all():
                                    raise RuntimeError("Non-finite output")
                                print(
                                    json.dumps(
                                        dict(
                                            model=model,
                                            backend=backend,
                                            frame=index,
                                            inference_ms=result["encoder_ms"],
                                            output=key,
                                        )
                                    ),
                                    flush=True,
                                )
                            if result["kind"] == "done":
                                if not seen:
                                    raise RuntimeError("Missing inference result")
                                break
        finally:
            runner.close()
            capture.close()


if __name__ == "__main__":
    main()
