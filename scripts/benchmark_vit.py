"""Unprofiled ViT benchmark, one Open-time dummy inference and ten timed runs."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scenes import default_image
from scenes.vit import MODEL, VitEngine, classify, preprocess
from settings import BACKENDS, CPU_THREADS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", type=Path, default=Path(__file__).resolve().parents[1] / "models"
    )
    parser.add_argument("--image", type=Path, default=default_image("vit"))
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("runs must be positive")
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f"Cannot read {args.image}")
    engine = VitEngine(args.models, args.backend)
    feed = {"input": preprocess(image)}
    times, predictions = [], []
    for _ in range(args.runs):
        start = time.perf_counter()
        outputs = engine.session.run(feed)
        times.append((time.perf_counter() - start) * 1000)
        predictions.append(classify(outputs[0]))
        print(f"Inference {len(times)}/{args.runs}: {times[-1]:.3f} ms", flush=True)
    result = dict(
        backend=args.backend,
        version=engine.session.version,
        model=MODEL,
        model_sha256=hashlib.sha256((args.models / MODEL).read_bytes()).hexdigest(),
        image_sha256=hashlib.sha256(args.image.read_bytes()).hexdigest(),
        times_ms=times,
        median_ms=float(np.median(times)),
        warmup_runs=1,
        open_warmup_ms=engine.load_ms,
        cpu_threads=CPU_THREADS,
        profiling=False,
        predictions=predictions,
        scope="network inference including original attention outputs; excludes preprocessing, classification and rendering",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    np.save(args.output.with_suffix(".npy"), outputs[0])
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
