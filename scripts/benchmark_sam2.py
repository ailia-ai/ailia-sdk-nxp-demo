"""Benchmark the five-frame SAM 2 tracking sequence with profiling disabled."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scenes.sam2_video import VideoEngine
from settings import BACKENDS, CPU_THREADS, MODEL_REVISION


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    engine = VideoEngine(root, args.backend)
    frames = sorted((root / "images/bedroom").glob("[0-9]*.jpg"))
    points = [[210 / 960, 350 / 540, 1], [250 / 960, 220 / 540, 1]]
    passes = []
    mask_pixels = []
    for iteration in range(4):
        before = {key: net.calls for key, net in engine.nets.items()}
        start = time.perf_counter()
        results = list(engine.track(frames, points))
        elapsed = (time.perf_counter() - start) * 1000
        calls = {key: net.calls - before[key] for key, net in engine.nets.items()}
        timing = {
            key: sum(frame["times_ms"][key] for frame in results) / calls[key]
            for key in engine.nets
            if calls[key]
        }
        timing["sequence_ms"] = elapsed
        print(args.backend, iteration, timing, flush=True)
        if iteration:
            passes.append(timing)
            mask_pixels.append([int(frame["mask"].sum()) for frame in results])
    result = dict(
        backend=args.backend,
        version=engine.nets["image_encoder"].session.version,
        model_revision=MODEL_REVISION,
        cpu_threads=CPU_THREADS,
        profiling=False,
        warmup_sequences=1,
        timed_sequences=3,
        passes=passes,
        mask_pixels=mask_pixels,
        calls_per_sequence=calls,
        median_ms={key: float(np.median([p[key] for p in passes])) for key in passes[0]},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
