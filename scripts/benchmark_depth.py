"""Measure Metric Depth Small with one Open warmup and three timed runs."""

import argparse
import hashlib
import json
import queue
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_worker import RuntimeWorker
from scenes.depth_anything_v2_inputs import MODEL, REVISION
from settings import BACKENDS, CPU_THREADS


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=root / "models")
    parser.add_argument("--image", type=Path, default=root / "images/bedroom/00000.jpg")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=BACKENDS, action="append")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model_hash = hashlib.sha256((args.models / MODEL).read_bytes()).hexdigest()
    runner, results = RuntimeWorker(args.models), {}
    try:
        for backend in args.backend or BACKENDS:
            runner.submit(
                dict(model="depth", backend=backend, action="benchmark", image=str(args.image))
            )
            deadline = time.monotonic() + 600
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError(backend)
                try:
                    event = runner.responses.get(timeout=1)
                except queue.Empty:
                    if not runner.process.is_alive():
                        raise RuntimeError(f"Worker exited: {backend}")
                    continue
                if event["kind"] == "error":
                    raise RuntimeError(event["text"])
                if event["kind"] == "result":
                    depth = event.pop("depth")
                    event.update(
                        image=str(args.image.resolve()),
                        model_sha256=model_hash,
                        revision=REVISION,
                        threads=CPU_THREADS,
                        profiling=False,
                        warmup_runs=1,
                    )
                    np.save(args.output.with_name(args.output.stem + "_" + backend + ".npy"), depth)
                    results[backend] = event
                    args.output.write_text(json.dumps(results, indent=2) + "\n")
                    print(f"{backend}: {event['encoder_ms']:.1f} ms", flush=True)
                if event["kind"] == "done":
                    break
    finally:
        runner.close()


if __name__ == "__main__":
    main()
