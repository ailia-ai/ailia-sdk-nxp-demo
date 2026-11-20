"""Measure all four Grounding DINO backends using the demo's runtime settings."""

import argparse
import hashlib
import json
import queue
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_worker import RuntimeWorker
from scenes.grounding_dino_inputs import DEFAULT_PROMPT, MODEL, REVISION
from settings import BACKENDS, CPU_THREADS


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=root / "models")
    parser.add_argument("--image", type=Path, default=root / "images/grounding_dino/demo.jpg")
    parser.add_argument("--caption", default=DEFAULT_PROMPT)
    parser.add_argument("--backend", choices=BACKENDS, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model_hash = hashlib.sha256((args.models / MODEL).read_bytes()).hexdigest()
    results = {}
    runner = RuntimeWorker(args.models)
    try:
        for backend in args.backend or BACKENDS:
            runner.submit(
                dict(
                    model="grounding",
                    backend=backend,
                    action="benchmark",
                    image=str(args.image),
                    caption=args.caption,
                )
            )
            deadline = time.monotonic() + args.timeout
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{backend} exceeded {args.timeout} seconds")
                try:
                    event = runner.responses.get(timeout=1)
                except queue.Empty:
                    if not runner.process.is_alive():
                        raise RuntimeError(f"{backend} exited unexpectedly")
                    continue
                if event["kind"] == "error":
                    raise RuntimeError(event.get("text", str(event)))
                if event["kind"] == "result":
                    event.update(model_sha256=model_hash, revision=REVISION,
                                 threads=CPU_THREADS, profiling=False, warmup_runs=1)
                    results[backend] = event
                    args.output.write_text(json.dumps(results, indent=2) + "\n")
                    print(
                        f"{backend}: {event['encoder_ms']:.1f} ms, "
                        f"{len(event['detections'])} detections",
                        flush=True,
                    )
                if event["kind"] == "done":
                    break
    finally:
        runner.close()


if __name__ == "__main__":
    main()
