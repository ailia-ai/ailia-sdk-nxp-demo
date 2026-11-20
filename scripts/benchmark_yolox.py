"""Profile-disabled YOLOX benchmark: one dummy Open inference, ten timed runs."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scenes.yolox import MODEL, YoloEngine, preprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("ailia_cpu", "ailia_npu", "ort_cpu", "ort_npu"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("runs must be positive")
    engine = YoloEngine(args.models, args.backend, model=args.model)
    image = cv2.imread(str(args.image))
    tensor, _ = preprocess(image)
    times = []
    for _ in range(args.runs):
        start = time.perf_counter()
        outputs = engine.session.run({"images": tensor})
        times.append((time.perf_counter() - start) * 1000)
        print(f"Inference {len(times)}/{args.runs}: {times[-1]:.3f} ms", flush=True)
    if not all(np.isfinite(v).all() for v in outputs):
        raise ValueError("Non-finite output")
    result = dict(
        backend=args.backend,
        version=engine.session.version,
        model=args.model,
        model_sha256=hashlib.sha256((args.models / args.model).read_bytes()).hexdigest(),
        image_sha256=hashlib.sha256(args.image.read_bytes()).hexdigest(),
        times_ms=times,
        median_ms=float(np.median(times)),
        open_warmup_ms=engine.load_ms,
        warmup_runs=1,
        cpu_threads=6,
        profiling=False,
        conv_device_output=True,
        scope="inference only; excludes preprocessing, NMS, loading and rendering",
    )
    if args.backend.startswith("ailia"):
        import ailia

        library = Path(ailia.__file__).parent / "linux/arm64-v8a/libailia.so"
        result["library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    np.save(args.output.with_suffix(".npy"), outputs[0])
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
