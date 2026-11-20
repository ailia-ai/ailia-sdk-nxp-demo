"""Headless smoke/benchmark runner. Saves masks and real measurements."""

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from runtime import BACKENDS
from scenes.sam2 import Engine
from settings import MODEL_REVISION


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=str(Path(__file__).parent / "models"))
    p.add_argument("--image", default=str(Path(__file__).parent / "images" / "truck.jpg"))
    p.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    p.add_argument("--output", type=Path, default=Path(__file__).parent / "selftest_results")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    if len(args.backends) > 1:
        # Native EP allocators may retain CMA until process exit. Match the GUI's
        # runtime-switch isolation rather than relying on Python garbage collection.
        for backend in args.backends:
            directory = args.output / backend
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--models",
                    args.models,
                    "--image",
                    args.image,
                    "--backends",
                    backend,
                    "--output",
                    str(directory),
                ],
                check=True,
            )
            records.extend(json.loads((directory / "results.json").read_text()))
        (args.output / "results.json").write_text(json.dumps(records, indent=2))
        return
    for backend in args.backends:
        engine = Engine(args.models, backend)
        first = engine.prepare(args.image)
        engine.segment([[0.6, 0.65, 1]])
        enc, click = [], []
        for _ in range(3):
            enc.append(engine.encode())
            result = engine.segment([[0.6, 0.65, 1]])
            click.append(result["click_ms"])
        if backend != "ort_npu":
            assert 0 < int(result["mask"].sum()) < result["mask"].size
        cv2.imwrite(str(args.output / (backend + ".png")), result["mask"] * 255)
        # Dynamic point count and foreground/background prompts must both work.
        multiple = engine.segment([[0.6, 0.65, 1], [0.9, 0.1, 0]])
        assert multiple["mask"].shape == (128, 128)
        record = dict(
            backend=backend,
            version=engine.encoder.version,
            load_ms=engine.load_ms,
            first_encoder_ms=first,
            encoder_runs_ms=enc,
            click_runs_ms=click,
            encoder_median_ms=float(np.median(enc)),
            click_median_ms=float(np.median(click)),
            mask_pixels=int(result["mask"].sum()),
            score=result["score"],
            profiling=False,
            threads=6,
            model_revision=MODEL_REVISION,
            accuracy_qualified=False if backend == "ort_npu" else None,
            warning="Known Neutron EP output errors; timing only" if backend == "ort_npu" else None,
        )
        records.append(record)
        print(json.dumps(record), flush=True)
        (args.output / "results.json").write_text(json.dumps(records, indent=2))
        del engine
        gc.collect()


if __name__ == "__main__":
    main()
