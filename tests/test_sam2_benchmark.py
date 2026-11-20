"""Sequence benchmark excludes warm-up and divides by actual invocation counts."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import benchmark_sam2


class FakeVideoEngine:
    def __init__(self, *args):
        self.nets = {
            key: SimpleNamespace(calls=0, session=SimpleNamespace(version="test"))
            for key in ("image_encoder", "prompt_encoder", "memory_attention")
        }

    def track(self, frames, points):
        import numpy as np

        assert len(frames) == 5
        for index in range(5):
            timing = dict(
                image_encoder=20,
                prompt_encoder=10 if index == 0 else 0,
                memory_attention=30 if index else 0,
            )
            for key, value in timing.items():
                self.nets[key].calls += bool(value)
            yield dict(times_ms=timing, mask=np.ones((2, 2), np.uint8))


class SequenceBenchmark(unittest.TestCase):
    def test_counts_and_warmup(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            with (
                patch.object(benchmark_sam2, "VideoEngine", FakeVideoEngine),
                patch("sys.argv", ["benchmark", "--backend", "ailia_cpu", "--output", str(output)]),
                patch("builtins.print"),
            ):
                benchmark_sam2.main()
            result = json.loads(output.read_text())
        self.assertEqual(len(result["passes"]), 3)
        self.assertEqual(
            result["calls_per_sequence"],
            dict(image_encoder=5, prompt_encoder=1, memory_attention=4),
        )
        self.assertEqual(result["median_ms"]["prompt_encoder"], 10)
        self.assertEqual(result["median_ms"]["memory_attention"], 30)
