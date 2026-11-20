"""Memory-bank and input-contract regressions without loading model weights."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scenes.sam2_video import TimedSession, VideoEngine


class ModelLayoutTests(unittest.TestCase):
    def test_all_tracking_models_use_shared_directory(self):
        root = Path(__file__).resolve().parents[1]
        with patch("scenes.sam2_video.TimedSession") as session:
            engine = VideoEngine(root, "ort_cpu")
        self.assertEqual(len(engine.nets), 7)
        self.assertEqual(session.call_count, 7)
        for call in session.call_args_list:
            path = call.args[0]
            self.assertEqual(path.parent, root / "models")
            self.assertTrue(path.is_file(), str(path))


class MemoryBankTests(unittest.TestCase):
    def adapter(self):
        adapter = TimedSession.__new__(TimedSession)
        adapter.memory_bank = True
        adapter.capture = None
        adapter.ms = 0.0
        adapter.calls = 0
        adapter.session = SimpleNamespace(run=lambda feed: feed)
        return adapter

    def test_padding_and_mask(self):
        adapter = self.adapter()
        original = {
            "memory_1": np.ones((1024, 1, 64), np.float32),
            "memory_2": np.ones((4, 1, 64), np.float32),
            "attention_mask_1": np.ones((1024, 1), bool),
            "attention_mask_2": np.ones((4, 1), bool),
        }
        result = adapter.run(original)
        self.assertEqual(result["memory_1"].shape, (4096, 1, 64))
        self.assertEqual(result["memory_2"].shape, (16, 1, 64))
        self.assertTrue(result["attention_mask_1"][:1024].all())
        self.assertFalse(result["attention_mask_1"][1024:].any())
        self.assertFalse(result["memory_1"][1024:].any())
        self.assertEqual(original["memory_1"].shape, (1024, 1, 64))

    def test_overflow_is_rejected(self):
        with self.assertRaises(ValueError):
            self.adapter().run({"memory_1": np.zeros((4097, 1, 64), np.float32)})


if __name__ == "__main__":
    unittest.main()
