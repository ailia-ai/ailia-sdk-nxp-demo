"""Grounding input geometry, text masks and resident-session behavior."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from grounding import GroundingEngine
from grounding_inputs import HEIGHT, WIDTH, Tokenizer, detections, preprocess
from history import append_history, load_history
from runtime_worker import RuntimeWorker

ROOT = Path(__file__).resolve().parents[1]


class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer(ROOT / "models/grounding_tokenizer/vocab.txt")

    def test_text_masks_separate_categories(self):
        feed, tokens = self.tokenizer.encode("Person. truck")
        self.assertEqual(tokens, ["[CLS]", "person", ".", "truck", ".", "[SEP]"])
        self.assertEqual(feed["input_ids"].dtype, np.int64)
        np.testing.assert_array_equal(feed["position_ids"], [[0, 0, 1, 0, 1, 0]])
        self.assertTrue(feed["attention_mask"][0, 1, 2])
        self.assertFalse(feed["attention_mask"][0, 1, 3])

    def test_prompt_limits(self):
        with self.assertRaises(ValueError):
            self.tokenizer.encode(" ")
        with self.assertRaises(ValueError):
            self.tokenizer.encode("person. " * 128)
        _, tokens = self.tokenizer.encode("café")
        self.assertIn("cafe", tokens)

    def test_letterbox_and_box_coordinates(self):
        samples, geometry = preprocess(np.zeros((100, 200, 3), np.uint8))
        self.assertEqual(samples.shape, (1, 3, HEIGHT, WIDTH))
        self.assertEqual(samples.dtype, np.float32)
        left, top, width, height = geometry
        box = [
            (left + width / 2) / WIDTH,
            (top + height / 2) / HEIGHT,
            width / WIDTH,
            height / HEIGHT,
        ]
        logits = np.array([[[-10, 10, -10, -10]]], np.float32)
        result = detections([logits, np.array([[box]])], ["[CLS]", "truck", ".", "[SEP]"], geometry)
        self.assertEqual(result[0]["label"], "truck")
        np.testing.assert_allclose(result[0]["box"], [0, 0, 1, 1], atol=1e-6)

    def test_reuse_and_reopen_for_token_shape(self):
        session = Mock(version="test")
        session.run.return_value = [np.zeros((1, 0, 256)), np.zeros((1, 0, 4))]
        with (
            patch.dict("os.environ", {}, clear=False),
            patch("grounding.Session", return_value=session) as factory,
        ):
            engine = GroundingEngine(ROOT / "models", "ailia_npu")
            engine.run(ROOT / "images/truck.jpg", "truck.")
            engine.run(ROOT / "images/truck.jpg", "person.", compare=True)
            self.assertEqual(factory.call_count, 1)
            self.assertEqual(session.run.call_count, 4)
            engine.run(ROOT / "images/truck.jpg", "person. truck.")
            self.assertEqual(factory.call_count, 2)
            self.assertIn("warmup_inputs", factory.call_args.kwargs)

    def test_history_retains_each_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.jsonl"
            values = ["1", "2", "3", "6", "test"]
            append_history(path, "ailia_npu", values, "truck.jpg")
            for _ in range(4):
                append_history(
                    path, "ailia_npu", values, "truck.jpg", model="grounding", caption="truck."
                )
            records = load_history(path, limit=1)
            self.assertEqual({r["model"] for r in records}, {"sam", "grounding"})

    def test_runtime_process_isolated_on_model_or_backend_change(self):
        context = Mock()
        processes = [Mock() for _ in range(3)]
        for process in processes:
            process.is_alive.return_value = True
        context.Process.side_effect = processes
        with patch("runtime_worker.mp.get_context", return_value=context):
            runner = RuntimeWorker(ROOT / "models")
            runner.submit(dict(model="sam", backend="ailia_npu"))
            runner.submit(dict(model="sam", backend="ailia_npu"))
            self.assertEqual(context.Process.call_count, 1)
            runner.submit(dict(model="grounding", backend="ailia_npu"))
            self.assertEqual(context.Process.call_count, 2)
            processes[0].close.assert_called_once()
            runner.submit(dict(model="grounding", backend="ort_npu"))
            self.assertEqual(context.Process.call_count, 3)
            processes[1].close.assert_called_once()
            runner.close()


if __name__ == "__main__":
    unittest.main()
