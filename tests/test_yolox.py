"""YOLOX preprocessing, decoding, and persistent benchmark history."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from history import append_history, load_history
from yolox import LABELS, REVISION, YoloEngine, decode, preprocess


class YoloTests(unittest.TestCase):
    def test_open_warmup_and_resident_session(self):
        session = Mock(version="test")
        session.run.return_value = [np.zeros((1, 2100, 85), np.float32)]
        root = Path(__file__).resolve().parents[1]
        with patch.dict("os.environ", {}), patch("engine.Session", return_value=session) as factory:
            engine = YoloEngine(root / "models", "ailia_npu")
            engine.run(root / "images/truck.jpg")
            engine.run(root / "images/truck.jpg", compare=True)
            self.assertEqual(factory.call_count, 1)
            self.assertEqual(session.run.call_count, 4)
            warmup = factory.call_args.kwargs["warmup_inputs"]["images"]
            self.assertEqual(warmup.shape, (1, 3, 320, 320))
            self.assertFalse(warmup.any())

    def test_model_and_calibration_manifest(self):
        root = Path(__file__).resolve().parents[1] / "models"
        manifest = json.loads((root / "yolox_manifest.json").read_text())
        self.assertEqual(manifest["revision"], REVISION)
        report = root / manifest["calibration"]["report"]
        self.assertEqual(
            hashlib.sha256(report.read_bytes()).hexdigest(),
            manifest["calibration"]["report_sha256"],
        )
        calibration = json.loads(report.read_text())
        self.assertEqual(len(set(calibration["images"])), 5000)
        for name, record in manifest["models"].items():
            self.assertEqual(
                hashlib.sha256((root / name).read_bytes()).hexdigest(), record["sha256"]
            )

    def test_letterbox_preserves_bgr_and_padding(self):
        image = np.full((80, 160, 3), [20, 40, 80], np.uint8)
        tensor, ratio = preprocess(image)
        self.assertEqual(ratio, 2)
        self.assertEqual(tensor.shape, (1, 3, 320, 320))
        self.assertEqual(tensor.dtype, np.float32)
        np.testing.assert_array_equal(tensor[0, :, 0, 0], [20, 40, 80])
        np.testing.assert_array_equal(tensor[0, :, 200, 0], [114, 114, 114])

    def test_decode(self):
        output = np.zeros((1, 2100, 85), np.float32)
        output[0, 0, :4] = [1, 2, 0, 0]
        output[0, 0, 4:6] = [1, 0.9]
        boxes, scores, classes = decode(output)
        np.testing.assert_array_equal(boxes, [[4, 12, 12, 20]])
        self.assertAlmostEqual(scores[0], 0.9)
        self.assertEqual(classes.tolist(), [0])
        self.assertEqual(len(LABELS), 80)

    def test_empty_and_invalid(self):
        output = np.zeros((1, 2100, 85), np.float32)
        self.assertEqual(len(decode(output)[0]), 0)
        output[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            decode(output)

    def test_history_survives_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.jsonl"
            append_history(path, "ailia_npu", ["1", "2", "3", "4", "warm"], "image", model="yolox")
            records = load_history(path)
            self.assertEqual(records[0]["model_revision"], REVISION)


if __name__ == "__main__":
    unittest.main()
