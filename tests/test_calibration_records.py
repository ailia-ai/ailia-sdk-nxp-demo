"""Verify that archived calibration records belong to the bundled models."""

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CalibrationRecordsTests(unittest.TestCase):
    def test_full_dataset_records(self):
        for filename in ("manifest.json", "grounding_manifest.json", "depth_manifest.json"):
            with self.subTest(model=filename):
                model = read(ROOT / "models" / filename)
                calibration = model["calibration"]
                records = ROOT / "models" / calibration["records"]
                manifest = read(records / "manifest.json")
                self.assertEqual(len(manifest["images"]), 5000)
                self.assertEqual(digest(records / "manifest.json"), calibration["manifest_sha256"])
                completion = read(records / "completion.json")
                self.assertTrue(completion["production"])
                self.assertEqual(completion["manifest_sha256"], calibration["manifest_sha256"])
                if "ranges_sha256" in calibration:
                    self.assertEqual(digest(records / "ranges.json"), calibration["ranges_sha256"])
                else:
                    shards = [read(p) for p in sorted(records.glob("ranges_*.json"))]
                    self.assertEqual(len(shards), 4)
                    for shard in shards:
                        self.assertEqual(shard["manifest_sha256"], calibration["manifest_sha256"])

    def test_yolox_report(self):
        model = read(ROOT / "models/yolox_manifest.json")
        calibration = model["calibration"]
        report = ROOT / "models" / calibration["report"]
        self.assertEqual(digest(report), calibration["report_sha256"])
        data = read(report)
        self.assertEqual(len(set(data["images"])), 5000)
        self.assertEqual(data["model_sha256"], model["models"]["yolox_s_320_int8.onnx"]["sha256"])


if __name__ == "__main__":
    unittest.main()
