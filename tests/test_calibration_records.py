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

    def test_full_attention_records(self):
        import onnx

        for filename, count, quantizers in (("depth_manifest.json", 24, 48),
                                             ("grounding_manifest.json", 109, 218)):
            with self.subTest(model=filename):
                model = read(ROOT / "models" / filename)
                calibration = model["attention_calibration"]
                records = ROOT / "models" / calibration["records"]
                manifest = read(records / "manifest.json")
                completion = read(records / "completion.json")
                ids = [image["id"] for image in manifest["images"]]
                self.assertEqual(len(ids), 5000)
                self.assertEqual(len(set(ids)), 5000)
                self.assertEqual(completion["images"], 5000)
                self.assertTrue(completion["production"])
                self.assertEqual(completion["quantizers"], quantizers)
                for name in ("manifest", "ranges"):
                    sha = digest(records / (name + ".json"))
                    self.assertEqual(sha, completion[name + "_sha256"])
                    self.assertEqual(sha, calibration[name + "_sha256"])
                for name, record in model["models"].items():
                    expected_hash = completion["model_sha256"]
                    if "convolution_calibration" in model:
                        conv = model["convolution_calibration"]
                        conv_records = ROOT / "models" / conv["records"]
                        conv_completion = read(conv_records / "completion.json")
                        self.assertEqual(expected_hash, conv_completion["source_sha256"])
                        expected_hash = conv_completion["model_sha256"]
                    self.assertEqual(digest(ROOT / "models" / name), expected_hash)
                    graph = onnx.load(ROOT / "models" / name, load_external_data=False)
                    self.assertNotIn("DynamicQuantizeLinear", [n.op_type for n in graph.graph.node])
                    self.assertEqual(record["dynamic_matmuls"], count)

    def test_depth_convolution_records(self):
        import onnx

        model = read(ROOT / "models/depth_manifest.json")
        calibration = model["convolution_calibration"]
        records = ROOT / "models" / calibration["records"]
        manifest = read(records / "manifest.json")
        completion = read(records / "completion.json")
        self.assertEqual(len({v["id"] for v in manifest["images"]}), 5000)
        self.assertEqual(completion["images"], 5000)
        self.assertTrue(completion["production"])
        self.assertEqual(completion["converted_convolutions"], 28)
        self.assertEqual(completion["retained_fp32_convolutions"], 3)
        for name in ("manifest", "ranges"):
            self.assertEqual(digest(records / (name + ".json")), completion[name + "_sha256"])
            self.assertEqual(completion[name + "_sha256"], calibration[name + "_sha256"])
        for name in model["models"]:
            graph = onnx.load(ROOT / "models" / name)
            self.assertEqual(sum(n.op_type == "ConvInteger" for n in graph.graph.node), 28)
            self.assertEqual(sum(n.op_type == "Conv" for n in graph.graph.node), 3)
            self.assertEqual(digest(ROOT / "models" / name), completion["model_sha256"])


if __name__ == "__main__":
    unittest.main()
