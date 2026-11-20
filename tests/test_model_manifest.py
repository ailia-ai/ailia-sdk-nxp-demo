"""Ensure the configured final exports match their distribution manifest."""

import hashlib
import json
import unittest
from pathlib import Path

from scenes.depth_anything_v2_inputs import MODEL as DEPTH_MODEL
from scenes.depth_anything_v2_inputs import REVISION as DEPTH_REVISION
from scenes.grounding_dino_inputs import MODEL as GROUNDING_MODEL
from scenes.grounding_dino_inputs import REVISION as GROUNDING_REVISION
from settings import ENCODER, MEMORY_ATTENTION, MODEL_REVISION
from warmup import inputs_for_model


class ModelManifestTests(unittest.TestCase):
    def test_depth_production_model(self):
        root = Path(__file__).resolve().parents[1] / "models"
        manifest = json.loads((root / "depth_manifest.json").read_text())
        self.assertEqual(manifest["revision"], DEPTH_REVISION)
        self.assertTrue(manifest["calibration"]["production"])
        self.assertEqual(manifest["calibration"]["images"], 5000)
        self.assertTrue(manifest["runtime"]["dynamic_matmul_npu"])
        path = root / DEPTH_MODEL
        self.assertLess(path.stat().st_size, 100 * 1024 * 1024)
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(), manifest["models"][DEPTH_MODEL]["sha256"]
        )

    def test_grounding_production_model_and_shards(self):
        root = Path(__file__).resolve().parents[1] / "models"
        manifest = json.loads((root / "grounding_manifest.json").read_text())
        self.assertEqual(manifest["revision"], GROUNDING_REVISION)
        self.assertTrue(manifest["calibration"]["production"])
        self.assertEqual(manifest["calibration"]["images"], 5000)
        record = manifest["models"][GROUNDING_MODEL]
        files = {GROUNDING_MODEL: record["sha256"], **record["external_data"]}
        files[manifest["tokenizer"]["file"]] = manifest["tokenizer"]["sha256"]
        for name, expected in files.items():
            path = root / name
            with self.subTest(file=name):
                self.assertLess(path.stat().st_size, 100 * 1024 * 1024)
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)

    def test_public_model_names_and_warmup_contract(self):
        root = Path(__file__).resolve().parents[1] / "models"
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertEqual(len(manifest["downloads"]), 3)
        for name, record in manifest["downloads"].items():
            with self.subTest(model=name):
                self.assertEqual(Path(name).name, name)
                self.assertTrue(name.startswith("sam2.1_tiny_"))
                self.assertTrue(record["url"].startswith("https://"))
                self.assertRegex(record["sha256"], r"^[0-9a-f]{64}$")
                if name.endswith(".onnx"):
                    self.assertTrue(inputs_for_model(name))

    def test_final_models_and_warmup_contract(self):
        root = Path(__file__).resolve().parents[1] / "models"
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertEqual(manifest["revision"], MODEL_REVISION)
        self.assertEqual(set(manifest["models"]), {ENCODER, MEMORY_ATTENTION})
        for name, record in manifest["models"].items():
            with self.subTest(model=name):
                self.assertEqual(
                    hashlib.sha256((root / name).read_bytes()).hexdigest(), record["sha256"]
                )
                self.assertTrue(inputs_for_model(name))


if __name__ == "__main__":
    unittest.main()
