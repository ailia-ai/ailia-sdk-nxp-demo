"""Calibration invariants with small synthetic inputs (no dataset download)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    import onnx
    from calibrate_sam2 import (
        RangeSession,
        dataset_manifest,
        digest,
        finalize,
        foreground_point,
        save_json,
    )
    from onnx import TensorProto, helper, numpy_helper
except ImportError:
    onnx = None


@unittest.skipIf(onnx is None, "Host-side calibration dependencies are optional")
class CalibrationTests(unittest.TestCase):
    def test_requires_complete_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation = root / "instances.json"
            annotation.write_text(json.dumps({"images": [{"id": 1}]}))
            with self.assertRaisesRegex(ValueError, "5,000"):
                dataset_manifest(root, annotation, root)

    def test_range_union_preserves_original_output(self):
        graph = helper.make_graph(
            [helper.make_node("MatMul", ["x", "w"], ["y"])],
            "test",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
            [numpy_helper.from_array(np.eye(2, dtype=np.float32), "w")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.onnx"
            onnx.save(model, path)
            session = RangeSession(path, "image_encoder", 1, collect=True)
            for values in ([[-3, 1]], [[0, 9]]):
                values = np.array(values, np.float32)
                np.testing.assert_array_equal(session.run({"x": values})[0], values)
            self.assertEqual(session.ranges["x"], [-3, 9])
            self.assertEqual(session.ranges["y"], [-3, 9])
            self.assertEqual(session.calls, 2)
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                session.run({"x": np.array([[np.inf, np.inf]], np.float32)})

    def test_incomplete_calibration_cannot_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"images": [{"id": 1}, {"id": 2}]}
            save_json(root / "manifest.json", manifest)
            from prepare_sam2 import SOURCES

            save_json(
                root / "ranges_000.json",
                dict(
                    manifest_sha256=digest(root / "manifest.json"),
                    shards=1,
                    completed=[1],
                    calls={key: 0 for key in SOURCES},
                    ranges={"image_encoder": {}, "memory_attention": {}},
                ),
            )
            with self.assertRaisesRegex(ValueError, "incomplete"):
                finalize(SimpleNamespace(output=root, workers=1, smoke_images=0), manifest)
            self.assertFalse((root / "models").exists())

    def test_prompt_is_inside_mask_or_explicit_background(self):
        mask = np.zeros((20, 30), np.uint8)
        mask[5:15, 10:20] = 1
        coco = SimpleNamespace(
            getAnnIds=lambda **kwargs: [1],
            loadAnns=lambda ids: [{"id": 1, "area": 100}],
            annToMask=lambda annotation: mask,
        )
        points, labels = foreground_point(coco, 1, 20, 30)
        x, y = np.rint(points[0] * [30 / 512, 20 / 512]).astype(int)
        self.assertEqual(mask[y, x], 1)
        self.assertEqual(labels.tolist(), [1])
        coco.loadAnns = lambda ids: []
        points, labels = foreground_point(coco, 1, 20, 30)
        self.assertEqual(labels.tolist(), [0])
        self.assertEqual(points.tolist(), [[256, 256]])
