"""Small numerical checks for integer lowering and portable external weights."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper as nh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import grounding_convert as converter


class ConversionTests(unittest.TestCase):
    def test_dynamic_integer_quantization_preserves_broadcasting(self):
        model = h.make_model(
            h.make_graph(
                [h.make_node("MatMul", ["a", "b"], [f"y{i}"]) for i in range(109)],
                "dynamic",
                [
                    h.make_tensor_value_info("a", T.FLOAT, [2, 3, 4]),
                    h.make_tensor_value_info("b", T.FLOAT, [1, 4, 5]),
                ],
                [h.make_tensor_value_info("y0", T.FLOAT, [2, 3, 5])],
            ),
            opset_imports=[h.make_opsetid("", 17)],
            ir_version=9,
        )
        converted = converter.quantize_dynamic(model)
        onnx.checker.check_model(converted)
        ops = [n.op_type for n in converted.graph.node]
        self.assertEqual(ops.count("DynamicQuantizeLinear"), 218)
        self.assertEqual(ops.count("MatMulInteger"), 109)
        self.assertNotIn("MatMul", ops)
        a = np.linspace(-2, 4, 24, dtype=np.float32).reshape(2, 3, 4)
        b = np.linspace(-3, 1, 20, dtype=np.float32).reshape(1, 4, 5)

        def quantize(x):
            lo, hi = min(float(x.min()), 0), max(float(x.max()), 0)
            scale = np.float32((hi - lo) / 255)
            zero = np.clip(np.rint(-lo / scale), 0, 255)
            return (np.clip(np.rint(x / scale) + zero, 0, 255).astype(np.int32) - int(zero)), scale

        qa, sa = quantize(a)
        qb, sb = quantize(b)
        session = ort.InferenceSession(
            converted.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        actual = session.run(None, {"a": a, "b": b})[0]
        np.testing.assert_allclose(actual, (qa @ qb).astype(np.float32) * (sa * sb), atol=1e-5)

    def test_dynamic_conversion_rejects_unexpected_graph(self):
        model = h.make_model(h.make_graph([], "empty", [], []))
        with self.assertRaisesRegex(ValueError, "Expected 109"):
            converter.quantize_dynamic(model)

    def test_integer_padding_restores_batch_shape_and_zero_point(self):
        weight = np.array([[1.5, -0.7], [-0.4, 2]], np.float32)
        node = h.make_node("MatMul", ["a", "w"], ["y"])
        model = h.make_model(
            h.make_graph(
                [node],
                "test",
                [h.make_tensor_value_info("a", T.FLOAT, [2, 3, 2])],
                [h.make_tensor_value_info("y", T.FLOAT, [2, 3, 2])],
                [nh.from_array(weight, "w")],
            ),
            opset_imports=[h.make_opsetid("", 17)],
            ir_version=9,
        )
        with (
            patch.object(converter, "targets", return_value=({"w": weight}, [node])),
            patch.object(converter, "rewrite"),
            patch.object(converter, "quantize_dynamic", side_effect=lambda model: model),
        ):
            converted = converter.quantize(model, {"a": [-2, 5]})
        onnx.checker.check_model(converted)
        session = ort.InferenceSession(
            converted.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        x = np.linspace(-2, 5, 12, dtype=np.float32).reshape(2, 3, 2)
        actual = session.run(None, {"a": x})[0]
        sa = np.float32(7 / 255)
        za = np.uint8(np.rint(2 / sa))
        sb = np.abs(weight).max(axis=0) / 127
        qa = np.clip(np.rint(x / sa) + za, 0, 255).astype(np.int32) - za
        qw = np.rint(weight / sb).astype(np.int32)
        expected = (qa @ qw).astype(np.float32) * sa * sb
        self.assertEqual(actual.shape, x.shape)
        np.testing.assert_allclose(actual, expected, atol=1e-5)

    def test_external_shards_are_relative_and_reloadable(self):
        weights = [np.eye(32, dtype=np.float32), np.full((32, 32), 0.125, np.float32)]
        model = h.make_model(
            h.make_graph(
                [
                    h.make_node("MatMul", ["x", "w0"], ["a"]),
                    h.make_node("MatMul", ["a", "w1"], ["y"]),
                ],
                "test",
                [h.make_tensor_value_info("x", T.FLOAT, [1, 32])],
                [h.make_tensor_value_info("y", T.FLOAT, [1, 32])],
                [nh.from_array(w, f"w{i}") for i, w in enumerate(weights)],
            ),
            opset_imports=[h.make_opsetid("", 17)],
            ir_version=9,
        )
        with tempfile.TemporaryDirectory() as directory:
            files = converter.save_sharded(model, Path(directory) / "model.onnx", limit=4096)
            self.assertEqual(len(files), 3)
            self.assertTrue(all(f.stat().st_size <= 4096 for f in files[1:]))
            reloaded = onnx.load(files[0])
            for tensor, expected in zip(reloaded.graph.initializer, weights):
                np.testing.assert_array_equal(nh.to_array(tensor), expected)


if __name__ == "__main__":
    unittest.main()
