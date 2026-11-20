"""Depth attention quantization is explicit and portable standard ONNX."""

import sys
import unittest
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnx import TensorProto as T
from onnx import helper as h

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from depth_convert import quantize_dynamic_matmuls


class DepthDynamicQuantization(unittest.TestCase):
    def model(self, count=24):
        return h.make_model(
            h.make_graph(
                [h.make_node("MatMul", ["a", "b"], [f"y{i}"]) for i in range(count)],
                "attention",
                [
                    h.make_tensor_value_info("a", T.FLOAT, [1, 2, 3, 5]),
                    h.make_tensor_value_info("b", T.FLOAT, [1, 2, 5, 4]),
                ],
                [h.make_tensor_value_info(f"y{i}", T.FLOAT, [1, 2, 3, 4]) for i in range(count)],
            ),
            opset_imports=[h.make_opsetid("", 17)],
            ir_version=9,
        )

    def test_explicit_operators_and_numerics(self):
        model = quantize_dynamic_matmuls(self.model())
        ops = [node.op_type for node in model.graph.node]
        self.assertEqual(ops.count("DynamicQuantizeLinear"), 48)
        self.assertEqual(ops.count("MatMulInteger"), 24)
        self.assertNotIn("MatMul", ops)
        self.assertTrue(all(node.domain == "" for node in model.graph.node))
        rng = np.random.default_rng(7)
        a = rng.normal(size=(1, 2, 3, 5)).astype(np.float32)
        b = rng.normal(size=(1, 2, 5, 4)).astype(np.float32)

        def quantize(x):
            lo, hi = min(x.min(), 0), max(x.max(), 0)
            scale = np.float32((hi - lo) / 255)
            zero = np.clip(np.rint(-lo / scale), 0, 255)
            q = np.clip(np.rint(x / scale) + zero, 0, 255).astype(np.int32)
            return q - np.int32(zero), scale

        qa, sa = quantize(a)
        qb, sb = quantize(b)
        expected = np.matmul(qa, qb).astype(np.float32) * (sa * sb)
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        for output in session.run(None, {"a": a, "b": b}):
            np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)

    def test_unexpected_graph_rejected(self):
        with self.assertRaises(ValueError):
            quantize_dynamic_matmuls(self.model(23))
