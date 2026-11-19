"""Host-only checks for standard integer MatMul conversion semantics."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    import onnx
    import onnxruntime as ort
    from depth_convert import quantize
    from onnx import TensorProto as T
    from onnx import helper as h
    from onnx import numpy_helper as nh
except ImportError:
    onnx = None


@unittest.skipIf(onnx is None, "Host ONNX conversion dependencies are not installed")
class DepthConversionTests(unittest.TestCase):
    def model(self):
        nodes, weights, outputs = [], [], []
        for i in range(48):
            name = f"y{i}"
            weights.append(nh.from_array(np.array([[1, -0.5], [0.25, 0.75]], np.float32), f"w{i}"))
            nodes.append(h.make_node("MatMul", ["x", f"w{i}"], [name]))
            outputs.append(h.make_tensor_value_info(name, T.FLOAT, [1, 3, 2]))
        nodes.append(h.make_node("MatMul", ["x", "dynamic_weight"], ["dynamic_y"]))
        outputs.append(h.make_tensor_value_info("dynamic_y", T.FLOAT, [1, 3, 2]))
        model = h.make_model(
            h.make_graph(
                nodes,
                "depth-conversion-test",
                [
                    h.make_tensor_value_info("x", T.FLOAT, [1, 3, 2]),
                    h.make_tensor_value_info("dynamic_weight", T.FLOAT, [2, 2]),
                ],
                outputs,
                weights,
            ),
            opset_imports=[h.make_opsetid("", 17)],
        )
        model.ir_version = 8
        return model

    def test_integer_operators_and_rounding(self):
        model = quantize(self.model(), {"x": [-1, 1]})
        onnx.checker.check_model(model)
        operators = [n.op_type for n in model.graph.node]
        self.assertEqual(operators.count("MatMulInteger"), 48)
        self.assertEqual(operators.count("QuantizeLinear"), 48)
        self.assertEqual(operators.count("MatMul"), 1)
        self.assertNotIn("DequantizeLinear", operators)
        self.assertTrue(all(n.domain == "" for n in model.graph.node))
        x = np.array([[[-0.8, -0.3], [0.2, 0.7], [0.5, -0.1]]], np.float32)
        weight = np.array([[1, -0.5], [0.25, 0.75]], np.float32)
        ws = np.abs(weight).max(axis=0) / 127
        wq = np.rint(weight / ws).astype(np.int32)
        scale = np.float32(2 / 255)
        zero = np.uint8(np.clip(np.rint(1 / scale), 0, 255))
        xq = np.clip(np.rint(x / scale) + zero, 0, 255).astype(np.int32)
        expected = ((xq - zero) @ wq).astype(np.float32) * (scale * ws)
        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        values = session.run(None, {"x": x, "dynamic_weight": weight})
        for value in values[:-1]:
            np.testing.assert_allclose(value, expected, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(values[-1], x @ weight, rtol=1e-6)

    def test_invalid_calibration_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid calibration"):
            quantize(self.model(), {"x": [float("nan"), 1]})
