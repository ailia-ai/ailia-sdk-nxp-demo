"""Fixed attention calibration preserves standard ONNX quantization semantics."""
import sys
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as T, helper as h

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from calibrate_attention import freeze_quantizers


class AttentionCalibrationTests(unittest.TestCase):
    def model(self):
        return h.make_model(h.make_graph(
            [h.make_node("DynamicQuantizeLinear", ["x"], ["q", "s", "z"])],
            "quantizer", [h.make_tensor_value_info("x", T.FLOAT, [5])],
            [h.make_tensor_value_info("q", T.UINT8, [5]),
             h.make_tensor_value_info("s", T.FLOAT, []),
             h.make_tensor_value_info("z", T.UINT8, [])]),
            opset_imports=[h.make_opsetid("", 17)], ir_version=9)

    def test_fixed_quantizer_rounding_and_clipping(self):
        model = freeze_quantizers(self.model(), {"x": [-1., 2.]})
        onnx.checker.check_model(model)
        self.assertEqual([n.op_type for n in model.graph.node], ["QuantizeLinear"])
        session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        x = np.array([-3., -1., 0., 2., 4.], np.float32)
        q, scale, zero = session.run(None, {"x": x})
        np.testing.assert_array_equal(q, np.clip(np.rint(x / scale) + zero, 0, 255).astype(np.uint8))
        self.assertEqual(int(zero), 85)

    def test_invalid_ranges_rejected(self):
        for values in ([float("nan"), 1.], [1., 2.], [-1., -2.]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                freeze_quantizers(self.model(), {"x": values})

    def test_missing_ranges_rejected(self):
        with self.assertRaises(KeyError):
            freeze_quantizers(self.model(), {})
