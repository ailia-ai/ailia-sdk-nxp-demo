"""Standard ONNX ConvInteger conversion and retained FP32 head checks."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    import onnx
    from depth_convert import quantize_convolutions
    from onnx import TensorProto as T
    from onnx import helper as h
    from onnx import numpy_helper as nh
    from onnx.reference import ReferenceEvaluator
except ImportError:
    onnx = None


@unittest.skipIf(onnx is None, "Host ONNX is required")
class DepthConvConversionTests(unittest.TestCase):
    def model(self):
        weights = np.array([[[[1.]], [[-0.5]]], [[[0.25]], [[0.75]]]], np.float32)
        bias = np.array([0.1, -0.2], np.float32)
        nodes = [h.make_node("Conv", ["x", "w", "b"], [f"y{i}"],
                             name=f"/depth_head/layer{i}/Conv", kernel_shape=[1, 1])
                 for i in range(28)]
        nodes += [h.make_node("Conv", ["x", "w", "b"], [f"head{i}"],
                              name=f"/depth_head/output_conv{i}/Conv") for i in range(3)]
        model = h.make_model(h.make_graph(nodes, "conv-test",
            [h.make_tensor_value_info("x", T.FLOAT, [1, 2, 2, 2])],
            [h.make_tensor_value_info(n.output[0], T.FLOAT, [1, 2, 2, 2]) for n in nodes],
            [nh.from_array(weights, "w"), nh.from_array(bias, "b")]),
            opset_imports=[h.make_opsetid("", 17)])
        model.ir_version = 8
        return model

    def test_integer_semantics_and_head_preservation(self):
        source = self.model()
        original_head = [n.SerializeToString() for n in source.graph.node[-3:]]
        model = quantize_convolutions(source, {"x": [-1, 1]})
        onnx.checker.check_model(model)
        ops = [n.op_type for n in model.graph.node]
        self.assertEqual(ops.count("ConvInteger"), 28)
        self.assertEqual(ops.count("Conv"), 3)
        self.assertEqual([n.SerializeToString() for n in model.graph.node if n.op_type == "Conv"], original_head)
        self.assertTrue(all(not n.domain for n in model.graph.node))
        x = np.array([[[[-2, -.3], [.2, 2]], [[.8, .4], [-.5, 0]]]], np.float32)
        actual = ReferenceEvaluator(model).run(None, {"x": x})
        arrays = {v.name: nh.to_array(v) for v in model.graph.initializer}
        q = next(n for n in model.graph.node if n.op_type == "QuantizeLinear")
        conv = next(n for n in model.graph.node if n.op_type == "ConvInteger")
        scale = next(n for n in model.graph.node if n.op_type == "Mul")
        xq = np.clip(np.rint(x / arrays[q.input[1]]) + arrays[q.input[2]], 0, 255).astype(np.int32)
        centered = xq - int(arrays[q.input[2]])
        weight = arrays[conv.input[1]][:, :, 0, 0].astype(np.int32) - int(arrays[conv.input[3]])
        expected = np.einsum('nchw,oc->nohw', centered, weight)
        expected = expected.astype(np.float32) * arrays[scale.input[1]] + arrays['b'][None, :, None, None]
        for value in actual[:28]:
            np.testing.assert_allclose(value, expected, rtol=0, atol=1e-7)

        # UINT8 storage is an exact offset encoding, not a second quantization.
        signed_model = copy.deepcopy(model)
        tensors = {v.name: v for v in signed_model.graph.initializer}
        for node in signed_model.graph.node:
            if node.op_type != "ConvInteger":
                continue
            name, zero = node.input[1], node.input[3]
            signed = (nh.to_array(tensors[name]).astype(np.int16) - 128).astype(np.int8)
            tensors[name].CopyFrom(nh.from_array(signed, name))
            tensors[zero].CopyFrom(nh.from_array(np.array(0, np.int8), zero))
        for a, b in zip(actual, ReferenceEvaluator(signed_model).run(None, {"x": x})):
            np.testing.assert_array_equal(a, b)

    def test_invalid_ranges_and_architecture_rejected(self):
        for r in ([float('nan'), 1], [1, -1]):
            with self.assertRaisesRegex(ValueError, "Invalid calibration"):
                quantize_convolutions(self.model(), {"x": r})
        model = self.model()
        del model.graph.node[0]
        with self.assertRaisesRegex(ValueError, "Expected 28"):
            quantize_convolutions(model, {"x": [-1, 1]})
