"""Numerical and validation tests for the offline integer conversion."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

spec = importlib.util.spec_from_file_location(
    "convert_qdq_matmul", Path(__file__).parents[1] / "scripts/convert_qdq_matmul.py"
)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def model(dynamic=False, nonuniform=False):
    rng = np.random.default_rng(7)
    w = rng.integers(-100, 100, (8, 5), dtype=np.int8)
    values = dict(
        sa=np.array(0.013, np.float32),
        za=np.array(127, np.uint8),
        sb=np.linspace(0.01, 0.05, 5, dtype=np.float32),
        zb=np.arange(5, dtype=np.int8) if nonuniform else np.zeros(5, np.int8),
    )
    inputs = [helper.make_tensor_value_info("a", TensorProto.UINT8, [2, 3, 8])]
    if dynamic:
        values.update(sb=np.array(0.021, np.float32), zb=np.array(121, np.uint8))
        inputs.append(helper.make_tensor_value_info("b", TensorProto.UINT8, [2, 8, 5]))
    else:
        values["b"] = w
    nodes = [
        helper.make_node("DequantizeLinear", ["a", "sa", "za"], ["af"]),
        helper.make_node("DequantizeLinear", ["b", "sb", "zb"], ["bf"], axis=1),
        helper.make_node("MatMul", ["af", "bf"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "test",
        inputs,
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 5])],
        [numpy_helper.from_array(v, k) for k, v in values.items()],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])


class ConversionTests(unittest.TestCase):
    def test_numerics(self):
        for dynamic in (False, True):
            with self.subTest(dynamic=dynamic):
                source = model(dynamic)
                original = source.SerializeToString()
                converted, stats = converter.convert(source)
                self.assertEqual(source.SerializeToString(), original)
                self.assertEqual(sum(stats.values()), 1)
                self.assertNotIn("DequantizeLinear", [n.op_type for n in converted.graph.node])
                rng = np.random.default_rng(11)
                feed = {"a": rng.integers(0, 256, (2, 3, 8), dtype=np.uint8)}
                if dynamic:
                    feed["b"] = rng.integers(0, 256, (2, 8, 5), dtype=np.uint8)
                expected = ReferenceEvaluator(source).run(None, feed)[0]
                actual = ReferenceEvaluator(converted).run(None, feed)[0]
                np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)

    def test_reject_nonuniform_zero_point(self):
        with self.assertRaisesRegex(ValueError, "Nonuniform"):
            converter.convert(model(nonuniform=True))

    def test_idempotent(self):
        converted, _ = converter.convert(model())
        twice, stats = converter.convert(converted)
        self.assertEqual(stats, {})
        self.assertEqual(converted.SerializeToString(), twice.SerializeToString())

    def test_preserve_shared_dequantized_output(self):
        source = model()
        source.graph.output.append(
            helper.make_tensor_value_info("af", TensorProto.FLOAT, [2, 3, 8])
        )
        converted, _ = converter.convert(source)
        self.assertTrue(
            any(
                n.op_type == "DequantizeLinear" and n.output[0] == "af"
                for n in converted.graph.node
            )
        )
        feed = {"a": np.full((2, 3, 8), 130, np.uint8)}
        np.testing.assert_array_equal(
            ReferenceEvaluator(source).run(None, feed)[1],
            ReferenceEvaluator(converted).run(None, feed)[1],
        )

    def test_preserve_unrelated_qdq(self):
        source = model()
        source.graph.node.append(helper.make_node("QuantizeLinear", ["y", "sa", "za"], ["q"]))
        source.graph.node.append(helper.make_node("DequantizeLinear", ["q", "sa", "za"], ["dq"]))
        source.graph.output.append(
            helper.make_tensor_value_info("dq", TensorProto.FLOAT, [2, 3, 5])
        )
        converted, _ = converter.convert(source)
        self.assertTrue(
            any(
                n.output[0] == "dq" and n.op_type == "DequantizeLinear"
                for n in converted.graph.node
            )
        )

    def test_opt_in_shared_keeps_other_consumers(self):
        source = model()
        source.graph.output.append(
            helper.make_tensor_value_info("af", TensorProto.FLOAT, [2, 3, 8])
        )
        converted, stats = converter.convert(source, include_shared=True)
        self.assertEqual(stats, {"constant_rhs": 1})
        self.assertTrue(any(n.op_type == "MatMulInteger" for n in converted.graph.node))
        feed = {"a": np.full((2, 3, 8), 130, np.uint8)}
        expected = ReferenceEvaluator(source).run(None, feed)
        actual = ReferenceEvaluator(converted).run(None, feed)
        np.testing.assert_allclose(actual[0], expected[0], atol=2e-5, rtol=2e-5)
        np.testing.assert_array_equal(actual[1], expected[1])

    def test_reject_non_last_axis(self):
        source = model()
        source.graph.node[1].attribute[0].i = 0
        with self.assertRaisesRegex(ValueError, "last-axis"):
            converter.convert(source)

    def test_nxp_benchmark_restores_rank_and_numerics(self):
        source = model()
        converted, _ = converter.convert(source, nxp_ep_benchmark=True)
        self.assertEqual(sum(n.op_type == "Reshape" for n in converted.graph.node), 2)
        integer = next(n for n in converted.graph.node if n.op_type == "MatMulInteger")
        cast = next(n for n in converted.graph.node if n.op_type == "Cast")
        self.assertEqual(cast.input[0], integer.output[0])
        feed = {"a": np.random.default_rng(8).integers(0, 256, (2, 3, 8), dtype=np.uint8)}
        expected = ReferenceEvaluator(source).run(None, feed)[0]
        actual = ReferenceEvaluator(converted).run(None, feed)[0]
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)

    def test_nxp_benchmark_leaves_dynamic_shapes_alone(self):
        converted, _ = converter.convert(model(dynamic=True), nxp_ep_benchmark=True)
        self.assertFalse(any(n.op_type == "Reshape" for n in converted.graph.node))


if __name__ == "__main__":
    unittest.main()
