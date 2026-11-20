"""All four backends share exactly one Open-time dummy inference."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from runtime import Session
from settings import ENCODER


class OpenWarmupTests(unittest.TestCase):
    def test_each_backend_warms_once_with_same_feed(self):
        path = Path(__file__).resolve().parents[1] / "models" / ENCODER
        feed = {"input_image": np.zeros((1, 3, 512, 512), np.float32)}
        for backend in ("ailia_cpu", "ailia_npu", "ort_cpu", "ort_npu"):
            with self.subTest(backend=backend):
                net = Mock()
                net.run.return_value = [np.zeros(1, np.float32)]
                net.get_providers.return_value = [
                    "NeutronExecutionProvider",
                    "CPUExecutionProvider",
                ]
                ort = SimpleNamespace(
                    __version__="test",
                    SessionOptions=Mock(),
                    InferenceSession=Mock(return_value=net),
                    get_available_providers=net.get_providers,
                )
                ailia = SimpleNamespace(
                    Net=Mock(return_value=net),
                    get_version=lambda: "test",
                    get_environment_count=lambda: 1,
                    get_environment=lambda i: SimpleNamespace(name="Neutron"),
                )
                with (
                    patch.dict(sys.modules, {"onnxruntime": ort, "ailia": ailia}),
                    patch("warmup.inputs_for_model", return_value=feed) as inputs,
                ):
                    Session(path, backend, backend.endswith("_npu"))
                inputs.assert_called_once_with(ENCODER)
                net.run.assert_called_once()
                expected_args = (None, feed) if backend.startswith("ort") else (feed,)
                self.assertEqual(len(net.run.call_args.args), len(expected_args))
                self.assertIs(net.run.call_args.args[-1], feed)
