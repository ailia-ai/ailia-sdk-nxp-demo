"""Depth preprocessing, fixed color scale, session reuse and history isolation."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from depth import DepthEngine
from depth_inputs import REVISION, colorize, preprocess
from engine import bind_worker_lifetime
from history import append_history, load_history

ROOT = Path(__file__).resolve().parents[1]


class DepthTests(unittest.TestCase):
    def test_worker_terminates_with_parent_on_linux(self):
        libc = Mock()
        libc.prctl.return_value = 0
        with (
            patch("engine.sys.platform", "linux"),
            patch("engine.os.getppid", side_effect=[123, 123]),
            patch("engine.ctypes.CDLL", return_value=libc),
        ):
            self.assertTrue(bind_worker_lifetime())
            self.assertEqual(libc.prctl.call_args.args[0], 1)

    def test_orphaned_worker_does_not_open_runtime(self):
        with patch("engine.sys.platform", "linux"), patch("engine.os.getppid", return_value=1):
            self.assertFalse(bind_worker_lifetime())

    def test_worker_checks_parent_exit_race(self):
        libc = Mock()
        libc.prctl.return_value = 0
        with (
            patch("engine.sys.platform", "linux"),
            patch("engine.os.getppid", side_effect=[123, 1]),
            patch("engine.ctypes.CDLL", return_value=libc),
        ):
            self.assertFalse(bind_worker_lifetime())

    def test_preprocessing(self):
        image = np.zeros((100, 200, 3), np.uint8)
        x = preprocess(image)
        self.assertEqual(x.shape, (1, 3, 518, 518))
        self.assertEqual(x.dtype, np.float32)
        self.assertTrue(x.flags.c_contiguous)
        np.testing.assert_allclose(
            x[0, :, 0, 0], -np.array([0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225], rtol=1e-6
        )

    def test_fixed_color_scale(self):
        near = colorize(np.full((2, 2), 2, np.float32), 2, 2)
        image = colorize(np.array([[2, 10], [2, 20]], np.float32), 2, 2)
        np.testing.assert_array_equal(near[0, 0], image[0, 0])
        self.assertFalse(np.array_equal(image[0, 0], image[1, 1]))
        depth = np.full((2, 2), 2, np.float32)
        before = depth.copy()
        self.assertFalse(np.array_equal(near, colorize(depth, 2, 2, maximum=5)))
        np.testing.assert_array_equal(depth, before)

    def test_resident_session_and_dynamic_flags(self):
        session = Mock(version="test")
        session.run.return_value = [np.ones((1, 518, 518), np.float32)]
        with patch.dict(os.environ), patch("depth.Session", return_value=session) as factory:
            engine = DepthEngine(ROOT / "models", "ailia_npu")
            self.assertEqual(os.environ["AILIA_NEUTRON_DYNAMIC_MATMUL"], "1")
            engine.run(ROOT / "images/bedroom/00000.jpg")
            result = engine.run(ROOT / "images/bedroom/00000.jpg", compare=True)
            self.assertEqual(factory.call_count, 1)
            self.assertEqual(session.run.call_count, 4)
            self.assertEqual(len(result["inference_runs_ms"]), 3)
            self.assertEqual(result["depth"].shape, (518, 518))
            self.assertEqual(
                factory.call_args.kwargs["warmup_inputs"]["image"].shape, (1, 3, 518, 518)
            )

    def test_nonfinite_output_rejected(self):
        session = Mock(version="test")
        session.run.return_value = [np.full((1, 518, 518), np.nan, np.float32)]
        with patch.dict(os.environ), patch("depth.Session", return_value=session):
            engine = DepthEngine(ROOT / "models", "ailia_npu")
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                engine.run(ROOT / "images/bedroom/00000.jpg")

    def test_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.jsonl"
            append_history(path, "ailia_npu", ["1"] * 5, "bedroom.jpg", model="depth")
            append_history(path, "ailia_npu", ["2"] * 5, "truck.jpg", model="sam")
            records = load_history(path, limit=1)
            self.assertEqual({x["model"] for x in records}, {"sam", "depth"})
            self.assertEqual(records[0]["model_revision"], REVISION)
