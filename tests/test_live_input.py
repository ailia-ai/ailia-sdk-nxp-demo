"""Exercise real video capture without requiring a USB camera."""

import queue
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from live_input import LiveInput, find_camera
from scenes.sam2 import Engine
from scenes.yolox import YoloEngine


class LiveInputTests(unittest.TestCase):
    def test_usb_discovery_skips_internal_codec_nodes(self):
        with tempfile.TemporaryDirectory() as directory, patch("live_input.sys.platform", "linux"):
            root = Path(directory)
            for index, driver in ((0, "codec"), (5, "uvcvideo"), (4, "uvcvideo")):
                device = root / f"video{index}" / "device"
                device.mkdir(parents=True)
                (device / "driver").symlink_to(root / driver)
            self.assertEqual(find_camera(sysfs=root), 4)
            self.assertEqual(find_camera(1, sysfs=root), 1)

    def test_no_usb_camera_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory, patch("live_input.sys.platform", "linux"):
            with self.assertRaisesRegex(ValueError, "No USB camera"):
                find_camera(sysfs=Path(directory))

    def test_stalled_camera_has_bounded_timeout(self):
        capture = LiveInput.__new__(LiveInput)
        capture.output = queue.Queue()
        capture.process = Mock()
        capture.process.is_alive.return_value = True
        capture.last_frame = 0
        with patch("live_input.time.monotonic", return_value=11):
            self.assertEqual(capture.read()[0], "error")

    def test_close_terminates_blocked_driver(self):
        capture = LiveInput.__new__(LiveInput)
        capture.stop, capture.output, capture.process = Mock(), Mock(), Mock()
        capture.process.is_alive.side_effect = [True, False]
        capture.close()
        capture.stop.set.assert_called_once()
        capture.process.terminate.assert_called_once()
        capture.process.close.assert_called_once()
        capture.output.close.assert_called_once()

    def receive(self, source):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            item = source.read()
            if item is not None:
                return item
            time.sleep(0.02)
        self.fail("Capture timed out")

    def test_video_loops_and_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "input.avi")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
            self.assertTrue(writer.isOpened())
            for level in (30, 80, 150):
                writer.write(np.full((48, 64, 3), level, np.uint8))
            writer.release()
            capture = LiveInput(path)
            try:
                kind, frame = self.receive(capture)
                self.assertEqual(kind, "frame")
                self.assertEqual(frame.shape, (48, 64, 3))
                time.sleep(0.5)  # Beyond EOF: the file must loop, not grow a queue.
                self.assertEqual(self.receive(capture)[0], "frame")
            finally:
                capture.close()

    def test_missing_video_reports_error(self):
        capture = LiveInput("/tmp/nonexistent-ailia-camera-test.avi")
        try:
            self.assertEqual(self.receive(capture)[0], "error")
        finally:
            capture.close()

    def test_sam_frames_do_not_reuse_path_cache(self):
        engine = Engine.__new__(Engine)
        engine.encode = Mock(return_value=1)
        frame = np.zeros((48, 64, 3), np.uint8)
        engine.prepare(frame)
        first = engine.input.copy()
        engine.prepare(np.full_like(frame, 255))
        self.assertIsNone(engine.path)
        self.assertEqual(engine.encode.call_count, 2)
        self.assertFalse(np.array_equal(first, engine.input))

    def test_yolox_frames_reuse_session(self):
        session = Mock(version="test")
        session.run.return_value = [np.zeros((1, 2100, 85), np.float32)]
        with patch("runtime.Session", return_value=session) as factory:
            engine = YoloEngine("models", "ailia_npu")
            for level in (0, 255):
                engine.run(np.full((48, 64, 3), level, np.uint8))
            self.assertEqual(factory.call_count, 1)
            self.assertEqual(session.run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
