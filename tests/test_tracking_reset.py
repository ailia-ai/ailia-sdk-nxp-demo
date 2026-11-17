"""Tracking completion restores click input without discarding history/cache."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from app import reset_tracking_view


class TrackingResetTests(unittest.TestCase):
    def test_reset_preserves_results_and_runtime(self):
        runner, records, results = object(), [{"frame": 4}], {"ailia_npu": "measured"}
        view = SimpleNamespace(
            video_frame=4,
            image_path="bedroom/00004.jpg",
            points=[[0.2, 0.6, 1]],
            mask=object(),
            load_image=Mock(),
            runner=runner,
            video_records=records,
            results=results,
            video_mode=True,
        )
        reset_tracking_view(view)
        self.assertEqual(view.video_frame, 0)
        self.assertEqual(Path(view.image_path).name, "00000.jpg")
        self.assertTrue(Path(view.image_path).is_file())
        self.assertEqual(view.points, [])
        self.assertIsNone(view.mask)
        self.assertIs(view.runner, runner)
        self.assertIs(view.video_records, records)
        self.assertIs(view.results, results)
        self.assertTrue(view.video_mode)
        view.load_image.assert_called_once_with()
