"""Input-mode availability is independent of GTK and hardware."""

import unittest

from app import input_mode_for_model


class InputModes(unittest.TestCase):
    def test_sam_has_all_three_modes(self):
        for mode in ("image", "video", "camera"):
            self.assertEqual(input_mode_for_model("sam", mode), mode)

    def test_other_models_disable_video(self):
        for model in ("yolox", "grounding", "depth", "vit"):
            self.assertEqual(input_mode_for_model(model, "video"), "image")
            self.assertEqual(input_mode_for_model(model, "camera"), "camera")
            self.assertEqual(input_mode_for_model(model, "image"), "image")

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            input_mode_for_model("sam", "unknown")
