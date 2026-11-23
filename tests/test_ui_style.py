"""Touch-sized controls and repeatable per-category overlays."""

import unittest

from scenes.yolox import LABELS
from ui_style import category_color, layout_metrics, stylesheet


class UiStyleTests(unittest.TestCase):
    def test_layout_scales_and_reflows(self):
        for width, height in ((1280, 720), (1920, 1080), (3840, 2160), (800, 1280)):
            m = layout_metrics(width, height)
            self.assertGreaterEqual(m["touch"], 48)
            self.assertGreaterEqual(m["font"], 18)
            self.assertEqual(m["portrait"], width < height)
            self.assertIn(b"font-size", stylesheet(m["font"], m["touch"]))
        self.assertGreater(layout_metrics(1920, 1080)["font"], layout_metrics(1280, 720)["font"])

    def test_category_colors_are_stable_and_distinct(self):
        colors = [category_color(label) for label in LABELS]
        self.assertEqual(len(set(colors)), 80)
        self.assertEqual(colors, [category_color(label) for label in LABELS])
        self.assertTrue(all(0 <= value <= 255 for color in colors for value in color))
