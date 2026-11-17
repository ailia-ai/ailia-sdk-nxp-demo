import unittest

from engine import fit_rect, image_point


class Coordinates(unittest.TestCase):
    def test_letterbox(self):
        rect = fit_rect(1000, 500, 1000, 1000)
        self.assertEqual(rect, (250, 0, 500, 500))
        self.assertEqual(image_point(500, 250, rect), (0.5, 0.5))
        self.assertIsNone(image_point(249, 250, rect))
        self.assertIsNone(image_point(750, 250, rect))

    def test_wide_image(self):
        rect = fit_rect(512, 512, 1920, 1080)
        self.assertEqual(image_point(256, 256, rect), (0.5, 0.5))
        self.assertIsNone(image_point(256, 0, rect))

    def test_empty_area(self):
        self.assertIsNone(image_point(0, 0, (0, 0, 0, 0)))


if __name__ == "__main__":
    unittest.main()
