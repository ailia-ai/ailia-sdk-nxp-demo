import unittest

import numpy as np

from scenes.vit import classify, preprocess


class VitTests(unittest.TestCase):
    def test_rgb_normalization(self):
        image = np.zeros((10, 20, 3), np.uint8)
        image[:, :, 2] = 255
        tensor = preprocess(image)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertEqual(tensor.dtype, np.float32)
        np.testing.assert_array_equal(tensor[0, :, 0, 0], [1, -1, -1])

    def test_top_five(self):
        values = np.zeros((1, 1000), np.float32)
        values[0, 207] = 10
        result = classify(values)
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0]["index"], 207)
        self.assertGreater(result[0]["score"], 0.9)

    def test_invalid_logits(self):
        for values in (np.zeros(10), np.full(1000, np.nan)):
            with self.assertRaises(ValueError):
                classify(values)
