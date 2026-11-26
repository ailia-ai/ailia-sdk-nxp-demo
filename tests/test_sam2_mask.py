"""SAM image/camera postprocessing keeps logits until original-size interpolation."""

import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from scenes.sam2 import Engine, postprocess_mask


class SamMaskTests(unittest.TestCase):
    def test_interpolates_logits_before_threshold(self):
        logits = np.array([[-1, 3], [-1, 3]], np.float32)
        mask = postprocess_mask(logits, (6, 8))
        expected = cv2.resize(logits, (8, 6), interpolation=cv2.INTER_LINEAR) > 0
        np.testing.assert_array_equal(mask, expected)
        nearest = cv2.resize((logits > 0).astype(np.uint8), (8, 6), interpolation=cv2.INTER_NEAREST)
        self.assertFalse(np.array_equal(mask, nearest))
        self.assertEqual(mask.dtype, np.uint8)

    def test_same_size_keeps_zero_threshold(self):
        logits = np.array([[-1, 0, 1]], np.float32)
        np.testing.assert_array_equal(postprocess_mask(logits, (1, 3)), [[0, 0, 1]])

    def test_engine_returns_original_size_selected_mask(self):
        engine = Engine.__new__(Engine)
        engine.image_shape = (6, 8)
        engine.features = {}
        engine.prompt = Mock()
        engine.prompt.run.return_value = (None, None, None)
        engine.decoder = Mock()
        masks = np.zeros((1, 4, 2, 2), np.float32)
        masks[0, 2] = [[-1, 3], [-1, 3]]
        engine.decoder.run.return_value = (masks, np.array([[1, 0.1, 0.9, 0.2]]))
        result = engine.segment([[0.5, 0.5, 1]])
        np.testing.assert_array_equal(result["mask"], postprocess_mask(masks[0, 2], (6, 8)))
