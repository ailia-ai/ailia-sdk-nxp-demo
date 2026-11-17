"""Shared preprocessing and environment defaults must preserve model inputs."""

import os
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from settings import configure_npu, preprocess_image


class SettingsTests(unittest.TestCase):
    def test_preprocessing_matches_original_pipeline(self):
        image = np.random.default_rng(0).integers(0, 256, (71, 103, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(cv2.resize(image, (512, 512)), cv2.COLOR_BGR2RGB)
        expected = rgb.astype(np.float32) / 255
        expected = (expected - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
            [0.229, 0.224, 0.225], np.float32
        )
        result = preprocess_image(image)
        np.testing.assert_array_equal(result, expected.transpose(2, 0, 1))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(result.flags.c_contiguous)

    def test_explicit_environment_is_preserved(self):
        with patch.dict(os.environ, {"AILIA_NEUTRON_DYNAMIC_ATTENTION": "0"}, clear=True):
            configure_npu()
            self.assertEqual(os.environ["AILIA_NEUTRON_DYNAMIC_ATTENTION"], "0")
            self.assertEqual(os.environ["NEUTRON_CMA_512SLOTS"], "1")


if __name__ == "__main__":
    unittest.main()
