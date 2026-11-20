"""Bundled model examples remain readable and match the published checksums."""

import hashlib
import unittest

import cv2

from scenes import DEFAULT_IMAGES, ROOT, default_image
from scenes.grounding_dino_inputs import DEFAULT_PROMPT


class SceneImages(unittest.TestCase):
    def test_each_model_has_a_readable_default(self):
        for model in DEFAULT_IMAGES:
            with self.subTest(model=model):
                self.assertIsNotNone(cv2.imread(str(default_image(model))))

    def test_detection_image_checksums(self):
        checksums = dict(
            reversed(line.split()) for line in (ROOT / "SHA256SUMS").read_text().splitlines()
        )
        for model in ("grounding", "yolox", "vit"):
            path = default_image(model)
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                checksums[str(path.relative_to(ROOT))],
            )

    def test_grounding_prompt_matches_upstream_example(self):
        self.assertEqual(DEFAULT_PROMPT, "Horse. Clouds. Grasses. Sky. Hill.")
