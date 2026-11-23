"""Bundled model examples remain readable and match the published checksums."""

import hashlib
import unittest

import cv2

from scenes import DEFAULT_IMAGES, ROOT, default_image, image_source
from scenes.grounding_dino_inputs import DEFAULT_PROMPT


class SceneImages(unittest.TestCase):
    def test_original_image_sources(self):
        for model in DEFAULT_IMAGES:
            self.assertTrue(image_source(default_image(model)).startswith("https://"))
        self.assertIn("4481399", image_source(default_image("yolox")))
        self.assertIn("6244939", image_source(default_image("vit")))
        self.assertIn("demo7.jpg", image_source(default_image("grounding")))
        self.assertEqual(
            image_source(default_image("depth")), image_source(ROOT / "images/bedroom/00004.jpg")
        )
        self.assertIsNone(image_source("/tmp/input.jpg"))

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
