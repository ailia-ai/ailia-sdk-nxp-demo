"""Model-specific scenes and their bundled example images."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES = {
    "sam": "truck.jpg",
    "grounding": "grounding_dino/demo.jpg",
    "depth": "bedroom/00000.jpg",
    "yolox": "yolox/input.jpg",
}


def default_image(model):
    return ROOT / "images" / DEFAULT_IMAGES[model]
