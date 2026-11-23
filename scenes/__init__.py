"""Model-specific scenes and their bundled example images."""

from importlib import import_module
from pathlib import Path

from settings import MODEL_REVISION

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES = {
    "sam": "truck.jpg",
    "grounding": "grounding_dino/demo.jpg",
    "depth": "bedroom/00000.jpg",
    "yolox": "yolox/input.jpg",
    "vit": "vit/input.jpg",
}

# Original image URLs cited by the corresponding ailia-models READMEs.
IMAGE_SOURCES = {
    "truck.jpg": "https://github.com/facebookresearch/segment-anything/blob/main/notebooks/images/truck.jpg",
    "bedroom": "https://github.com/facebookresearch/segment-anything-2/tree/main/notebooks/videos/bedroom",
    "grounding_dino/demo.jpg": "https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/main/assets/demo7.jpg",
    "yolox/input.jpg": "https://pixabay.com/ja/photos/%E3%83%AD%E3%83%B3%E3%83%89%E3%83%B3%E5%B8%82-%E9%8A%80%E8%A1%8C-%E3%83%AD%E3%83%B3%E3%83%89%E3%83%B3-4481399/",
    "vit/input.jpg": "https://pixabay.com/photos/labrador-retriever-dog-pet-labrador-6244939/",
}


def image_source(path):
    try:
        relative = Path(path).resolve().relative_to(ROOT / "images")
    except ValueError:
        return None
    key = "bedroom" if relative.parent.as_posix() == "bedroom" else relative.as_posix()
    return IMAGE_SOURCES.get(key)


def default_image(model):
    return ROOT / "images" / DEFAULT_IMAGES[model]


def model_revision(model):
    """Resolve scene revisions without loading every inference engine at startup."""
    if model == "sam":
        return MODEL_REVISION
    module = {
        "grounding": "grounding_dino_inputs",
        "depth": "depth_anything_v2_inputs",
        "yolox": "yolox",
        "vit": "vit",
    }[model]
    return import_module(f"scenes.{module}").REVISION
