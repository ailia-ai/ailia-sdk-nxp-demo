"""Shared fixed-resolution preprocessing for Metric Depth Small."""

import cv2
import numpy as np

MODEL = "depth_anything_v2_metric_hypersim_vits_518_int8.onnx"
REVISION = "depth-coco-val2017-ql-v3"
SIZE = 518


def preprocess(image):
    rgb = cv2.cvtColor(
        cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB
    )
    value = rgb.astype(np.float32) / 255
    value = (value - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )
    return np.ascontiguousarray(value.transpose(2, 0, 1)[None])


def colorize(depth, width, height, maximum=20.0):
    """Shared selected color scale: dark near, bright far; do not alter depth data."""
    if maximum <= 0:
        raise ValueError("Display range must be positive")
    value = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)
    gray = np.rint(np.clip(value / maximum, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB)
