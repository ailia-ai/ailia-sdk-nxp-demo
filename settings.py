"""Shared model, CPU-thread and NPU settings for the demo."""

import os

import cv2
import numpy as np

IMAGE_SIZE = 512
CPU_THREADS = 6
BACKENDS = ("ailia_cpu", "ailia_npu", "ort_cpu", "ort_npu")
ENCODER = "image_encoder_hiera_t_2.1_512_full_int8_layout.onnx"


def configure_npu():
    """Apply demo defaults without overriding explicit environment settings."""
    for name in (
        "NEUTRON_CMA_512SLOTS",
        "AILIA_NEUTRON_QDQ_STREAM_WEIGHTS",
        "AILIA_NEUTRON_DYNAMIC_ATTENTION",
    ):
        os.environ.setdefault(name, "1")


def preprocess_image(image):
    """Convert an OpenCV BGR image to normalized contiguous CHW float32."""
    rgb = cv2.cvtColor(cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    return np.ascontiguousarray(((rgb - mean) / std).transpose(2, 0, 1))
