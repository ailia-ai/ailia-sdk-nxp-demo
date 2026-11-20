"""YOLOX-S 320 UINT8 features / INT8 weights, shared demo and evaluation code."""

import time
from pathlib import Path

import cv2
import numpy as np

MODEL = "yolox_s_320_int8.onnx"
REVISION = "yolox-s-coco-val2017-v1"
SIZE = 320
LABELS = (
    "person bicycle car motorcycle airplane bus train truck boat traffic_light "
    "fire_hydrant stop_sign parking_meter bench bird cat dog horse sheep cow "
    "elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee "
    "skis snowboard sports_ball kite baseball_bat baseball_glove skateboard "
    "surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl banana "
    "apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch "
    "potted_plant bed dining_table toilet tv laptop mouse remote keyboard cell_phone "
    "microwave oven toaster sink refrigerator book clock vase scissors teddy_bear "
    "hair_drier toothbrush"
).split()


def preprocess(image):
    ratio = min(SIZE / image.shape[0], SIZE / image.shape[1])
    scaled = cv2.resize(image, (int(image.shape[1] * ratio), int(image.shape[0] * ratio)))
    canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
    canvas[: scaled.shape[0], : scaled.shape[1]] = scaled
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32), ratio


def decode(output, threshold=0.4, nms=0.45):
    values = np.asarray(output)[0]
    if values.shape != (2100, 85) or not np.isfinite(values).all():
        raise ValueError("Invalid YOLOX output")
    grids, strides = [], []
    for stride in (8, 16, 32):
        y, x = np.mgrid[: SIZE // stride, : SIZE // stride]
        grids.append(np.stack([x, y], axis=-1).reshape(-1, 2))
        strides.append(np.full((x.size, 1), stride))
    strides = np.concatenate(strides)
    xy = (values[:, :2] + np.concatenate(grids)) * strides
    wh = np.exp(values[:, 2:4]) * strides
    boxes = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)
    probabilities = values[:, 4:5] * values[:, 5:]
    classes, scores = probabilities.argmax(axis=1), probabilities.max(axis=1)
    order = np.where(scores >= threshold)[0]
    order = order[np.argsort(scores[order])[::-1]]
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        rest = order[1:]
        extent = np.maximum(
            0, np.minimum(boxes[i, 2:], boxes[rest, 2:]) - np.maximum(boxes[i, :2], boxes[rest, :2])
        )
        intersection = extent.prod(axis=1)
        union = (
            np.maximum(0, boxes[i, 2:] - boxes[i, :2]).prod()
            + np.maximum(0, boxes[rest, 2:] - boxes[rest, :2]).prod(axis=1)
            - intersection
        )
        order = rest[intersection / np.maximum(union, 1e-9) <= nms]
    return boxes[keep], scores[keep], classes[keep]


class YoloEngine:
    def __init__(self, directory, backend, model=MODEL):
        from engine import Session
        self.backend = backend
        start = time.perf_counter()
        self.session = Session(
            Path(directory) / model,
            backend,
            npu=backend.endswith("_npu"),
            warmup_inputs={"images": np.zeros((1, 3, SIZE, SIZE), np.float32)},
        )
        self.load_ms = (time.perf_counter() - start) * 1000

    def run(self, path, compare=False):
        image = path if isinstance(path, np.ndarray) else cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read {path}")
        tensor, ratio = preprocess(image)
        times = []
        for _ in range(3 if compare else 1):
            start = time.perf_counter()
            outputs = self.session.run({"images": tensor})
            times.append((time.perf_counter() - start) * 1000)
        boxes, scores, classes = decode(outputs[0])
        boxes /= ratio * np.array([image.shape[1], image.shape[0]] * 2)
        detections = [
            dict(box=box.tolist(), score=float(score), label=LABELS[int(category)])
            for box, score, category in zip(boxes, scores, classes)
        ]
        return dict(
            encoder_ms=float(np.median(times)),
            click_ms=0.0,
            inference_runs_ms=times,
            detections=detections,
            load_ms=self.load_ms,
            version=self.session.version,
            mode="detection median / 3 runs" if compare else "detection / latest",
        )
