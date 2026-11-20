"""Evaluate YOLOX on all COCO val2017 images, or score saved detections on a host."""

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolox import MODEL, YoloEngine, decode, preprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--backend", choices=("ailia_cpu", "ailia_npu", "ort_cpu", "ort_npu"), default="ailia_npu"
    )
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    annotation = json.loads(args.annotations.read_text())
    entries = sorted(annotation["images"], key=lambda e: e["id"])
    if len(entries) != 5000 or len({e["id"] for e in entries}) != 5000:
        parser.error("Expected the complete COCO val2017 annotation file")
    if args.score_only:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        reference = COCO(str(args.annotations))
        evaluation = COCOeval(reference, reference.loadRes(str(args.detections)), "bbox")
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()
        metrics = dict(
            images=5000,
            AP=float(evaluation.stats[0]),
            AP50=float(evaluation.stats[1]),
            AP75=float(evaluation.stats[2]),
            note="COCO val2017 is also used for calibration, not an independent test set.",
        )
        args.detections.with_suffix(".metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n"
        )
        return
    if args.images is None:
        parser.error("--images is required unless --score-only is used")
    categories = sorted(e["id"] for e in annotation["categories"])
    engine = YoloEngine(args.models, args.backend, args.model)
    results = []
    for index, entry in enumerate(entries):
        image = cv2.imread(str(args.images / entry["file_name"]))
        if image is None or image.shape[:2] != (entry["height"], entry["width"]):
            raise ValueError(f"Invalid COCO image: {entry['file_name']}")
        tensor, ratio = preprocess(image)
        output = engine.session.run({"images": tensor})[0]
        boxes, scores, classes = decode(output, threshold=0.001, nms=0.65)
        for box, score, category in zip(boxes / ratio, scores, classes):
            results.append(
                dict(
                    image_id=entry["id"],
                    category_id=categories[int(category)],
                    bbox=[
                        float(box[0]),
                        float(box[1]),
                        float(box[2] - box[0]),
                        float(box[3] - box[1]),
                    ],
                    score=float(score),
                )
            )
        if (index + 1) % 100 == 0:
            print(f"{index + 1}/5000", flush=True)
    args.detections.parent.mkdir(parents=True, exist_ok=True)
    args.detections.write_text(json.dumps(results))


if __name__ == "__main__":
    main()
