"""Single-object SAM2.1 memory tracking, shared by GUI and calibration tests."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from engine import ENCODER, Session
from sam2_video_predictor import SAM2VideoPredictor
from settings import BACKENDS, configure_npu, preprocess_image


class TimedSession:
    def __init__(self, path, backend, npu=False, capture=None, memory_bank=False, cache=None):
        self.session = (
            cache.get(path, backend, npu) if cache is not None else Session(path, backend, npu)
        )
        self.ms = 0.0
        self.calls = 0
        self.capture = capture
        self.memory_bank = memory_bank

    def run(self, feed):
        if self.memory_bank:
            # The demo contains five frames, so at most four prior spatial
            # memories and four object pointers (4 tokens each) are required.
            # Masked padding keeps the NPU graph shape fixed across frames.
            feed = dict(feed)
            for key, value in feed.items():
                if key.endswith("_1") or key.endswith("_2"):
                    capacity = 4096 if key.endswith("_1") else 16
                    if value.shape[0] > capacity:
                        raise ValueError("This memory bank supports the five-frame demo only")
                    padded = np.zeros((capacity, *value.shape[1:]), value.dtype)
                    padded[: value.shape[0]] = value
                    feed[key] = padded
        if self.capture is not None:
            self.capture.mkdir(parents=True, exist_ok=True)
            np.savez(self.capture / f"feed_{self.calls:03d}.npz", **feed)
        start = time.perf_counter()
        outputs = self.session.run(feed)
        self.ms += (time.perf_counter() - start) * 1000
        self.calls += 1
        return outputs


class VideoEngine:
    def __init__(
        self,
        directory,
        backend,
        memory_model="memory_attention_512_qdq.onnx",
        capture=None,
        cache=None,
    ):
        configure_npu()
        self.directory = Path(directory)
        self.backend = backend
        models = self.directory / "models"
        self.nets = {}
        for key, path in {
            "image_encoder": models / ENCODER,
            "prompt_encoder": models / "prompt_512.onnx",
            "mask_decoder": models / "decoder_512.onnx",
            "memory_attention": models / memory_model,
            "memory_encoder": models / "memory_encoder_512.onnx",
            "mlp": models / "mlp_original.onnx",
            "obj_ptr_tpos_proj": models / "obj_ptr_tpos_proj_original.onnx",
        }.items():
            self.nets[key] = TimedSession(
                path,
                backend,
                backend.endswith("_npu") and key in ("image_encoder", "memory_attention"),
                capture if key == "memory_attention" else None,
                key == "memory_attention",
                cache=cache,
            )

    def track(self, frames, points):
        if len(frames) != 5:
            raise ValueError("This demo requires exactly five frames.")
        if not points:
            raise ValueError("Click the target on the first bedroom frame before tracking.")
        predictor = SAM2VideoPredictor(onnx=False, benchmark=False, legacy=False)
        state = predictor.init_state(version="2.1", model_type="hiera_t", image_size=512)
        points = np.asarray(points, np.float32)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or len(points) > 64
            or not np.isfinite(points).all()
            or (points[:, :2] < 0).any()
            or (points[:, :2] >= 1).any()
            or not np.isin(points[:, 2], [0, 1]).all()
        ):
            raise ValueError("Use 1 to 64 normalized points with labels 0 or 1.")
        for index, path in enumerate(frames):
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Cannot read frame: {path}")
            h, w = image.shape[:2]
            tensor = preprocess_image(image)
            before = {key: net.ms for key, net in self.nets.items()}
            start = time.perf_counter()
            predictor.append_image(state, tensor, h, w, self.nets["image_encoder"])
            if index == 0:
                predictor.add_new_points_or_box(
                    state,
                    frame_idx=0,
                    obj_id=1,
                    points=points[:, :2] * [w, h],
                    labels=points[:, 2].astype(np.int32),
                    **self.nets,
                )
                predictor.propagate_in_video_preflight(state, **self.nets)
            _, _, logits = predictor.propagate_in_video(state, frame_idx=index, **self.nets)
            times = {key: net.ms - before[key] for key, net in self.nets.items()}
            if not np.isfinite(logits).all():
                raise RuntimeError(f"Non-finite tracking output at frame {index}")
            yield dict(
                kind="video_frame",
                backend=self.backend,
                frame=index,
                image=str(path),
                mask=(logits[0, 0] > 0).astype(np.uint8),
                times_ms=times,
                total_ms=(time.perf_counter() - start) * 1000,
                points=points.tolist(),
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=BACKENDS, default="ort_cpu")
    parser.add_argument("--memory-model", default="memory_attention_512_qdq.onnx")
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = Path(__file__).parent
    args.output.mkdir(parents=True, exist_ok=True)
    engine = VideoEngine(base, args.backend, args.memory_model, args.capture)
    records = []
    for result in engine.track(
        sorted((base / "bedroom").glob("[0-9]*.jpg")),
        [[210 / 960, 350 / 540, 1], [250 / 960, 220 / 540, 1]],
    ):
        cv2.imwrite(str(args.output / f"frame_{result['frame']}.png"), result.pop("mask") * 255)
        records.append(result)
        print(json.dumps(result), flush=True)
        (args.output / "results.json").write_text(json.dumps(records, indent=2))
    if args.backend.startswith("ailia"):
        for name in ("image_encoder", "memory_attention"):
            (args.output / f"{name}_summary.txt").write_text(
                engine.nets[name].session.net.get_summary()
            )


if __name__ == "__main__":
    main()
