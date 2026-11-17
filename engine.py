"""SAM2.1 Tiny 512 inference; no GUI dependency, no profiling in timings."""

import gc
import time
from pathlib import Path

import cv2
import numpy as np

from settings import BACKENDS, CPU_THREADS, ENCODER, configure_npu, preprocess_image


def fit_rect(width, height, image_width, image_height):
    scale = min(width / image_width, height / image_height)
    w, h = image_width * scale, image_height * scale
    return (width - w) / 2, (height - h) / 2, w, h


def image_point(x, y, rect):
    left, top, width, height = rect
    if width <= 0 or height <= 0 or not (left <= x < left + width and top <= y < top + height):
        return None
    return ((x - left) / width, (y - top) / height)


class Session:
    def __init__(self, path, backend, npu=False):
        if backend not in BACKENDS:
            raise ValueError(f"Unknown backend: {backend}")
        if not Path(path).is_file():
            raise FileNotFoundError(f"Model not found: {path}. See README.md for setup.")
        if backend.startswith("ort"):
            import onnxruntime as ort

            providers = ["CPUExecutionProvider"]
            if npu:
                if "NeutronExecutionProvider" not in ort.get_available_providers():
                    raise RuntimeError("This ONNX Runtime has no NeutronExecutionProvider")
                providers.insert(0, "NeutronExecutionProvider")
            options = ort.SessionOptions()
            options.intra_op_num_threads = CPU_THREADS
            options.inter_op_num_threads = 1
            options.enable_profiling = False
            options.log_severity_level = 3
            self.net = ort.InferenceSession(str(path), options, providers=providers)
            if npu and "NeutronExecutionProvider" not in self.net.get_providers():
                raise RuntimeError("ORT did not activate NeutronExecutionProvider")
            self.run = lambda feed: self.net.run(None, feed)
            self.version = ort.__version__
        else:
            import ailia

            env = 0
            if npu:
                candidates = [
                    i
                    for i in range(ailia.get_environment_count())
                    if "Neutron" in ailia.get_environment(i).name
                ]
                if not candidates:
                    raise RuntimeError("The ailia SDK has no Neutron environment")
                env = candidates[0]
            self.net = ailia.Net(
                None,
                str(path),
                env_id=env,
                num_thread=CPU_THREADS,
                memory_mode=ailia.get_memory_mode(
                    reduce_constant=True,
                    ignore_input_with_initializer=True,
                    reduce_interstage=True,
                    reuse_interstage=True,
                ),
            )
            self.run = self.net.run
            self.version = ailia.get_version()


class Engine:
    def __init__(self, directory, backend):
        self.backend = backend
        self.directory = Path(directory)
        # Shared settings from the measured SAM512 QDQ configuration.
        configure_npu()
        start = time.perf_counter()
        self.encoder = Session(self.directory / ENCODER, backend, backend.endswith("_npu"))
        # Explicitly CPU on both runtimes: dynamic point prompts do not trigger
        # NPU recompilation. UI labels identify this hybrid execution policy.
        self.prompt = Session(self.directory / "prompt_512.onnx", backend)
        self.decoder = Session(self.directory / "decoder_512.onnx", backend)
        self.no_mem = (
            np.load(self.directory / "pretrained_weights.npz")["v2.1_hiera_t_no_mem_embed"]
            .astype(np.float32)
            .reshape(1, 256, 1, 1)
        )
        self.load_ms = (time.perf_counter() - start) * 1000
        self.path = None
        self.features = None

    def prepare(self, path):
        self.path = str(path)
        image = cv2.imread(self.path)
        if image is None:
            raise ValueError("Cannot read image: " + self.path)
        start = time.perf_counter()
        self.input = preprocess_image(image)[None]
        self.preprocess_ms = (time.perf_counter() - start) * 1000
        return self.encode()

    def encode(self):
        start = time.perf_counter()
        values = self.encoder.run({"input_image": self.input})
        self.encoder_ms = (time.perf_counter() - start) * 1000
        if len(values) != 7 or not all(np.isfinite(v).all() for v in values):
            raise RuntimeError("Unexpected/non-finite encoder outputs")
        self.features = dict(
            image_embeddings=values[6] + self.no_mem,
            high_res_features1=values[4],
            high_res_features2=values[5],
        )
        return self.encoder_ms

    def segment(self, points):
        if not points or len(points) > 64:
            raise ValueError("Use 1 to 64 foreground/background clicks")
        points = np.asarray(points, np.float32)
        if not np.isfinite(points).all() or (points[:, :2] < 0).any() or (points[:, :2] >= 1).any():
            raise ValueError("Click outside the image")
        start = time.perf_counter()
        p0 = time.perf_counter()
        sparse, dense, pe = self.prompt.run(
            dict(
                coords=points[None, :, :2] * 512,
                labels=points[None, :, 2].astype(np.int32),
                masks=np.zeros((1, 1, 128, 128), np.float32),
                masks_enable=np.array([0], np.int32),
            )
        )
        prompt_ms = (time.perf_counter() - p0) * 1000
        p0 = time.perf_counter()
        masks, scores, *_ = self.decoder.run(
            dict(
                self.features,
                image_pe=pe,
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
            )
        )
        decoder_ms = (time.perf_counter() - p0) * 1000
        if not np.isfinite(masks).all() or not np.isfinite(scores).all():
            raise RuntimeError("Non-finite decoder output")
        index = int(np.argmax(scores[0, 1:])) + 1
        mask = (masks[0, index] > 0).astype(np.uint8)
        return dict(
            mask=mask,
            score=float(scores[0, index]),
            prompt_ms=prompt_ms,
            decoder_ms=decoder_ms,
            click_ms=(time.perf_counter() - start) * 1000,
        )


def worker(directory, requests, responses):
    engine = None
    video_engine = None
    while True:
        job = requests.get()
        if job is None:
            return
        try:
            if job["action"] in ("video", "video_compare"):
                from video import VideoEngine

                engine = None
                gc.collect()
                for backend in BACKENDS if job["action"] == "video_compare" else [job["backend"]]:
                    responses.put(dict(kind="progress", text="Loading video models / " + backend))
                    if video_engine is None or video_engine.backend != backend:
                        video_engine = None
                        gc.collect()
                        video_engine = VideoEngine(Path(directory).parent, backend)
                    frames = sorted((Path(directory).parent / "bedroom").glob("[0-9]*.jpg"))
                    if len(frames) != 5:
                        raise ValueError("Expected exactly five bedroom frames")
                    for result in video_engine.track(frames, job["points"]):
                        responses.put(result)
                responses.put(
                    dict(kind="done", text="Tracking complete — Bedroom resets to frame 1.")
                )
                continue
            video_engine = None
            backends = BACKENDS if job["action"] == "benchmark" else [job["backend"]]
            for backend in backends:
                responses.put(dict(kind="progress", text="Loading / running " + backend))
                new_engine = engine is None or engine.backend != backend
                if new_engine:
                    engine = None
                    gc.collect()
                    engine = Engine(directory, backend)
                cached = engine.path == job["image"] and engine.features is not None
                if not cached:
                    engine.prepare(job["image"])
                if new_engine and backend.startswith("ailia"):
                    # The first encoder inference initializes the runtime. Keep
                    # it outside the displayed timing, then measure a warm run.
                    responses.put(
                        dict(kind="progress", text="Warm-up complete; measuring " + backend)
                    )
                    engine.encode()
                result = dict(
                    kind="result",
                    backend=backend,
                    load_ms=engine.load_ms,
                    encoder_ms=engine.encoder_ms,
                    cached=cached,
                    version=engine.encoder.version,
                    mode="warm / latest" if backend.startswith("ailia") else "first / latest",
                )
                if job["action"] == "benchmark":
                    # One explicit warm-up followed by 3 measured runs for every backend.
                    engine.encode()
                    points = job["points"] or [[0.5, 0.5, 1]]
                    engine.segment(points)
                    enc_times, click_times = [], []
                    for _ in range(3):
                        enc_times.append(engine.encode())
                        decoded = engine.segment(points)
                        click_times.append(decoded["click_ms"])
                    result.update(
                        decoded,
                        encoder_ms=float(np.median(enc_times)),
                        click_ms=float(np.median(click_times)),
                        encoder_runs_ms=enc_times,
                        click_runs_ms=click_times,
                        mode="warm median / 3 runs",
                    )
                elif job["points"]:
                    result.update(engine.segment(job["points"]))
                responses.put(result)
            responses.put(dict(kind="done"))
        except Exception as exc:
            import traceback

            traceback.print_exc()
            engine = None
            video_engine = None
            gc.collect()
            responses.put(dict(kind="error", text=f"{type(exc).__name__}: {exc}"))
