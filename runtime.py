"""Shared inference sessions and model worker; no GUI dependency or profiling."""

import ctypes
import gc
import os
import signal
import sys
from pathlib import Path

import numpy as np

from settings import BACKENDS, CPU_THREADS


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
    def __init__(self, path, backend, npu=False, warmup_inputs=None, validate_outputs=None):
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
                memory_mode=11,
            )
            self.run = self.net.run
            self.version = ailia.get_version()
        from warmup import inputs_for_model

        outputs = self.run(
            inputs_for_model(Path(path).name) if warmup_inputs is None else warmup_inputs
        )
        valid = (
            validate_outputs(outputs)
            if validate_outputs
            else all(np.isfinite(value).all() for value in outputs)
        )
        if not valid:
            raise RuntimeError(f"Non-finite Open-time warm-up output: {path}")


class SessionCache:
    """Sessions for one selected backend, shared by image and video engines."""

    def __init__(self, backend):
        self.backend = backend
        self.sessions = {}

    def get(self, path, backend, npu=False):
        if backend != self.backend:
            raise ValueError("A session cache cannot mix backends")
        key = (Path(path).resolve(), npu)
        if key not in self.sessions:
            self.sessions[key] = Session(path, backend, npu)
        return self.sessions[key]


def bind_worker_lifetime():
    """Release native NPU allocations even if Wayland terminates the GUI abruptly."""
    if not sys.platform.startswith("linux"):
        return True
    parent = os.getppid()
    if parent == 1:
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return os.getppid() == parent


def worker(directory, requests, responses):
    from scenes.sam2 import Engine

    if not bind_worker_lifetime():
        return
    engine = None
    video_engine = None
    cache = None
    grounding = None
    depth_engine = None
    yolox_engine = None
    vit_engine = None

    def select_backend(backend):
        nonlocal engine, video_engine, cache
        if cache is None or cache.backend != backend:
            engine = video_engine = cache = None
            gc.collect()
            cache = SessionCache(backend)

    def backend_order(job, compare):
        # Leave the UI-selected runtime resident after Compare all.
        if job.get("single_backend"):
            return [job["backend"]]
        return (
            [b for b in BACKENDS if b != job["backend"]] + [job["backend"]]
            if compare
            else [job["backend"]]
        )

    while True:
        job = requests.get()
        if job is None:
            return
        try:
            if job.get("model") == "vit":
                from scenes.vit import VitEngine

                for backend in backend_order(job, job["action"] == "benchmark"):
                    responses.put(dict(kind="progress", text="Loading / running ViT / " + backend))
                    if vit_engine is None or vit_engine.backend != backend:
                        vit_engine = None
                        gc.collect()
                        vit_engine = VitEngine(directory, backend)
                    result = vit_engine.run(job["image"], compare=job["action"] == "benchmark")
                    responses.put(
                        dict(
                            kind="result",
                            model="vit",
                            backend=backend,
                            action=job["action"],
                            **result,
                        )
                    )
                responses.put(dict(kind="done", text="Ready — press Classify or open an image."))
                continue
            if job.get("model") == "yolox":
                from scenes.yolox import YoloEngine

                for backend in backend_order(job, job["action"] == "benchmark"):
                    responses.put(
                        dict(kind="progress", text="Loading / running YOLOX-S / " + backend)
                    )
                    if yolox_engine is None or yolox_engine.backend != backend:
                        yolox_engine = None
                        gc.collect()
                        yolox_engine = YoloEngine(directory, backend)
                    result = yolox_engine.run(job["image"], compare=job["action"] == "benchmark")
                    responses.put(
                        dict(
                            kind="result",
                            model="yolox",
                            backend=backend,
                            action=job["action"],
                            **result,
                        )
                    )
                responses.put(dict(kind="done", text="Ready — press Detect."))
                continue
            if job.get("model") == "depth":
                from scenes.depth_anything_v2 import DepthEngine

                for backend in backend_order(job, job["action"] == "benchmark"):
                    responses.put(
                        dict(kind="progress", text="Loading / running Depth / " + backend)
                    )
                    if depth_engine is None or depth_engine.backend != backend:
                        depth_engine = None
                        gc.collect()
                        depth_engine = DepthEngine(directory, backend)
                    result = depth_engine.run(job["image"], compare=job["action"] == "benchmark")
                    responses.put(
                        dict(
                            kind="result",
                            model="depth",
                            backend=backend,
                            action=job["action"],
                            **result,
                        )
                    )
                responses.put(
                    dict(kind="done", text="Ready — open an image or press Estimate depth.")
                )
                continue
            if job.get("model", "sam") == "grounding":
                from scenes.grounding_dino import GroundingEngine

                for backend in backend_order(job, job["action"] == "benchmark"):
                    responses.put(
                        dict(kind="progress", text="Loading / running Grounding DINO / " + backend)
                    )
                    if grounding is None or grounding.backend != backend:
                        grounding = None
                        gc.collect()
                        grounding = GroundingEngine(directory, backend)
                    result = grounding.run(
                        job["image"], job["caption"], compare=job["action"] == "benchmark"
                    )
                    responses.put(
                        dict(
                            kind="result",
                            model="grounding",
                            backend=backend,
                            action=job["action"],
                            **result,
                        )
                    )
                responses.put(dict(kind="done", text="Ready — edit the prompt and press Detect."))
                continue
            if job["action"] in ("video", "video_compare"):
                from scenes.sam2_video import VideoEngine

                for backend in backend_order(job, job["action"] == "video_compare"):
                    select_backend(backend)
                    responses.put(dict(kind="progress", text="Loading video models / " + backend))
                    if video_engine is None or video_engine.backend != backend:
                        video_engine = VideoEngine(Path(directory).parent, backend, cache=cache)
                    frames = sorted(
                        (Path(directory).parent / "images" / "bedroom").glob("[0-9]*.jpg")
                    )
                    if len(frames) != 5:
                        raise ValueError("Expected exactly five bedroom frames")
                    for result in video_engine.track(frames, job["points"]):
                        responses.put(result)
                responses.put(dict(kind="done", action=job["action"], text="Tracking complete."))
                continue
            backends = backend_order(job, job["action"] == "benchmark")
            for backend in backends:
                select_backend(backend)
                responses.put(dict(kind="progress", text="Loading / running " + backend))
                new_engine = engine is None or engine.backend != backend
                if new_engine:
                    engine = Engine(directory, backend, cache=cache)
                if job.get("video_mode") and video_engine is None:
                    from scenes.sam2_video import VideoEngine

                    video_engine = VideoEngine(Path(directory).parent, backend, cache=cache)
                cached = (
                    not isinstance(job["image"], np.ndarray)
                    and engine.path == job["image"]
                    and engine.features is not None
                )
                if not cached:
                    engine.prepare(job["image"])
                result = dict(
                    kind="result",
                    backend=backend,
                    load_ms=engine.load_ms,
                    encoder_ms=engine.encoder_ms,
                    cached=cached,
                    version=engine.encoder.version,
                    mode="warm / latest",
                    action=job["action"],
                )
                if job["action"] == "benchmark":
                    # Both runtimes already performed the same dummy inference at Open.
                    points = job["points"] or [[0.5, 0.5, 1]]
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
            cache = None
            grounding = None
            yolox_engine = None
            vit_engine = None
            gc.collect()
            responses.put(
                dict(kind="error", action=job["action"], text=f"{type(exc).__name__}: {exc}")
            )
