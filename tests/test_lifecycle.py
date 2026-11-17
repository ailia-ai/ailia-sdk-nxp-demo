"""Model Open warm-up, shared sessions and durable comparison history."""

import tempfile
import unittest
from pathlib import Path
from queue import SimpleQueue
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from engine import Engine, Session, SessionCache, worker
from history import append_history, load_history
from settings import ENCODER
from video import VideoEngine

ROOT = Path(__file__).resolve().parents[1]


class LifecycleTests(unittest.TestCase):
    def test_worker_reuses_engine_until_backend_switch(self):
        requests, responses = SimpleQueue(), SimpleQueue()
        made = []

        def create(_directory, backend, cache):
            instance = Mock(backend=backend, path=None, features=None, load_ms=1, encoder_ms=1)
            instance.encoder.version = "test"
            instance.segment.return_value = {}

            def prepare(path):
                instance.path, instance.features = path, {}

            instance.prepare.side_effect = prepare
            made.append(instance)
            return instance

        for backend in ("ailia_npu", "ailia_npu", "ort_npu", "ort_npu"):
            requests.put(
                dict(action="segment", backend=backend, image="truck.jpg", points=[[0.5, 0.5, 1]])
            )
        requests.put(None)
        with patch("engine.Engine", side_effect=create) as factory:
            worker(ROOT / "models", requests, responses)
        self.assertEqual(factory.call_count, 2)
        for instance in made:
            self.assertEqual(instance.prepare.call_count, 1)
            self.assertEqual(instance.segment.call_count, 2)
            instance.encode.assert_not_called()

    def test_open_warms_once_with_mode_11(self):
        net = Mock()
        net.run.return_value = [np.ones(1, np.float32)]
        api = SimpleNamespace(Net=Mock(return_value=net), get_version=lambda: "test")
        with patch.dict("sys.modules", {"ailia": api}):
            session = Session(ROOT / "models" / ENCODER, "ailia_cpu")
            self.assertEqual(api.Net.call_args.kwargs["memory_mode"], 11)
            self.assertEqual(net.run.call_count, 1)
            session.run({"input_image": np.zeros((1, 3, 512, 512), np.float32)})
            self.assertEqual(net.run.call_count, 2)

    def test_image_video_share_sessions(self):
        cache = SessionCache("ort_cpu")
        with patch("engine.Session") as factory:
            image = Engine(ROOT / "models", "ort_cpu", cache=cache)
            video = VideoEngine(ROOT, "ort_cpu", cache=cache)
            self.assertEqual(factory.call_count, 7)
            self.assertIs(image.encoder, video.nets["image_encoder"].session)
            self.assertIs(image.prompt, video.nets["prompt_encoder"].session)
            self.assertIs(image.decoder, video.nets["mask_decoder"].session)
            cache.get(ROOT / "models" / ENCODER, "ort_cpu")
            self.assertEqual(factory.call_count, 7)
        with self.assertRaises(ValueError):
            cache.get(ROOT / "models" / ENCODER, "ailia_cpu")

    def test_history_keeps_previous_backends_and_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.jsonl"
            values = ["1", "2", "3", "6", "video frame 5/5"]
            for backend in ("ailia_npu", "ort_npu", "ailia_npu"):
                append_history(path, backend, values, "00004.jpg")
            records = load_history(path)
            self.assertEqual([r["backend"] for r in records], ["ailia_npu", "ort_npu", "ailia_npu"])
            self.assertEqual(len(load_history(path, limit=2)), 2)
            self.assertEqual(len(path.read_text().splitlines()), 3)
            for _ in range(5):
                append_history(path, "ailia_npu", values, "00004.jpg")
            self.assertIn("ort_npu", [r["backend"] for r in load_history(path, limit=2)])


if __name__ == "__main__":
    unittest.main()
