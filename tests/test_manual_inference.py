"""Exercise the GTK model-switch handler without GTK or native runtimes."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

import app


class ManualInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(Path(app.__file__).read_text())
        namespace = dict(vars(app))
        for name in (
            "switch_model", "run_inference", "select_prompt", "select_input_mode", "poll_capture",
            "toggle_camera_inference",
            "set_status_if_changed", "show_progress",
        ):
            method = next(
                node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == name
            )
            exec(compile(ast.Module(body=[method], type_ignores=[]), app.__file__, "exec"), namespace)
            setattr(cls, name, staticmethod(namespace[name]))

    def view(self, mode):
        return SimpleNamespace(
            input_mode=mode,
            live_timing=Mock(),
            runner=Mock(),
            cells={},
            update_model_controls=Mock(),
            load_image=Mock(),
            show_ready=Mock(),
            submit=Mock(),
            capture=None,
            start_camera=Mock(),
            busy=False,
            stop_capture=Mock(),
            controls=[],
            camera_inference=False,
            status=Mock(),
            camera_run_button=Mock(),
            backend="ailia_npu",
        )

    def test_model_switch_waits_for_explicit_action(self):
        for model in ("sam", "vit", "depth", "grounding", "yolox"):
            with self.subTest(model=model):
                view = self.view("image")
                self.switch_model(view, Mock(get_active_id=Mock(return_value=model)))
                view.submit.assert_not_called()
                view.show_ready.assert_called_once_with()
                view.load_image.assert_called_once_with()
                self.assertFalse(view.busy)

    def test_camera_switch_waits_when_not_started(self):
        view = self.view("camera")
        self.switch_model(view, Mock(get_active_id=Mock(return_value="yolox")))
        view.start_camera.assert_not_called()
        view.show_ready.assert_called_once_with()

    def test_select_camera_waits(self):
        view = self.view("image")
        view.model = "yolox"
        self.select_input_mode(view, "camera")
        view.start_camera.assert_called_once_with()
        view.submit.assert_not_called()
        self.assertFalse(view.camera_inference)

    def test_execution_starts_camera_once(self):
        for action in ("live", "detect", "classify", "estimate"):
            view = self.view("camera")
            self.run_inference(view, action)
            view.start_camera.assert_called_once_with()
            view.submit.assert_not_called()
            view.capture = Mock()
            self.run_inference(view, action)
            view.start_camera.assert_called_once_with()
            self.assertTrue(view.camera_inference)

    def test_preview_does_not_infer_until_run(self):
        view = self.view("camera")
        view.closing = False
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        view.capture = Mock()
        view.capture.read.return_value = ("frame", frame)
        view.update_pixbuf = Mock()
        self.assertTrue(self.poll_capture(view))
        view.update_pixbuf.assert_called_once_with()
        view.submit.assert_not_called()
        self.run_inference(view, "detect")
        self.assertTrue(self.poll_capture(view))
        view.submit.assert_called_once_with("live")

    def test_image_execution_submits(self):
        view = self.view("image")
        self.run_inference(view, "detect")
        view.submit.assert_called_once_with("detect")
        view.start_camera.assert_not_called()

    def test_stop_keeps_preview_and_finishes_current_frame(self):
        for busy in (False, True):
            view = self.view("camera")
            view.camera_inference = True
            view.busy = busy
            view.capture = Mock()
            self.toggle_camera_inference(view)
            self.assertFalse(view.camera_inference)
            self.assertEqual(view.busy, busy)
            view.capture.close.assert_not_called()
            view.runner.close.assert_not_called()
            view.camera_run_button.set_label.assert_called_once_with("Run")

    def test_run_button_starts_inference(self):
        view = self.view("camera")
        view.run_inference = Mock()
        self.toggle_camera_inference(view)
        view.run_inference.assert_called_once_with("live")

    def test_live_progress_does_not_replace_status(self):
        view = self.view("camera")
        view.current_action = "live"
        view.set_status_if_changed = Mock()
        self.show_progress(view, "Loading / running YOLOX / ailia_npu")
        view.set_status_if_changed.assert_not_called()
        view.current_action = "benchmark"
        self.show_progress(view, "Loading / running YOLOX / ort_cpu")
        view.set_status_if_changed.assert_called_once()

    def test_unchanged_status_is_not_redrawn(self):
        view = self.view("camera")
        view.status.get_text.return_value = "Camera · continuous inference"
        self.set_status_if_changed(view, "Camera · continuous inference")
        view.status.set_text.assert_not_called()
        self.set_status_if_changed(view, "Loading… ailia_npu")
        view.status.set_text.assert_called_once_with("Loading… ailia_npu")

    def test_prompt_preset_only_updates_text(self):
        view = self.view("image")
        view.prompt_entry = Mock()
        self.select_prompt(view, Mock(get_active_id=Mock(return_value="person. chair.")))
        view.prompt_entry.set_text.assert_called_once_with("person. chair.")
        view.submit.assert_not_called()
        self.select_prompt(view, Mock(get_active_id=Mock(return_value="")))
        self.assertEqual(view.prompt_entry.set_text.call_count, 1)
