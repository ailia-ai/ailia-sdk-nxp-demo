"""Exercise the GTK model-switch handler without GTK or native runtimes."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import app


class ManualInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(Path(app.__file__).read_text())
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "switch_model"
        )
        namespace = dict(vars(app))
        exec(compile(ast.Module(body=[method], type_ignores=[]), app.__file__, "exec"), namespace)
        cls.switch_model = staticmethod(namespace["switch_model"])

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

    def test_camera_switch_keeps_continuous_inference(self):
        view = self.view("camera")
        self.switch_model(view, Mock(get_active_id=Mock(return_value="yolox")))
        view.start_camera.assert_called_once_with()
        view.show_ready.assert_not_called()
