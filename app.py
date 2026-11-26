#!/usr/bin/env python3
"""HDMI/Wayland GTK3 interactive demo. Inference runs in one separate process."""

import argparse
import json
import queue
import signal
import time
from pathlib import Path

import cv2
import numpy as np

from history import append_history, load_history
from live_input import LiveInput, find_camera
from runtime import BACKENDS, fit_rect, image_point
from runtime_worker import RuntimeWorker
from scenes import default_image, image_source, model_revision
from scenes.depth_anything_v2_inputs import colorize
from scenes.grounding_dino_inputs import DEFAULT_PROMPT
from ui_style import category_color, layout_metrics, stylesheet


def requested_backends(action, selected):
    """Only an explicit Compare action schedules all four runtimes."""
    if action in ("benchmark", "video_compare"):
        return [backend for backend in BACKENDS if backend != selected] + [selected]
    return [selected]


def current_timing(result):
    """Cached SAM image features do not consume encoder time for a new click."""
    enc = result["encoder_ms"]
    if result.get("cached") and result.get("action") != "benchmark":
        enc = 0.0
    click = result.get("click_ms")
    return [
        f"{enc:.1f}",
        "—",
        "—" if click is None else f"{click:.1f}",
        f"{enc + (click or 0):.1f}",
        result["mode"],
    ]


def reset_measurement_table(cells):
    for row in cells.values():
        for cell in row[1:]:
            cell.set_text("—")


def input_mode_for_model(model, mode):
    """Only SAM's five-frame memory tracker exposes Video mode."""
    if mode not in ("image", "video", "camera"):
        raise ValueError(f"Unknown input mode: {mode}")
    return "image" if mode == "video" and model != "sam" else mode


def reset_tracking_view(view):
    """Return to a clean first-frame prompt without clearing timings or sessions."""
    view.video_frame = 0
    view.image_path = str(Path(__file__).parent / "images" / "bedroom" / "00000.jpg")
    view.points, view.mask = [], None
    view.load_image()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, default=Path(__file__).parent / "models")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--backend", choices=BACKENDS, default="ailia_npu")
    parser.add_argument(
        "--model", choices=("sam", "grounding", "depth", "yolox", "vit"), default="sam"
    )
    parser.add_argument("--caption", default=DEFAULT_PROMPT, help="Grounding DINO detection prompt")
    parser.add_argument(
        "--camera", type=int, help="Camera index (default: auto-detect USB UVC camera)"
    )
    parser.add_argument(
        "--video-input", type=Path, help="Diagnostic camera replay from a looping video file"
    )
    parser.add_argument(
        "--bedroom", action="store_true", help="Open the five-frame memory tracking demo"
    )
    parser.add_argument(
        "--autoplay-bedroom",
        action="store_true",
        help="Track the standard bedroom demo points once at startup",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="Save this application window after a completed action (diagnostics)",
    )
    args = parser.parse_args()
    if args.autoplay_bedroom:
        args.bedroom = True
    if args.model != "sam":
        args.bedroom = args.autoplay_bedroom = False
    if args.image is None:
        args.image = default_image(args.model)
    import gi

    gi.require_version("Gtk", "3.0")
    gi.require_version("Gdk", "3.0")
    from gi.repository import Gdk, GdkPixbuf, GLib, GLibUnix, Gtk

    class Demo(Gtk.Window):
        def __init__(self):
            super().__init__(title="ailia Vision Demo | i.MX95")
            self.set_default_size(1280, 720)
            self.connect("destroy", self.shutdown)
            self.points, self.mask, self.busy = [], None, False
            self.model = args.model
            self.detections = []
            self.depth = None
            self.results = {}
            self.closing = False
            self.capture = None
            self.camera_index = args.camera
            self.live_frame = None
            self.after_live = None
            self.input_mode = (
                "camera"
                if args.camera is not None or args.video_input
                else "video"
                if args.bedroom
                else "image"
            )
            self.syncing_mode = False
            self.live_started = None
            self.live_snapshot_scheduled = False
            self.video_mode = self.input_mode == "video"
            self.video_frame = 0
            self.video_records = []
            self.autoplay_pending = args.autoplay_bedroom
            if self.autoplay_pending:
                self.points = [[210 / 960, 350 / 540, 1], [250 / 960, 220 / 540, 1]]
            self.image_path = str(args.image.resolve())
            if args.bedroom:
                self.image_path = str(Path(__file__).parent / "images" / "bedroom" / "00000.jpg")
            self.backend = args.backend
            self.runner = RuntimeWorker(args.models)
            self.pending_jobs = []
            self.css = Gtk.CssProvider()
            self.layout_key = None
            self.resize_pending = False
            Gtk.StyleContext.add_provider_for_screen(
                Gdk.Screen.get_default(), self.css, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
            )
            root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
            root.set_border_width(12)
            self.add(root)
            heading = Gtk.Box(spacing=14)
            self.model_selector = Gtk.ComboBoxText()
            self.model_selector.append("sam", "SAM 2.1 Tiny")
            self.model_selector.append("grounding", "Grounding DINO")
            self.model_selector.append("depth", "Depth Anything V2")
            self.model_selector.append("yolox", "YOLOX-S 320")
            self.model_selector.append("vit", "ViT-B/16 INT4")
            self.model_selector.set_active_id(self.model)
            self.model_selector.connect("changed", self.switch_model)
            heading.pack_start(self.model_selector, False, False, 0)
            self.title_label = Gtk.Label(xalign=0)
            self.title_label.set_ellipsize(3)
            self.title_label.get_style_context().add_class("title")
            heading.pack_start(self.title_label, True, True, 0)
            quit_button = Gtk.Button(label="Exit")
            quit_button.connect("clicked", lambda *_: self.destroy())
            heading.pack_end(quit_button, False, False, 0)
            logo = Gtk.Image.new_from_pixbuf(
                GdkPixbuf.Pixbuf.new_from_file_at_scale(
                    str(Path(__file__).parent / "images" / "ailia.jpg"), 160, 160, True
                )
            )
            logo.set_tooltip_text("ailia")
            root.pack_start(heading, False, False, 0)
            source_bar = Gtk.Box(spacing=8)
            self.mode_buttons = {}
            for mode, label in (("image", "Image"), ("video", "Video"), ("camera", "Camera")):
                button = Gtk.ToggleButton(label=label)
                button.get_style_context().add_class("input-mode")
                button.connect("toggled", self.mode_toggled, mode)
                self.mode_buttons[mode] = button
                source_bar.pack_start(button, False, False, 0)
            self.source_label = Gtk.Label(xalign=1)
            self.source_label.set_ellipsize(3)
            source_bar.pack_start(self.source_label, True, True, 8)
            self.source_button = Gtk.Button(label="Open image…")
            self.source_button.connect("clicked", self.choose_source)
            source_bar.pack_end(self.source_button, False, False, 0)
            root.pack_start(source_bar, False, False, 0)
            body = Gtk.Box(spacing=18)
            self.body = body
            root.pack_start(body, True, True, 0)
            self.canvas = Gtk.DrawingArea()
            self.canvas.set_size_request(320, 180)
            self.canvas.add_events(Gdk.EventMask.BUTTON_PRESS_MASK)
            self.canvas.connect("draw", self.draw)
            self.canvas.connect("button-press-event", self.click)
            body.pack_start(self.canvas, True, True, 0)
            side = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
            self.side_scroll = Gtk.ScrolledWindow()
            self.side_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
            self.side_scroll.add(side)
            body.pack_start(self.side_scroll, False, False, 0)
            side.pack_start(Gtk.Label(label="Runtime", xalign=0), False, False, 0)
            self.selector = Gtk.ComboBoxText()
            for key in BACKENDS:
                runtime, device = key.split("_")
                self.selector.append(
                    key, f"{runtime if runtime == 'ailia' else 'ORT'} {device.upper()}"
                )
            self.selector.set_active_id(self.backend)
            self.selector.connect("changed", self.switch)
            side.pack_start(self.selector, False, False, 0)
            self.controls = [self.model_selector, self.selector]
            self.controls.append(self.source_button)
            self.prompt_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
            self.prompt_box.pack_start(
                Gtk.Label(label="Detection prompt (English)", xalign=0), False, False, 0
            )
            self.prompt_entry = Gtk.Entry()
            self.prompt_entry.set_text(args.caption)
            self.prompt_entry.connect("activate", lambda *_: self.submit("detect"))
            self.prompt_box.pack_start(self.prompt_entry, False, False, 0)
            detect = Gtk.Button(label="Detect")
            self.grounding_button = detect
            detect.connect("clicked", lambda *_: self.submit("detect"))
            self.prompt_box.pack_start(detect, False, False, 0)
            self.controls.extend([self.prompt_entry, detect])
            side.pack_start(self.prompt_box, False, False, 0)
            self.yolox_button = Gtk.Button(label="Detect")
            self.yolox_button.connect("clicked", lambda *_: self.submit("detect"))
            side.pack_start(self.yolox_button, False, False, 0)
            self.controls.append(self.yolox_button)
            self.vit_button = Gtk.Button(label="Classify")
            self.vit_button.connect("clicked", lambda *_: self.submit("classify"))
            side.pack_start(self.vit_button, False, False, 0)
            self.controls.append(self.vit_button)
            self.classification_label = Gtk.Label(xalign=0)
            self.classification_label.set_line_wrap(True)
            self.classification_label.set_max_width_chars(28)
            side.pack_start(self.classification_label, False, False, 0)
            self.depth_controls = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
            self.depth_button = Gtk.Button(label="Estimate depth")
            self.depth_button.connect("clicked", lambda *_: self.submit("estimate"))
            self.depth_controls.pack_start(self.depth_button, False, False, 0)
            self.show_depth = Gtk.CheckButton(label="Show depth map")
            self.show_depth.set_active(True)
            self.show_depth.connect("toggled", lambda *_: self.update_pixbuf())
            self.depth_controls.pack_start(self.show_depth, False, False, 0)
            self.depth_range = Gtk.ComboBoxText()
            for maximum in (5, 10, 20):
                self.depth_range.append(str(maximum), f"Display range: 0–{maximum} m")
            self.depth_range.set_active_id("5")
            self.depth_range.connect("changed", lambda *_: self.update_pixbuf())
            self.depth_controls.pack_start(self.depth_range, False, False, 0)
            side.pack_start(self.depth_controls, False, False, 0)
            self.controls.append(self.depth_button)
            self.sam_controls = []
            point_controls = Gtk.Box(spacing=6)
            side.pack_start(point_controls, False, False, 0)
            for title, callback in [
                ("Undo point", self.undo),
                ("Clear points", self.clear),
            ]:
                button = Gtk.Button(label=title)
                button.connect("clicked", callback)
                self.controls.append(button)
                self.sam_controls.append(button)
                point_controls.pack_start(button, True, True, 0)
            self.track_button = Gtk.Button(label="Track 5 frames")
            self.track_button.get_style_context().add_class("primary")
            self.track_button.connect("clicked", self.track)
            side.pack_start(self.track_button, False, False, 0)
            self.controls.append(self.track_button)
            info = Gtk.Label(
                label="Left: foreground (+)\nRight: background (−)\nBedroom: click frame 1, then Track.\nNPU: Encoder + MemoryAttention.\nOther tracking operations: CPU.\nINT8 integer-v1 · 6 threads · no profile\nORT NPU: timing only.\nOutputs may be incorrect or vary.",
                xalign=0,
            )
            info.set_line_wrap(True)
            self.info = info
            self.info.set_max_width_chars(30)
            side.pack_start(info, False, False, 0)
            self.status = Gtk.Label(label="Starting…", xalign=0)
            self.status.set_ellipsize(3)
            self.status.set_max_width_chars(30)
            side.pack_start(self.status, False, False, 0)
            self.live_timing = Gtk.Label(xalign=0)
            self.live_timing.set_line_wrap(True)
            self.live_timing.set_max_width_chars(30)
            side.pack_start(self.live_timing, False, False, 0)
            grid = Gtk.Grid(column_spacing=24, row_spacing=6)
            grid.set_hexpand(True)
            self.table_scroll = Gtk.ScrolledWindow()
            self.table_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.NEVER)
            self.table_scroll.add(grid)
            root.pack_start(self.table_scroll, False, False, 0)
            self.compare_button = Gtk.Button(label="Compare all 4")
            self.compare_button.set_valign(Gtk.Align.CENTER)
            self.compare_button.connect("clicked", self.benchmark)
            heading.pack_end(self.compare_button, False, False, 0)
            self.controls.append(self.compare_button)
            self.table_headers = []
            for col, text in enumerate(
                [
                    "Backend",
                    "Encoder (ms)",
                    "Memory (ms)",
                    "Click / other (ms)",
                    "Sum (ms)",
                    "Measurement",
                ]
            ):
                label = Gtk.Label(label=text, xalign=0)
                self.table_headers.append(label)
                grid.attach(label, 0, col, 1, 1)
            self.cells = {}
            for row, backend in enumerate(BACKENDS, 1):
                cells = [
                    Gtk.Label(label=t, xalign=0)
                    for t in [backend.replace("_", " "), "—", "—", "—", "—", "not measured"]
                ]
                for col, cell in enumerate(cells):
                    cell.set_hexpand(True)
                    cell.set_ellipsize(3)
                    grid.attach(cell, row, col, 1, 1)
                self.cells[backend] = cells
            self.history_path = Path(__file__).parent / "benchmark_history.jsonl"
            self.history_rows = Gtk.ListStore(*([str] * 7))
            view = Gtk.TreeView(model=self.history_rows)
            for col, name in enumerate(
                [
                    "Time / input",
                    "Backend",
                    "Encoder / inference",
                    "Memory",
                    "Click / other",
                    "Sum",
                    "Measurement",
                ]
            ):
                view.append_column(Gtk.TreeViewColumn(name, Gtk.CellRendererText(), text=col))
            scroll = Gtk.ScrolledWindow()
            scroll.set_size_request(-1, 140)
            scroll.add(view)
            history_box = Gtk.Expander(label="Measurement history")
            history_box.set_tooltip_text("Last 200 measurements, including previous runs")
            history_box.add(scroll)
            footer = Gtk.Box(spacing=18)
            footer_text = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
            footer_text.set_valign(Gtk.Align.END)
            footer.pack_start(footer_text, True, True, 0)
            logo.set_valign(Gtk.Align.END)
            footer.pack_end(logo, False, False, 0)
            root.pack_end(footer, False, False, 0)
            footer_text.pack_start(history_box, False, False, 0)
            try:
                for record in load_history(self.history_path):
                    self.show_measurement(record, update_table=False)
            except OSError as exc:
                print("Cannot load measurement history:", exc, flush=True)
            self.attribution = Gtk.Label(xalign=0)
            self.attribution.get_style_context().add_class("muted")
            self.attribution.set_ellipsize(3)
            footer_text.pack_start(self.attribution, False, False, 0)
            self.load_image()
            # Hide mode-specific controls before mapping the window; otherwise
            # GTK sizes a 720p window for every model's controls at once.
            root.show_all()
            self.update_model_controls()
            self.connect("size-allocate", self.resize_layout)
            self.resize_layout(self, self.get_allocation())
            self.show()
            if not args.windowed:
                self.fullscreen()
            GLib.timeout_add(30, self.poll)
            GLib.timeout_add(30, self.poll_capture)
            if self.input_mode == "camera":
                self.start_camera()
            else:
                self.submit("prepare")

        def resize_layout(self, _widget, allocation):
            # GTK cannot reliably reflow containers during their allocation signal.
            if not self.resize_pending:
                self.resize_pending = True
                GLib.idle_add(self.apply_layout)

        def apply_layout(self):
            self.resize_pending = False
            allocation = self.get_allocation()
            width, height = max(640, allocation.width), max(480, allocation.height)
            metrics = layout_metrics(width, height)
            self.table_scroll.set_min_content_height(
                (metrics["font"] + 12) * (5 if self.model == "sam" else 2)
            )
            key = tuple(metrics.values())
            if key == self.layout_key:
                return
            self.layout_key = key
            self.css.load_from_data(stylesheet(metrics["font"], metrics["touch"]))
            portrait = metrics["portrait"]
            self.body.set_orientation(
                Gtk.Orientation.VERTICAL if portrait else Gtk.Orientation.HORIZONTAL
            )
            self.side_scroll.set_size_request(
                -1 if portrait else metrics["side"],
                metrics["touch"] * 3 if portrait else -1,
            )
            self.title_label.set_visible(not portrait)

        def stop_capture(self):
            if self.capture is not None:
                self.capture.close()
                self.capture = None

        def mode_toggled(self, button, mode):
            if self.syncing_mode:
                return
            if button.get_active():
                self.select_input_mode(mode)
            else:
                self.update_model_controls()

        def select_input_mode(self, mode):
            mode = input_mode_for_model(self.model, mode)
            self.after_live = None
            self.stop_capture()
            if self.busy:
                self.runner.close()
            self.pending_jobs = []
            self.busy = False
            for control in self.controls:
                control.set_sensitive(True)
            self.input_mode = mode
            reset_measurement_table(self.cells)
            self.live_frame = None
            self.live_timing.set_text("")
            self.points, self.mask, self.detections, self.depth = [], None, [], None
            self.video_mode, self.video_frame = mode == "video", 0
            self.image_path = str(
                Path(__file__).parent / "images/bedroom/00000.jpg"
                if self.video_mode
                else default_image(self.model)
            )
            self.update_model_controls()
            self.load_image()
            if mode == "camera":
                self.start_camera()
            else:
                self.submit("prepare")

        def choose_source(self, *_):
            if self.input_mode == "image":
                self.open_image()
            elif self.input_mode == "video":
                self.select_input_mode("video")
            else:
                self.select_input_mode("camera")

        def start_camera(self):
            # Camera mode is continuously inferred, without a Start/Detect step.
            self.live_snapshot_scheduled = False
            self.rgb = np.zeros((480, 640, 3), np.uint8)
            self.update_pixbuf()
            try:
                source = str(args.video_input) if args.video_input else find_camera(args.camera)
                self.capture = LiveInput(source)
            except (ValueError, OSError) as exc:
                self.status.set_text(str(exc))
                self.source_label.set_text("Camera not connected")
                return
            if isinstance(source, int):
                self.camera_index = source
            self.source_label.set_text(
                "Camera replay: " + Path(source).name
                if isinstance(source, str)
                else f"USB camera {source}"
            )
            self.attribution.set_text("Input: " + self.source_label.get_text())
            self.attribution.set_tooltip_text(None)
            self.status.set_text("Opening camera…")

        def poll_capture(self):
            if self.closing:
                return False
            if self.capture is None or self.busy:
                return True
            item = self.capture.read()
            if item is None:
                return True
            kind, value = item
            if kind == "error":
                self.stop_capture()
                self.status.set_text(value)
                self.live_timing.set_text("Disconnected — press Reconnect.")
                return True
            self.live_frame = value
            self.live_started = time.monotonic()
            self.submit("live")
            return True

        def update_measurement_table(self, backend, values):
            for cell, text in zip(self.cells[backend][1:], values):
                cell.set_text(text)

        def show_measurement(self, record, update_table=True):
            backend = record["backend"]
            values = list(record["values"])
            values[-1] += " / " + record.get("model_revision", "legacy-qdq")
            if update_table and record.get("model", "sam") == self.model:
                self.update_measurement_table(backend, values)
            label = (
                record["time"]
                + " / "
                + record.get("model", "sam")
                + " / "
                + Path(record.get("image", "")).name
            )
            if record.get("caption"):
                label += " / " + record["caption"]
            self.history_rows.append([label, backend, *values])
            if len(self.history_rows) > 200:
                self.history_rows.remove(self.history_rows.get_iter_first())

        def input_name(self):
            if self.capture is not None:
                return (
                    str(args.video_input) if args.video_input else f"USB camera {self.camera_index}"
                )
            return self.image_path

        def record_measurement(self, backend, values, image=None):
            image = self.input_name() if image is None else image
            try:
                record = append_history(
                    self.history_path,
                    backend,
                    values,
                    image,
                    model=self.model,
                    caption=self.prompt_entry.get_text() if self.model == "grounding" else None,
                )
            except OSError as exc:
                print("Cannot save measurement history:", exc, flush=True)
                record = dict(
                    time="unsaved",
                    backend=backend,
                    values=values,
                    image=image,
                    model_revision=model_revision(self.model),
                    model=self.model,
                )
            self.show_measurement(record)

        def load_image(self):
            self.classification_label.set_text("")
            if self.capture is not None:
                self.update_pixbuf()
                return
            source = image_source(self.image_path)
            self.attribution.set_text(
                "Image source: " + source if source else "Image: user-selected file"
            )
            self.attribution.set_tooltip_text(source)
            self.source_label.set_text(
                "Bedroom · frame 1 / 5" if self.video_mode else Path(self.image_path).name
            )
            bgr = cv2.imread(self.image_path)
            if bgr is None:
                raise ValueError("Cannot read " + self.image_path)
            self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.update_pixbuf()

        def update_pixbuf(self):
            rgb = self.rgb.copy()
            if self.model == "depth" and self.depth is not None and self.show_depth.get_active():
                rgb = colorize(
                    self.depth,
                    rgb.shape[1],
                    rgb.shape[0],
                    maximum=float(self.depth_range.get_active_id()),
                )
            if self.model in ("grounding", "yolox"):
                height, width = rgb.shape[:2]
                for item in self.detections:
                    x0, y0, x1, y1 = (
                        np.array(item["box"]) * [width, height, width, height]
                    ).astype(int)
                    color = (
                        category_color(item["label"]) if self.model == "yolox" else (50, 225, 170)
                    )
                    thickness = max(2, round(min(height, width) / 240))
                    scale = max(0.4, min(height, width) / 950)
                    text = f"{item['label']} {item['score']:.2f}"
                    (tw, th), baseline = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
                    )
                    tx = max(0, min(x0, width - tw - 8))
                    ty = max(th + 8, min(y0 - 4, height - baseline - 4))
                    cv2.rectangle(rgb, (x0, y0), (x1, y1), color, thickness)
                    cv2.rectangle(
                        rgb,
                        (tx, ty - th - 6),
                        (min(width - 1, tx + tw + 8), ty + baseline + 3),
                        color,
                        -1,
                    )
                    cv2.putText(
                        rgb,
                        text,
                        (tx + 4, ty),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        scale,
                        (255, 255, 255) if self.model == "yolox" else (16, 24, 32),
                        thickness,
                        cv2.LINE_AA,
                    )
            if self.mask is not None:
                selected = cv2.resize(
                    self.mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                rgb[selected] = (rgb[selected] * 0.52 + np.array([50, 225, 170]) * 0.48).astype(
                    np.uint8
                )
            h, w = rgb.shape[:2]
            self.pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(
                GLib.Bytes.new(rgb.tobytes()), GdkPixbuf.Colorspace.RGB, False, 8, w, h, w * 3
            )
            self.canvas.queue_draw()

        def draw(self, area, cr):
            cr.set_source_rgb(0.055, 0.085, 0.11)
            cr.paint()
            h, w = self.rgb.shape[:2]
            self.rect = fit_rect(area.get_allocated_width(), area.get_allocated_height(), w, h)
            x, y, width, height = self.rect
            cr.save()
            cr.translate(x, y)
            cr.scale(width / w, height / h)
            Gdk.cairo_set_source_pixbuf(cr, self.pixbuf, 0, 0)
            cr.paint()
            cr.restore()
            for px, py, label in [] if self.video_mode and self.video_frame else self.points:
                cr.arc(x + px * width, y + py * height, 6, 0, np.pi * 2)
                cr.set_source_rgb(*((0.2, 1, 0.65) if label else (1, 0.3, 0.3)))
                cr.fill_preserve()
                cr.set_source_rgb(1, 1, 1)
                cr.set_line_width(2)
                cr.stroke()

        def click(self, _area, event):
            if self.model == "depth" and self.depth is not None and not self.busy:
                point = image_point(event.x, event.y, self.rect)
                if point is not None:
                    height, width = self.depth.shape
                    x = min(width - 1, int(point[0] * width))
                    y = min(height - 1, int(point[1] * height))
                    self.status.set_text(
                        f"Estimated depth at clicked pixel: {self.depth[y, x]:.2f} m"
                    )
                return
            if (
                self.model != "sam"
                or (self.busy and self.capture is None)
                or event.button not in (1, 3)
            ):
                return
            if self.video_mode and self.video_frame != 0:
                self.status.set_text("Press Bedroom to return to frame 1 before adding points.")
                return
            point = image_point(event.x, event.y, self.rect)
            if point is not None and len(self.points) < 64:
                self.points.append([*point, int(event.button == 1)])
                self.canvas.queue_draw()
                self.submit("segment")

        def submit(self, action):
            if self.busy:
                return
            if self.input_mode == "camera" and action != "live" and self.live_frame is None:
                self.status.set_text("Waiting for a camera frame…")
                return
            self.busy = True
            for control in self.controls:
                control.set_sensitive(action == "live")
            self.status.set_text("Working… " + self.backend)
            backends = requested_backends(action, self.backend)
            self.pending_jobs = [
                dict(
                    action=action,
                    backend=backend,
                    image=self.live_frame.copy()
                    if self.capture is not None and self.live_frame is not None
                    else self.image_path,
                    points=list(self.points),
                    video_mode=self.video_mode,
                    model=self.model,
                    caption=self.prompt_entry.get_text(),
                )
                for backend in backends
            ]
            self.runner.submit(self.pending_jobs.pop(0))

        def update_model_controls(self):
            grounding = self.model == "grounding"
            sam = self.model == "sam"
            camera = self.input_mode == "camera"
            self.syncing_mode = True
            for mode, button in self.mode_buttons.items():
                button.set_active(mode == self.input_mode)
                button.set_sensitive(mode != "video" or sam)
            self.syncing_mode = False
            self.mode_buttons["video"].set_tooltip_text(
                "SAM 2 five-frame memory tracking" if sam else "Video is available for SAM 2 only"
            )
            self.source_button.set_label(
                "Reconnect" if camera else "Reset to frame 1" if self.video_mode else "Open image…"
            )
            self.track_button.set_visible(sam and self.video_mode)
            self.compare_button.set_label("Compare current frame" if camera else "Compare all 4")
            self.title_label.set_text(
                "ViT-B/16 INT4 / 224 × 224 / i.MX95"
                if self.model == "vit"
                else "YOLOX-S / 320 × 320 / i.MX95"
                if self.model == "yolox"
                else "Depth Anything V2 Metric Small / 518 × 518 / i.MX95"
                if self.model == "depth"
                else "Grounding DINO / 400 × 599 / i.MX95"
                if grounding
                else "SAM 2.1 Tiny / 512 × 512 / i.MX95"
            )
            self.prompt_box.set_visible(grounding)
            self.grounding_button.set_visible(grounding and not camera)
            self.yolox_button.set_visible(self.model == "yolox" and not camera)
            self.vit_button.set_visible(self.model == "vit" and not camera)
            self.classification_label.set_visible(self.model == "vit")
            self.depth_controls.set_visible(self.model == "depth")
            self.depth_button.set_visible(self.model == "depth" and not camera)
            for control in self.sam_controls:
                control.set_visible(sam)
            self.table_headers[1].set_text("Encoder (ms)" if sam else "Inference (ms)")
            for column in (2, 3, 4):
                self.table_headers[column].set_visible(sam)
                for cells in self.cells.values():
                    cells[column].set_visible(sam)
            self.table_headers[5].hide()
            for cells in self.cells.values():
                cells[5].hide()
            self.info.set_text(
                "ImageNet · top 5 classes"
                if self.model == "vit"
                else "Continuous detection on each camera frame."
                if camera and self.model in ("yolox", "grounding")
                else "Click a target; subsequent camera frames reuse those points."
                if camera and sam
                else "Continuous depth estimation."
                if camera
                else "YOLOX-S · 320 × 320\nCOCO · 80 classes"
                if self.model == "yolox"
                else "Indoor metric depth.\nClick to inspect distance."
                if self.model == "depth"
                else "Separate category names with periods."
                if grounding
                else "Click the target on frame 1, then Track 5 frames."
                if self.video_mode
                else "Left click: foreground (+)\nRight click: background (−)"
            )
            self.resize_layout(self, self.get_allocation())

        def switch_model(self, selector):
            self.after_live = None
            self.model = selector.get_active_id()
            self.input_mode = input_mode_for_model(self.model, self.input_mode)
            self.live_timing.set_text("")
            self.runner.close()
            self.busy = False
            self.pending_jobs = []
            self.video_mode, self.video_frame = False, 0
            self.points, self.mask, self.detections = [], None, []
            self.depth = None
            self.image_path = str(default_image(self.model))
            reset_measurement_table(self.cells)
            self.update_model_controls()
            self.load_image()
            if self.input_mode != "camera":
                self.submit("prepare")
            elif self.capture is None:
                self.start_camera()

        def switch(self, selector):
            self.after_live = None
            self.live_timing.set_text("")
            if self.capture is not None:
                self.runner.close()
                self.busy = False
                self.pending_jobs = []
            self.backend = selector.get_active_id()
            reset_measurement_table(self.cells)
            self.depth = None
            if self.video_mode:
                self.image_path = str(Path(__file__).parent / "images" / "bedroom" / "00000.jpg")
                self.video_frame = 0
            self.mask = None
            self.detections = []
            self.load_image()
            if self.input_mode != "camera":
                self.submit("prepare")

        def benchmark(self, *_):
            if self.capture is not None and self.busy:
                self.after_live = "benchmark"
                self.status.set_text("Compare all queued after the current frame.")
                return
            if self.video_mode:
                self.track(compare=True)
            else:
                self.submit("benchmark")

        def bedroom(self, *_):
            self.select_input_mode("video")

        def track(self, *_, compare=False):
            if not self.video_mode or not self.points:
                self.status.set_text(
                    "Press Bedroom, click the target on frame 1, then Track 5 frames."
                )
                return
            self.video_records = []
            self.submit("video_compare" if compare else "video")

        def undo(self, *_):
            if self.points:
                self.points.pop()
            self.mask = None
            self.update_pixbuf()
            if self.points:
                self.submit("segment")

        def clear(self, *_):
            self.points, self.mask = [], None
            self.update_pixbuf()

        def open_image(self, *_):
            dialog = Gtk.FileChooserDialog(
                title="Select image", parent=self, action=Gtk.FileChooserAction.OPEN
            )
            dialog.add_buttons("Cancel", Gtk.ResponseType.CANCEL, "Open", Gtk.ResponseType.OK)
            filt = Gtk.FileFilter()
            filt.set_name("Images")
            for mime in ["image/jpeg", "image/png"]:
                filt.add_mime_type(mime)
            dialog.add_filter(filt)
            if dialog.run() == Gtk.ResponseType.OK:
                self.stop_capture()
                self.input_mode = "image"
                self.video_mode = False
                self.image_path = dialog.get_filename()
                self.points, self.mask = [], None
                self.detections = []
                self.depth = None
                self.update_model_controls()
                self.load_image()
                self.submit("prepare")
            dialog.destroy()

        def poll(self):
            if self.closing:
                return False
            while True:
                if self.runner.responses is None:
                    break
                try:
                    result = self.runner.responses.get_nowait()
                except queue.Empty:
                    break
                kind = result["kind"]
                if kind == "progress":
                    self.status.set_text(
                        "Camera · continuous inference"
                        if self.input_mode == "camera"
                        and self.live_timing.get_text()
                        and not self.pending_jobs
                        else result["text"]
                    )
                elif kind == "video_frame":
                    backend = result["backend"]
                    times = result["times_ms"]
                    enc = times["image_encoder"]
                    memory = times["memory_attention"]
                    other = sum(times.values()) - enc - memory
                    self.record_measurement(
                        backend,
                        [
                            f"{enc:.1f}",
                            f"{memory:.1f}",
                            f"{other:.1f}",
                            f"{enc + memory + other:.1f}",
                            f"video frame {result['frame'] + 1}/5",
                        ],
                        image=result["image"],
                    )
                    if backend == self.backend:
                        self.image_path = result["image"]
                        self.video_frame = result["frame"]
                        self.mask = result["mask"]
                        self.load_image()
                        self.source_label.set_text(f"Bedroom · frame {self.video_frame + 1} / 5")
                    self.status.set_text(
                        f"{backend} frame {result['frame'] + 1}/5 | MemoryAttention {times['memory_attention']:.1f} ms | processing {result['total_ms']:.1f} ms"
                    )
                    self.live_timing.set_text(
                        f"Inference: {sum(times.values()):.1f} ms\n"
                        f"Frame processing: {result['total_ms']:.1f} ms"
                    )
                    self.video_records.append({k: v for k, v in result.items() if k != "mask"})
                    try:
                        (Path(__file__).parent / "video_results.json").write_text(
                            json.dumps(self.video_records, indent=2)
                        )
                    except OSError as exc:
                        print("Cannot save video timings:", exc, flush=True)
                elif kind == "result":
                    backend = result["backend"]
                    if (
                        self.capture is not None
                        and backend == self.backend
                        and self.live_frame is not None
                    ):
                        # Display the exact inferred frame, never a newer camera frame
                        # underneath stale boxes, masks, or depth values.
                        self.rgb = cv2.cvtColor(self.live_frame, cv2.COLOR_BGR2RGB)
                        self.mask, self.detections, self.depth = None, [], None
                        self.update_pixbuf()
                    values = current_timing(result)
                    self.update_measurement_table(backend, values)
                    if result.get("action") == "live":
                        elapsed = time.monotonic() - self.live_started if self.live_started else 0
                        self.live_timing.set_text(
                            f"Inference: {values[3]} ms\nProcessing: {1 / elapsed if elapsed else 0:.2f} FPS"
                        )
                        self.status.set_text("Camera · continuous inference")
                    else:
                        self.live_timing.set_text(f"Inference: {values[3]} ms")
                        if result.get("action") != "prepare":
                            self.record_measurement(backend, values)
                    if backend == self.backend and "mask" in result and self.points:
                        self.mask = result["mask"]
                        self.update_pixbuf()
                    if backend == self.backend and "detections" in result:
                        self.detections = result["detections"]
                        self.update_pixbuf()
                    if backend == self.backend and "depth" in result:
                        self.depth = result["depth"]
                        self.update_pixbuf()
                    if backend == self.backend and "classifications" in result:
                        self.classification_label.set_text(
                            "\n".join(
                                f"{item['score']:.1%}  {item['label'].split(',')[0]}"
                                for item in result["classifications"]
                            )
                        )
                    if result.get("action") == "live":
                        continue
                    record = {k: v for k, v in result.items() if k not in ("mask", "depth")}
                    record.update(
                        image=self.input_name(), points=self.points, threads=6, profiling=False
                    )
                    self.results[self.model + ":" + backend] = record
                    try:
                        (Path(__file__).parent / "last_results.json").write_text(
                            json.dumps(self.results, indent=2)
                        )
                    except OSError as exc:
                        print("Cannot save timings:", exc, flush=True)
                elif kind in ("done", "error"):
                    if kind == "error" and self.model == "vit":
                        self.classification_label.set_text("")
                    if kind == "error" and self.model == "depth":
                        self.depth = None
                        self.update_pixbuf()
                    if kind == "error" and self.model in ("grounding", "yolox"):
                        self.detections = []
                        self.update_pixbuf()
                    if kind == "done" and self.pending_jobs:
                        self.runner.submit(self.pending_jobs.pop(0))
                        continue
                    self.pending_jobs = []
                    self.busy = False
                    for control in self.controls:
                        control.set_sensitive(True)
                    if self.input_mode != "camera":
                        self.update_model_controls()
                    if self.capture is None or kind == "error":
                        self.status.set_text(result.get("text", "Ready — click the image."))
                    if kind == "error" and self.capture is not None:
                        self.stop_capture()
                        self.status.set_text(result["text"])
                        self.live_timing.set_text("Live input stopped: " + result["text"])
                    if self.after_live and kind == "done":
                        action, self.after_live = self.after_live, None
                        self.submit(action)
                    if self.video_mode and result.get("action") in ("video", "video_compare"):
                        reset_tracking_view(self)
                        if kind == "done":
                            self.status.set_text(
                                "Tracking complete — click a new target on frame 1."
                            )
                    if self.autoplay_pending:
                        self.autoplay_pending = False
                        if kind == "done":
                            GLib.idle_add(self.track)
                    if args.snapshot and (self.capture is None or not self.live_snapshot_scheduled):
                        self.live_snapshot_scheduled = self.capture is not None
                        GLib.timeout_add(300, self.snapshot)
            if self.runner.process is not None and not self.runner.process.is_alive():
                self.stop_capture()
                self.runner.close()
                self.busy = False
                self.pending_jobs = []
                for control in self.controls:
                    control.set_sensitive(True)
                self.status.set_text(
                    "Inference stopped. Reconnect or select an input to retry. See demo.log."
                )
            return True

        def snapshot(self):
            import cairo

            self.check_resize()
            surface = cairo.ImageSurface(
                cairo.FORMAT_ARGB32, self.get_allocated_width(), self.get_allocated_height()
            )
            Gtk.Widget.draw(self, cairo.Context(surface))
            destination = Path(args.snapshot)
            temporary = destination.with_name("." + destination.name + ".tmp")
            surface.write_to_png(str(temporary))
            temporary.replace(destination)
            return False

        def shutdown(self, *_):
            self.closing = True
            if self.capture is not None:
                self.capture.close()
            self.runner.close()
            Gtk.main_quit()

    demo = Demo()

    def stop_signal():
        demo.destroy()
        return False

    GLibUnix.signal_add(GLib.PRIORITY_DEFAULT, signal.SIGTERM, stop_signal)
    GLibUnix.signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT, stop_signal)
    Gtk.main()


if __name__ == "__main__":
    main()
