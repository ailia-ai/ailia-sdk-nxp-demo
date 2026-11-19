#!/usr/bin/env python3
"""HDMI/Wayland GTK3 interactive demo. Inference runs in one separate process."""

import argparse
import json
import queue
import signal
from pathlib import Path

import cv2
import numpy as np

from depth_inputs import REVISION as DEPTH_REVISION
from depth_inputs import colorize
from engine import BACKENDS, fit_rect, image_point
from grounding_inputs import DEFAULT_PROMPT
from grounding_inputs import REVISION as GROUNDING_REVISION
from history import append_history, load_history
from runtime_worker import RuntimeWorker
from settings import MODEL_REVISION


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
    parser.add_argument("--model", choices=("sam", "grounding", "depth"), default="sam")
    parser.add_argument("--caption", default=DEFAULT_PROMPT, help="Grounding DINO detection prompt")
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
        args.image = (
            Path(__file__).parent
            / "images"
            / ("bedroom/00000.jpg" if args.model == "depth" else "truck.jpg")
        )
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
            self.video_mode = args.bedroom
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
            css = Gtk.CssProvider()
            css.load_from_data(b"""
                window { background: #101820; color: #ecf3f8; }
                button, combobox { font-size: 14px; padding: 5px; }
                button { background-image: none; background-color: #e8eef4; color: #14202b; }
                button label, combobox cellview { color: #14202b; }
                button:disabled { background-color: #8795a3; }
                menu, menuitem, menuitem label, menuitem cellview,
                popover, list, row { background-color: #e8eef4; color: #14202b; }
                menuitem:hover, menuitem:hover label, menuitem:hover cellview,
                row:selected { background-color: #c0d9ee; color: #14202b; }
                label { font-size: 14px; }
                .title { font-size: 23px; font-weight: bold; }
            """)
            Gtk.StyleContext.add_provider_for_screen(
                Gdk.Screen.get_default(), css, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
            )
            root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
            root.set_border_width(12)
            self.add(root)
            heading = Gtk.Box(spacing=14)
            self.model_selector = Gtk.ComboBoxText()
            self.model_selector.append("sam", "SAM 2.1 Tiny")
            self.model_selector.append("grounding", "Grounding DINO")
            self.model_selector.append("depth", "Depth Anything V2")
            self.model_selector.set_active_id(self.model)
            self.model_selector.connect("changed", self.switch_model)
            heading.pack_start(self.model_selector, False, False, 0)
            self.title_label = Gtk.Label(xalign=0)
            self.title_label.get_style_context().add_class("title")
            heading.pack_start(self.title_label, True, True, 0)
            logo = Gtk.Image.new_from_pixbuf(
                GdkPixbuf.Pixbuf.new_from_file_at_scale(
                    str(Path(__file__).parent / "images" / "ailia.jpg"), 56, 56, True
                )
            )
            logo.set_tooltip_text("ailia")
            heading.pack_end(logo, False, False, 0)
            root.pack_start(heading, False, False, 0)
            body = Gtk.Box(spacing=18)
            root.pack_start(body, True, True, 0)
            self.canvas = Gtk.DrawingArea()
            self.canvas.set_size_request(400, 280)
            self.canvas.add_events(Gdk.EventMask.BUTTON_PRESS_MASK)
            self.canvas.connect("draw", self.draw)
            self.canvas.connect("button-press-event", self.click)
            body.pack_start(self.canvas, True, True, 0)
            side = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
            side.set_size_request(250, -1)
            body.pack_start(side, False, False, 0)
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
            self.prompt_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
            self.prompt_box.pack_start(
                Gtk.Label(label="Detection prompt (English)", xalign=0), False, False, 0
            )
            self.prompt_entry = Gtk.Entry()
            self.prompt_entry.set_text(args.caption)
            self.prompt_entry.connect("activate", lambda *_: self.submit("detect"))
            self.prompt_box.pack_start(self.prompt_entry, False, False, 0)
            detect = Gtk.Button(label="Detect")
            detect.connect("clicked", lambda *_: self.submit("detect"))
            self.prompt_box.pack_start(detect, False, False, 0)
            self.controls.extend([self.prompt_entry, detect])
            side.pack_start(self.prompt_box, False, False, 0)
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
            for title, callback in [
                (f"Compare all {len(BACKENDS)}", self.benchmark),
                ("Undo point", self.undo),
                ("Clear points", self.clear),
                ("Open image…", self.open_image),
                ("Bedroom (5 frames)", self.bedroom),
                ("Track 5 frames", self.track),
            ]:
                button = Gtk.Button(label=title)
                button.connect("clicked", callback)
                self.controls.append(button)
                if title not in (f"Compare all {len(BACKENDS)}", "Open image…"):
                    self.sam_controls.append(button)
                side.pack_start(button, False, False, 0)
            info = Gtk.Label(
                label="Left: foreground (+)\nRight: background (−)\nBedroom: click frame 1, then Track.\nNPU: Encoder + MemoryAttention.\nOther tracking operations: CPU.\nINT8 integer-v1 · 6 threads · no profile\nORT NPU: timing only.\nOutputs may be incorrect or vary.",
                xalign=0,
            )
            info.set_line_wrap(True)
            self.info = info
            side.pack_start(info, False, False, 0)
            quit_button = Gtk.Button(label="Exit")
            quit_button.connect("clicked", lambda *_: self.destroy())
            side.pack_end(quit_button, False, False, 0)
            self.status = Gtk.Label(label="Starting…", xalign=0)
            self.status.set_line_wrap(True)
            root.pack_start(self.status, False, False, 0)
            grid = Gtk.Grid(column_spacing=28, row_spacing=10)
            root.pack_start(grid, False, False, 0)
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
                grid.attach(label, col, 0, 1, 1)
            self.cells = {}
            for row, backend in enumerate(BACKENDS, 1):
                cells = [
                    Gtk.Label(label=t, xalign=0)
                    for t in [backend.replace("_", " "), "—", "—", "—", "—", "not measured"]
                ]
                for col, cell in enumerate(cells):
                    grid.attach(cell, col, row, 1, 1)
                self.cells[backend] = cells
            self.recorded_backends = set()
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
            history_box = Gtk.Expander(
                label="Measurement history (last 200; includes previous runs)"
            )
            history_box.add(scroll)
            root.pack_start(history_box, False, False, 0)
            try:
                for record in load_history(self.history_path):
                    self.show_measurement(record)
            except OSError as exc:
                print("Cannot load measurement history:", exc, flush=True)
            self.note = Gtk.Label(xalign=0)
            self.note.set_line_wrap(True)
            root.pack_start(self.note, False, False, 0)
            self.attribution = Gtk.Label(xalign=0)
            self.attribution.set_line_wrap(True)
            root.pack_start(self.attribution, False, False, 0)
            self.load_image()
            self.show_all()
            self.update_model_controls()
            if not args.windowed:
                self.fullscreen()
            GLib.timeout_add(100, self.poll)
            self.submit("prepare")

        def show_measurement(self, record):
            backend = record["backend"]
            values = list(record["values"])
            values[-1] += " / " + record.get("model_revision", "legacy-qdq")
            if record.get("model", "sam") == self.model:
                for cell, text in zip(self.cells[backend][1:], values):
                    cell.set_text(text)
                self.recorded_backends.add(backend)
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

        def record_measurement(self, backend, values, image=None):
            image = self.image_path if image is None else image
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
                    model_revision={"grounding": GROUNDING_REVISION, "depth": DEPTH_REVISION}.get(
                        self.model, MODEL_REVISION
                    ),
                    model=self.model,
                )
            self.show_measurement(record)

        def load_image(self):
            if (
                self.video_mode
                or Path(self.image_path).resolve().parent
                == (Path(__file__).parent / "images" / "bedroom").resolve()
            ):
                self.attribution.set_text(
                    "Demo images: Segment Anything official repository (SAM 2 / bedroom) — github.com/facebookresearch/sam2"
                )
            elif (
                Path(self.image_path).resolve()
                == (Path(__file__).parent / "images" / "truck.jpg").resolve()
            ):
                self.attribution.set_text(
                    "Demo image: Segment Anything official repository (truck.jpg) — github.com/facebookresearch/segment-anything"
                )
            else:
                self.attribution.set_text("Image: user-selected file")
            self.note.set_text(
                "Video: actual per-frame times, not 3-run medians. Sum = network calls; excludes loading/drawing. NPU includes CPU fallback."
                if self.video_mode
                else "ailia / ORT: warm-up at model Open. ailia memory mode 11. Compare all: median of 3 runs. Times exclude loading/drawing. NPU includes CPU fallback."
            )
            self.note.set_text(
                self.note.get_text() + " Previous values retained; check input/time in history."
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
            if self.model == "grounding":
                height, width = rgb.shape[:2]
                for item in self.detections:
                    x0, y0, x1, y1 = (
                        np.array(item["box"]) * [width, height, width, height]
                    ).astype(int)
                    cv2.rectangle(rgb, (x0, y0), (x1, y1), (50, 225, 170), 2)
                    cv2.putText(
                        rgb,
                        f"{item['label']} {item['score']:.2f}",
                        (x0, max(16, y0 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (50, 225, 170),
                        2,
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
            if self.model != "sam" or self.busy or event.button not in (1, 3):
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
            self.busy = True
            for control in self.controls:
                control.set_sensitive(False)
            self.status.set_text("Working… " + self.backend)
            backends = (
                [backend for backend in BACKENDS if backend != self.backend] + [self.backend]
                if action in ("benchmark", "video_compare")
                else [self.backend]
            )
            self.pending_jobs = [
                dict(
                    action=action,
                    backend=backend,
                    image=self.image_path,
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
            self.title_label.set_text(
                "Depth Anything V2 Metric Small / 518 × 518 / i.MX95"
                if self.model == "depth"
                else "Grounding DINO / 400 × 599 / i.MX95"
                if grounding
                else "SAM 2.1 Tiny / 512 × 512 / i.MX95"
            )
            self.prompt_box.set_visible(grounding)
            self.depth_controls.set_visible(self.model == "depth")
            for control in self.sam_controls:
                control.set_visible(sam)
            self.table_headers[1].set_text("Encoder (ms)" if sam else "Inference (ms)")
            for column in (2, 3, 4):
                self.table_headers[column].set_visible(sam)
                for cells in self.cells.values():
                    cells[column].set_visible(sam)
            self.info.set_text(
                "Indoor metric depth (Hypersim).\nDark: near / Bright: far.\nDisplay range changes colors only.\nClick to inspect estimated depth.\nailia NPU: fixed + dynamic MatMul.\nORT NPU: fixed MatMul only.\nCompare all: median of 3 runs.\n6 threads · no profile"
                if self.model == "depth"
                else "Enter English category names separated by periods.\nDetect: one warm inference.\nCompare all: median of 3 runs.\nNPU includes CPU fallback.\nORT NPU outputs may vary."
                if grounding
                else "Left: foreground (+)\nRight: background (−)\nBedroom: click frame 1, then Track.\nNPU: Encoder + MemoryAttention.\nOther tracking operations: CPU.\n6 threads · no profile\nORT NPU: timing only. Outputs may be incorrect."
            )

        def switch_model(self, selector):
            self.model = selector.get_active_id()
            self.runner.close()
            self.video_mode, self.video_frame = False, 0
            self.points, self.mask, self.detections = [], None, []
            self.depth = None
            if self.model == "depth":
                self.image_path = str(Path(__file__).parent / "images" / "bedroom" / "00000.jpg")
            self.recorded_backends.clear()
            for cells in self.cells.values():
                for cell in cells[1:]:
                    cell.set_text("—")
            try:
                records = load_history(self.history_path)
            except OSError as exc:
                print("Cannot load measurement history:", exc, flush=True)
                records = []
            for record in records:
                if record.get("model", "sam") == self.model:
                    values = list(record["values"])
                    values[-1] += " / " + record.get("model_revision", "legacy-qdq")
                    for cell, text in zip(self.cells[record["backend"]][1:], values):
                        cell.set_text(text)
                    self.recorded_backends.add(record["backend"])
            self.update_model_controls()
            self.load_image()
            self.submit("prepare")

        def switch(self, selector):
            self.backend = selector.get_active_id()
            self.depth = None
            if self.video_mode:
                self.image_path = str(Path(__file__).parent / "images" / "bedroom" / "00000.jpg")
                self.video_frame = 0
            self.mask = None
            self.detections = []
            self.load_image()
            self.submit("prepare")

        def benchmark(self, *_):
            if self.video_mode:
                self.track(compare=True)
            else:
                self.submit("benchmark")

        def bedroom(self, *_):
            self.video_mode, self.video_frame = True, 0
            self.image_path = str(Path(__file__).parent / "images" / "bedroom" / "00000.jpg")
            self.points, self.mask, self.video_records = [], None, []
            self.load_image()
            self.submit("prepare")

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
                self.video_mode = False
                self.image_path = dialog.get_filename()
                self.points, self.mask = [], None
                self.detections = []
                self.depth = None
                self.load_image()
                self.submit("prepare")
            dialog.destroy()

        def poll(self):
            if self.closing:
                return False
            while True:
                try:
                    result = self.runner.responses.get_nowait()
                except queue.Empty:
                    break
                kind = result["kind"]
                if kind == "progress":
                    self.status.set_text(result["text"])
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
                    self.status.set_text(
                        f"{backend} frame {result['frame'] + 1}/5 | MemoryAttention {times['memory_attention']:.1f} ms | processing {result['total_ms']:.1f} ms"
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
                    enc, click = result["encoder_ms"], result.get("click_ms")
                    # Preparing an image must not erase an existing benchmark
                    # (especially the last full video measurement).
                    if result.get("action") != "prepare" or backend not in self.recorded_backends:
                        self.record_measurement(
                            backend,
                            [
                                f"{enc:.1f}",
                                "—",
                                "—" if click is None else f"{click:.1f}",
                                "—" if click is None else f"{enc + click:.1f}",
                                result["mode"],
                            ],
                        )
                    if backend == self.backend and "mask" in result and self.points:
                        self.mask = result["mask"]
                        self.update_pixbuf()
                    if backend == self.backend and "detections" in result:
                        self.detections = result["detections"]
                        self.update_pixbuf()
                    if backend == self.backend and "depth" in result:
                        self.depth = result["depth"]
                        self.update_pixbuf()
                    record = {k: v for k, v in result.items() if k not in ("mask", "depth")}
                    record.update(
                        image=self.image_path, points=self.points, threads=6, profiling=False
                    )
                    self.results[self.model + ":" + backend] = record
                    try:
                        (Path(__file__).parent / "last_results.json").write_text(
                            json.dumps(self.results, indent=2)
                        )
                    except OSError as exc:
                        print("Cannot save timings:", exc, flush=True)
                elif kind in ("done", "error"):
                    if kind == "error" and self.model == "depth":
                        self.depth = None
                        self.update_pixbuf()
                    if kind == "error" and self.model == "grounding":
                        self.detections = []
                        self.update_pixbuf()
                    if kind == "done" and self.pending_jobs:
                        self.runner.submit(self.pending_jobs.pop(0))
                        continue
                    self.pending_jobs = []
                    self.busy = False
                    for control in self.controls:
                        control.set_sensitive(True)
                    self.status.set_text(result.get("text", "Ready — click the image."))
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
                    if args.snapshot:
                        GLib.timeout_add(300, self.snapshot)
            if self.runner.process is not None and not self.runner.process.is_alive():
                self.status.set_text(
                    "Inference process stopped. Close and restart the demo. See demo.log."
                )
                return False
            return True

        def snapshot(self):
            import cairo

            surface = cairo.ImageSurface(
                cairo.FORMAT_ARGB32, self.get_allocated_width(), self.get_allocated_height()
            )
            Gtk.Widget.draw(self, cairo.Context(surface))
            surface.write_to_png(str(args.snapshot))
            return False

        def shutdown(self, *_):
            self.closing = True
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
