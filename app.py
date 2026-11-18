#!/usr/bin/env python3
"""HDMI/Wayland GTK3 interactive demo. Inference runs in one separate process."""

import argparse
import json
import queue
import signal
from pathlib import Path

import cv2
import numpy as np

from engine import BACKENDS, fit_rect, image_point
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
    parser.add_argument("--image", type=Path, default=Path(__file__).parent / "images" / "truck.jpg")
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--backend", choices=BACKENDS, default="ailia_npu")
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
    import gi

    gi.require_version("Gtk", "3.0")
    gi.require_version("Gdk", "3.0")
    from gi.repository import Gdk, GdkPixbuf, GLib, GLibUnix, Gtk

    class Demo(Gtk.Window):
        def __init__(self):
            super().__init__(title="SAM 2.1 | i.MX95")
            self.set_default_size(1280, 720)
            self.connect("destroy", self.shutdown)
            self.points, self.mask, self.busy = [], None, False
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
            title = Gtk.Label(label="SAM 2.1 Tiny  /  512 × 512  /  i.MX95", xalign=0)
            title.get_style_context().add_class("title")
            root.pack_start(title, False, False, 0)
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
            self.controls = [self.selector]
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
                side.pack_start(button, False, False, 0)
            info = Gtk.Label(
                label="Left: foreground (+)\nRight: background (−)\nBedroom: click frame 1, then Track.\nNPU: Encoder + MemoryAttention.\nOther tracking operations: CPU.\nINT8 integer-v1 · 6 threads · no profile\nORT NPU: timing only.\nOutputs may be incorrect or vary.",
                xalign=0,
            )
            info.set_line_wrap(True)
            side.pack_start(info, False, False, 0)
            quit_button = Gtk.Button(label="Exit")
            quit_button.connect("clicked", lambda *_: self.destroy())
            side.pack_end(quit_button, False, False, 0)
            self.status = Gtk.Label(label="Starting…", xalign=0)
            self.status.set_line_wrap(True)
            root.pack_start(self.status, False, False, 0)
            grid = Gtk.Grid(column_spacing=28, row_spacing=10)
            root.pack_start(grid, False, False, 0)
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
                grid.attach(Gtk.Label(label=text, xalign=0), col, 0, 1, 1)
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
                    "Encoder",
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
            if not args.windowed:
                self.fullscreen()
            GLib.timeout_add(100, self.poll)
            self.submit("prepare")

        def show_measurement(self, record):
            backend = record["backend"]
            values = list(record["values"])
            values[-1] += " / " + record.get("model_revision", "legacy-qdq")
            for cell, text in zip(self.cells[backend][1:], values):
                cell.set_text(text)
            self.recorded_backends.add(backend)
            label = record["time"] + " / " + Path(record.get("image", "")).name
            self.history_rows.append([label, backend, *values])
            if len(self.history_rows) > 200:
                self.history_rows.remove(self.history_rows.get_iter_first())

        def record_measurement(self, backend, values, image=None):
            image = self.image_path if image is None else image
            try:
                record = append_history(self.history_path, backend, values, image)
            except OSError as exc:
                print("Cannot save measurement history:", exc, flush=True)
                record = dict(
                    time="unsaved",
                    backend=backend,
                    values=values,
                    image=image,
                    model_revision=MODEL_REVISION,
                )
            self.show_measurement(record)

        def load_image(self):
            if self.video_mode:
                self.attribution.set_text(
                    "Demo images: Segment Anything official repository (SAM 2 / bedroom) — github.com/facebookresearch/sam2"
                )
            elif Path(self.image_path).resolve() == (Path(__file__).parent / "images" / "truck.jpg").resolve():
                self.attribution.set_text(
                    "Demo image: Segment Anything official repository (truck.jpg) — github.com/facebookresearch/segment-anything"
                )
            else:
                self.attribution.set_text("Image: user-selected file")
            self.note.set_text(
                "Video: actual per-frame times, not 3-run medians. Sum = network calls; excludes loading/drawing. NPU includes CPU fallback."
                if self.video_mode
                else "ailia: warm-up at model Open; memory mode 11. Compare all: median of 3 runs. Times exclude loading/drawing. NPU includes CPU fallback."
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
            if self.busy or event.button not in (1, 3):
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
                )
                for backend in backends
            ]
            self.runner.submit(self.pending_jobs.pop(0))

        def switch(self, selector):
            self.backend = selector.get_active_id()
            if self.video_mode:
                self.image_path = str(Path(__file__).parent / "images" / "bedroom" / "00000.jpg")
                self.video_frame = 0
            self.mask = None
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
                    record = {k: v for k, v in result.items() if k != "mask"}
                    record.update(
                        image=self.image_path, points=self.points, threads=6, profiling=False
                    )
                    self.results[backend] = record
                    try:
                        (Path(__file__).parent / "last_results.json").write_text(
                            json.dumps(self.results, indent=2)
                        )
                    except OSError as exc:
                        print("Cannot save timings:", exc, flush=True)
                elif kind in ("done", "error"):
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
