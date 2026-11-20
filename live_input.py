"""Bounded, isolated OpenCV capture for USB cameras and looping video files."""

import multiprocessing as mp
import queue
import sys
import time
from pathlib import Path

import cv2


def find_camera(index=None, sysfs=Path("/sys/class/video4linux")):
    """Skip i.MX's internal video codec nodes when selecting a USB camera."""
    if index is not None:
        return index
    if sys.platform != "linux":
        return 0
    candidates = [
        int(node.name[5:])
        for node in sysfs.glob("video*")
        if node.name[5:].isdigit() and (node / "device" / "driver").resolve().name == "uvcvideo"
    ]
    if not candidates:
        raise ValueError("No USB camera found. Connect a UVC camera or specify --camera INDEX.")
    return min(candidates)


def capture_frames(source, output, stop):
    # Prefer OpenCV's MJPEG reader for AVI test clips: some BSP GStreamer
    # auto-decoders select an incompatible hardware JPEG pipeline.
    capture = cv2.VideoCapture()
    if isinstance(source, str) and source.lower().endswith(".avi"):
        capture.open(source, cv2.CAP_OPENCV_MJPEG)
    if not capture.isOpened():
        capture.open(source)
    try:
        if not capture.isOpened():
            output.put(("error", f"Cannot open camera/video: {source}"))
            return
        if isinstance(source, int):
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fps = capture.get(cv2.CAP_PROP_FPS)
        interval = 1 / fps if isinstance(source, str) and 0 < fps <= 240 else 0
        while not stop.is_set():
            started = time.monotonic()
            ok, frame = capture.read()
            if not ok and isinstance(source, str):
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = capture.read()
            item = ("frame", frame) if ok else ("error", "Camera/video stopped delivering frames")
            while not stop.is_set():
                try:
                    output.put_nowait(item)
                    break
                except queue.Full:
                    try:
                        output.get(timeout=0.01)
                    except queue.Empty:
                        pass
            if not ok:
                return
            stop.wait(max(0, interval - (time.monotonic() - started)))
    finally:
        capture.release()


class LiveInput:
    """One latest frame; a blocked camera driver cannot block the GTK thread."""

    def __init__(self, source):
        context = mp.get_context("spawn")
        self.output = context.Queue(maxsize=1)
        self.stop = context.Event()
        self.process = context.Process(
            target=capture_frames, args=(source, self.output, self.stop), daemon=True
        )
        self.process.start()
        self.last_frame = time.monotonic()

    def read(self):
        try:
            item = self.output.get_nowait()
            self.last_frame = time.monotonic()
            return item
        except queue.Empty:
            if not self.process.is_alive():
                return "error", "Camera/video capture stopped"
            if time.monotonic() - self.last_frame > 10:
                return "error", "Camera/video did not deliver a frame within 10 seconds"
            return None

    def close(self):
        self.stop.set()
        self.process.join(timeout=0.2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=0.5)
        if self.process.is_alive():
            self.process.kill()
            self.process.join()
        self.process.close()
        self.output.cancel_join_thread()
        self.output.close()
