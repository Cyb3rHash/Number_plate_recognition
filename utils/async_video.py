"""
Async video capture utilities with frame queue and frame-skip strategy for low latency streaming.

This module is optional and can be used by streaming endpoints to minimize camera-induced latency.
"""

import os
import time
import threading
from queue import Queue, Empty
from typing import Generator, Optional

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore


class AsyncCamera:
    def __init__(self, src: int | str = None, width: int = 640, height: int = 480, frame_skip: int = 1, queue_size: int = 4):
        if src is None:
            src = os.getenv("VIDEO_SOURCE", "0")
            if src.isdigit():
                src = int(src)
        self.src = src
        self.width = width
        self.height = height
        self.frame_skip = max(1, frame_skip)
        self.queue: "Queue" = Queue(maxsize=max(1, queue_size))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap = None

    def start(self):
        if cv2 is None:
            print("[AsyncCamera] OpenCV not available; cannot start camera.")
            return self
        self._cap = cv2.VideoCapture(self.src)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, 15)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        def _reader():
            skip = 0
            while not self._stop.is_set():
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue
                skip = (skip + 1) % self.frame_skip
                if skip != 0:
                    continue
                if self.queue.full():
                    try:
                        self.queue.get_nowait()
                    except Exception:
                        pass
                try:
                    self.queue.put_nowait(frame)
                except Exception:
                    pass

        self._thread = threading.Thread(target=_reader, daemon=True)
        self._thread.start()
        return self

    def read(self):
        try:
            return self.queue.get(timeout=0.1)
        except Empty:
            return None

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.5)
        if self._cap:
            self._cap.release()
