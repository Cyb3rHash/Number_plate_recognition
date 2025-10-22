"""
Video utilities: lightweight MJPEG generator that overlays detections.

This module avoids any heavy initialization at import time. It expects a detector
object exposing detect_plates(image) and will call it per-frame. It uses OpenCV's
VideoCapture lazily when the generator starts.

The generator yields multipart JPEG frames suitable for Flask streaming responses.
"""

import time
from typing import Generator

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


def _draw_detections(frame, detections):
    """Overlay bounding boxes and confidence."""
    if cv2 is None:
        return
    for det in detections:
        x1, y1, x2, y2 = det.get("bbox", [0, 0, 0, 0])
        conf = float(det.get("confidence", 0.0))
        label = f"{det.get('class_name', 'plate')} {conf:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # type: ignore
        cv2.putText(frame, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)  # type: ignore


def _encode_jpeg(frame):
    if cv2 is None:
        # return raw if cv2 missing (won't be used normally)
        return True, frame
    return cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])


# PUBLIC_INTERFACE
def mjpeg_generator(detector, cam_index: int = 0, width: int = 640, height: int = 480, target_fps: int = 10) -> Generator[bytes, None, None]:
    """Yield MJPEG frames with overlaid detections using the provided detector.

    Args:
        detector: An object with detect_plates(image) -> List[dict] interface.
        cam_index: Camera index or RTSP for cv2.VideoCapture.
        width: Capture width.
        height: Capture height.
        target_fps: Desired FPS to throttle processing.

    Yields:
        multipart JPEG frames (bytes) for Flask streaming.

    Behavior:
        - If OpenCV is missing, yields synthetic frames.
        - If camera cannot be opened, quickly falls back to synthetic frames.
        - Never blocks app startup or /health responses.
    """
    import numpy as np  # local import for faster import time of module

    if cv2 is None:
        # Synthetic stream
        h, w = height, width
        while True:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            ok, jpeg = _encode_jpeg(frame)
            frame_bytes = jpeg.tobytes() if ok else frame.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n\r\n")
            time.sleep(1.0 / max(1, target_fps))
        # unreachable

    # Try to open camera with a short timeout loop; fallback to synthetic
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    start = time.time()
    opened = cap.isOpened()
    while not opened and (time.time() - start) < 1.0:
        time.sleep(0.05)
        opened = cap.isOpened()

    if not opened:
        # Fallback synthetic frames if camera cannot open
        try:
            cap.release()
        except Exception:
            pass
        h, w = height, width
        while True:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            ok, jpeg = _encode_jpeg(frame)
            frame_bytes = jpeg.tobytes() if ok else frame.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n\r\n")
            time.sleep(1.0 / max(1, target_fps))
        # unreachable

    try:
        last_time = 0.0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            now = time.time()
            min_interval = 1.0 / max(1, target_fps)
            if now - last_time < min_interval:
                # Yield current frame without detection to maintain smoothness
                ok_j, jpeg = _encode_jpeg(frame)
                frame_bytes = jpeg.tobytes() if ok_j else frame.tobytes()
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n\r\n")
                continue
            last_time = now

            # Run detection in a try/except to avoid breaking stream
            try:
                detections = detector.detect_plates(frame)
            except Exception:
                detections = []
            _draw_detections(frame, detections)

            ok_j, jpeg = _encode_jpeg(frame)
            frame_bytes = jpeg.tobytes() if ok_j else frame.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n\r\n")
    finally:
        try:
            cap.release()
        except Exception:
            pass
