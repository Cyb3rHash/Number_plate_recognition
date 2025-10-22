"""
Video utilities: lightweight MJPEG generator that overlays detections.

This module avoids any heavy initialization at import time. It expects a detector
object exposing detect_plates(image) and will call it per-frame. It uses OpenCV's
VideoCapture lazily when the generator starts.

The generator yields multipart JPEG frames suitable for Flask streaming responses.
"""

import time
from typing import Generator, Optional

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


def _draw_detections(frame, detections):
    """Overlay bounding boxes and confidence."""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det.get("confidence", 0.0)
        label = f"{det.get('class_name', 'plate')} {conf:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # type: ignore
        cv2.putText(frame, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)  # type: ignore


# PUBLIC_INTERFACE
def mjpeg_generator(detector, cam_index: int = 0, width: int = 640, height: int = 480, target_fps: int = 10) -> Generator[bytes, None, None]:
    """Yield MJPEG frames with overlaid detections using the provided detector.

    Args:
        detector: An object with detect_plates(image) -> List[dict] interface.
        cam_index: Camera index for cv2.VideoCapture.
        width: Capture width.
        height: Capture height.
        target_fps: Desired FPS to throttle processing.

    Yields:
        multipart JPEG frames (bytes) for Flask streaming.
    """
    if cv2 is None:
        # Fallback to synthetic frames if OpenCV isn't available
        import numpy as np  # type: ignore
        h, w = height, width
        while True:
            frame = (np.zeros((h, w, 3), dtype=np.uint8) + 30)
            cv2_msg = f"OpenCV not available; synthetic stream"
            # simple text
            # Using PIL would add dependency; rely on black frame
            _, jpeg = cv2.imencode(".jpg", frame) if cv2 else (True, frame)  # type: ignore
            frame_bytes = jpeg.tobytes() if cv2 else frame.tobytes()  # type: ignore
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n\r\n")
            time.sleep(1.0 / max(1, target_fps))
    else:
        cap = cv2.VideoCapture(cam_index)
        # Best-effort settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, target_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        try:
            last_time = 0.0
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.05)
                    continue

                # Optionally throttle
                now = time.time()
                if target_fps > 0:
                    min_interval = 1.0 / target_fps
                    if now - last_time < min_interval:
                        # Skip detection but still yield latest frame to keep stream smooth
                        _, jpeg = cv2.imencode(".jpg", frame)
                        frame_bytes = jpeg.tobytes()
                        yield (b"--frame\r\n"
                               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n\r\n")
                        continue
                    last_time = now

                # Run detection and draw
                try:
                    detections = detector.detect_plates(frame)
                except Exception:
                    detections = []
                _draw_detections(frame, detections)

                _, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                frame_bytes = jpeg.tobytes()
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n\r\n")
        finally:
            cap.release()
