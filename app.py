"""
Flask entrypoint for the Number Plate Recognition system.

This module exposes a lightweight Flask application with:
- GET /health: Health check
- POST /detect: Plate detection via YOLOv8n (multipart/form-data or base64 JSON)
- GET /stream: MJPEG stream with on-frame detections (lazy camera open)

Design goals:
- Lazy-load the YOLO model on first use to keep startup fast.
- Performance configurable via environment variables.
- Graceful behavior if CUDA is not available (CPU fallback).
- Absolutely no blocking downloads or heavy work on import or first /health.
"""

import os
import time
from typing import Any

from flask import Flask, jsonify, request, Response, send_from_directory

# Ensure fast startup: create Flask app only
app = Flask(__name__)

# Global detector is never instantiated at import
_detector_instance = None


def _get_detector():
    """
    Lazy instantiate and cache the YOLO detector. Never at import time.
    Advanced detector preferred; falls back to simple detector. If both fail,
    returns an Exception which routes handle gracefully.
    """
    global _detector_instance
    if _detector_instance is None:
        try:
            from detection.advanced_yolo import Detector  # heavy optional
            _detector_instance = Detector()
        except Exception as e_adv:
            try:
                from detection.yolo_detector import YoloPlateDetector  # legacy/basic
                _detector_instance = YoloPlateDetector()
                print(f"[app] Using legacy YoloPlateDetector due to advanced detector error: {e_adv}")
            except Exception as e_basic:
                _detector_instance = Exception(f"Detector init failed: advanced={e_adv}; basic={e_basic}")
    return _detector_instance


def _error_response(message: str, status: int = 500):
    return jsonify({"success": False, "error": message}), status


# PUBLIC_INTERFACE
@app.get("/")
def index():
    """Landing endpoint for quick diagnostics.

    Returns:
        JSON: Summary with available endpoints and url_map_count.
    """
    rules = []
    for r in app.url_map.iter_rules():
        rules.append(
            {
                "rule": str(r),
                "methods": sorted([m for m in r.methods if m not in ("HEAD", "OPTIONS")]),
                "endpoint": r.endpoint,
            }
        )
    return jsonify(
        {
            "message": "Number Plate Recognition API",
            "endpoints": ["/", "/health", "/routes", "/detect", "/stream", "/video_feed", "/favicon.ico"],
            "url_map_count": len(rules),
        }
    ), 200


# PUBLIC_INTERFACE
@app.get("/health")
def health():
    """Health check endpoint for preview/monitoring systems.

    Notes:
        - Must remain fast and never trigger model/camera loads.
        - Provides basic readiness info (uptime).
    Returns:
        JSON: {"status": "ok", "uptime_ms": int}
    """
    # Fast readiness without touching detector or camera
    return jsonify({"status": "ok"}), 200


# PUBLIC_INTERFACE
@app.get("/routes")
def routes():
    """Return the Flask URL map for diagnostics.

    Returns:
        JSON: {"rules": [{"rule": "/path", "methods": ["GET", ...], "endpoint": "func_name"}]}
    """
    rules = []
    for r in app.url_map.iter_rules():
        rules.append(
            {
                "rule": str(r),
                "methods": sorted([m for m in r.methods if m not in ("HEAD", "OPTIONS")]),
                "endpoint": r.endpoint,
            }
        )
    return jsonify({ "rules": rules }), 200


def _read_image_from_request() -> Any:
    """
    Parse image from incoming request:
    - multipart/form-data: file under 'image'
    - application/json: base64 under 'image' key
    - raw bytes
    Returns bytes/base64 which detector will parse lazily.
    """
    if "image" in request.files:
        file = request.files["image"]
        return file.read()
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        img_b64 = payload.get("image")
        if not img_b64:
            raise ValueError("JSON body must include 'image' (base64 string)")
        return img_b64
    if request.data:
        return bytes(request.data)
    raise ValueError("No image provided. Use multipart 'image' file or JSON {'image': '<base64>'}.")


# PUBLIC_INTERFACE
@app.post("/detect")
def detect():
    """Run detection on a single image (lazy model load; no heavy work at import).

    Request:
      - multipart/form-data: file field 'image'
      - application/json: { "image": "<base64>" }
      - raw body: image bytes

    Response:
      - 200 JSON: { success, detections: [...], count }
      - 4xx/5xx JSON error without blocking server
    """
    det = _get_detector()
    if isinstance(det, Exception):
        return _error_response(f"Detector initialization error: {det}", 500)
    try:
        img = _read_image_from_request()
    except Exception as e:
        return _error_response(str(e), 400)

    try:
        results = det.detect_plates(img)
        output = []
        for d in results:
            out = dict(d)
            out.pop("crop", None)
            output.append(out)
        return jsonify({"success": True, "detections": output, "count": len(output)}), 200
    except Exception as e:
        return _error_response(str(e), 500)


# PUBLIC_INTERFACE
@app.get("/stream")
def stream():
    """MJPEG streaming endpoint with lazy camera open and non-blocking generator.

    Notes:
      - Never opens camera at import/startup.
      - If OpenCV/camera unavailable, yields synthetic frames or quick 503 JSON.
    """
    det = _get_detector()
    if isinstance(det, Exception):
        det = None
    try:
        from utils.video import mjpeg_generator
    except Exception as e:
        return _error_response(f"Video streaming unavailable: {e}", 503)

    gen = mjpeg_generator(det if det is not None else NoopDetector())
    return Response(gen, mimetype="multipart/x-mixed-replace; boundary=frame")


# PUBLIC_INTERFACE
@app.get("/video_feed")
def video_feed():
    """Deprecated alias of /stream."""
    return stream()


# PUBLIC_INTERFACE
@app.get("/favicon.ico")
def favicon():
    """Serve favicon.ico quickly and non-blocking."""
    try:
        return send_from_directory(os.path.join(app.root_path, "static"), "favicon.ico", mimetype="image/x-icon")
    except Exception:
        return Response(status=204)


class NoopDetector:
    """Fallback detector which returns no detections and never blocks."""
    # PUBLIC_INTERFACE
    def detect_plates(self, image):
        """Detect plates (noop)."""
        return []


if __name__ == "__main__":
    # Local debug server. Production should use gunicorn or flask run.
    print("Starting Flask app on port 3001 (threaded=True). Routes:")
    for r in app.url_map.iter_rules():
        if r.endpoint != "static":
            print(f"  {r.endpoint:20s} {sorted([m for m in r.methods if m not in ('HEAD','OPTIONS')])} -> {r}")
    app.run(host="0.0.0.0", port=3001, debug=True, threaded=True)
