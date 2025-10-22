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
"""

import os
import json
from typing import Any, Dict, List, Tuple

from flask import Flask, jsonify, request, Response, send_from_directory

# Create a fresh Flask app for API usage
app = Flask(__name__)

# Lazy import detector so import won't fail if ultralytics not installed yet.
_detector_instance = None


def _get_detector():
    """
    Lazy instantiate and cache the YOLO detector. Never at import time.
    """
    global _detector_instance
    if _detector_instance is None:
        # Prefer advanced detector; fallback to basic one if import fails
        try:
            from detection.advanced_yolo import Detector  # advanced stack
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
        JSON: Summary with available endpoints and url_map.
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

    Returns:
        JSON: {"status": "ok"} with HTTP 200 status to indicate the server is running.
    """
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
        # Skip Flask internal static endpoints unless useful
        rules.append(
            {
                "rule": str(r),
                "methods": sorted([m for m in r.methods if m not in ("HEAD", "OPTIONS")]),
                "endpoint": r.endpoint,
            }
        )
    return jsonify({"rules": rules}), 200


def _read_image_from_request() -> Any:
    """
    Parse image from incoming request:
    - multipart/form-data: file under 'image'
    - application/json: base64 under 'image' key
    - raw bytes
    Returns a numpy BGR array or base64/bytes for the detector to parse.
    """
    # Multipart form with file
    if "image" in request.files:
        file = request.files["image"]
        data = file.read()
        return data  # detector can handle bytes/base64

    # JSON with base64
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        img_b64 = payload.get("image")
        if not img_b64:
            raise ValueError("JSON body must include 'image' (base64 string)")
        return img_b64

    # Raw body
    if request.data:
        return bytes(request.data)

    raise ValueError("No image provided. Use multipart 'image' file or JSON {'image': '<base64>'}.")


# PUBLIC_INTERFACE
@app.post("/detect")
def detect():
    """Run detection on a single image.

    Request:
        - Content-Type: multipart/form-data with file field "image"
          OR
        - Content-Type: application/json with key "image" containing base64 data
          OR raw image bytes as request body

    Response JSON:
        {
          "success": true,
          "detections": [
            { "bbox": [x1,y1,x2,y2], "confidence": 0.91, "class_id": 0, "class_name": "plate" }
          ],
          "count": 1
        }

    Errors are returned as:
        { "success": false, "error": "<message>" }
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
        # Convert crops to none in API output to keep payload small (internal use only)
        output = []
        for d in results:
            out = dict(d)
            out.pop("crop", None)
            output.append(out)
        return jsonify({"success": True, "detections": output, "count": len(output)}), 200
    except Exception as e:
        # If model not present/ultralytics missing, surface clear error
        return _error_response(str(e), 500)


# PUBLIC_INTERFACE
@app.get("/stream")
def stream():
    """MJPEG streaming endpoint with YOLOv8 overlays.

    Returns:
        multipart/x-mixed-replace stream of JPEG frames.
    Notes:
        - Camera is lazily opened when the stream is requested.
        - If OpenCV is not available, a synthetic stream is served.
    """
    det = _get_detector()
    if isinstance(det, Exception):
        # Still allow a stream with no detections (synthetic or raw)
        det = None

    try:
        from utils.video import mjpeg_generator  # local import
    except Exception as e:
        return _error_response(f"Video streaming unavailable: {e}", 500)

    gen = mjpeg_generator(det if det is not None else NoopDetector())
    return Response(gen, mimetype="multipart/x-mixed-replace; boundary=frame")


# PUBLIC_INTERFACE
@app.get("/video_feed")
def video_feed():
    """Deprecated alias of /stream kept for backward compatibility.

    Returns:
        Same MJPEG stream as /stream.

    Notes:
        Prefer using /stream. This endpoint may be removed in a future release.
    """
    return stream()


# PUBLIC_INTERFACE
@app.get("/favicon.ico")
def favicon():
    """Serve site favicon.

    Returns:
        The favicon.ico static file with image/x-icon mimetype.
    """
    # Serve from the Flask 'static' folder
    try:
        return send_from_directory(os.path.join(app.root_path, "static"), "favicon.ico", mimetype="image/x-icon")
    except Exception:
        # Fallback: return 204 if icon missing (should not happen as we ship a placeholder)
        return Response(status=204)


class NoopDetector:
    """Fallback detector which returns no detections."""
    # PUBLIC_INTERFACE
    def detect_plates(self, image):
        """Detect plates (noop)."""
        return []


if __name__ == "__main__":
    # For local debug run. In production, use `flask run --host 0.0.0.0 --port 3001`
    print("Starting Flask app with the following URL rules:")
    for r in app.url_map.iter_rules():
        print(f"  {r.endpoint:20s} {sorted([m for m in r.methods if m not in ('HEAD','OPTIONS')])} -> {r}")
    print("URL map summary (ensure includes '/', '/health', '/routes', '/detect', '/stream', '/video_feed', '/favicon.ico'):")
    print(app.url_map)
    app.run(host="0.0.0.0", port=3001, debug=True, threaded=True)
