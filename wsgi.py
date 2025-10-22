"""
WSGI entrypoint for the Number Plate Recognition Flask application.

This file ensures the correct Flask app (from app.py) is served by default
in environments where FLASK_APP is not set. Most WSGI servers (e.g., gunicorn)
look for a module-level variable named `application`.

App summary:
- GET /health  -> health check
- GET /routes  -> diagnostics of all routes
- POST /detect -> YOLOv8n detection API
- GET /stream  -> MJPEG streaming with optional YOLO overlay
"""
from app import app as application  # Expose as `application` for WSGI servers

# Optional: allow running with `python wsgi.py` for quick local checks
if __name__ == "__main__":
    # For local debug; preview systems will import `application`
    print("Starting via wsgi.py; serving `application` from app.py")
    # Print routes to help diagnose if 404s occur
    for r in application.url_map.iter_rules():
        if r.endpoint != "static":
            print(f"  {r.endpoint:20s} {sorted([m for m in r.methods if m not in ('HEAD','OPTIONS')])} -> {r}")
    application.run(host="0.0.0.0", port=3001, debug=True, threaded=True)
