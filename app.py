"""
Flask entrypoint for the Number Plate Recognition system.

This module exposes the Flask application instance required by the Flask CLI
so that `flask run --host 0.0.0.0 --port 3001` can discover and serve the app.

It imports the main application object from myapp.py and registers a lightweight
health-check route that is safe in environments without camera or MongoDB access.

Routes:
- GET /health: Returns 200 OK with JSON {"status": "ok"} to indicate the app is up.
"""

from flask import jsonify
from myapp import app  # Reuse the existing application and routes


# PUBLIC_INTERFACE
@app.get("/health")
def health():
    """Health check endpoint for preview/monitoring systems.

    Returns:
        JSON: {"status": "ok"} with HTTP 200 status to indicate the server is running.
    """
    return jsonify({"status": "ok"}), 200
