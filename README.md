# 🚗 Indian License Plate Recognition System (YOLOv8n-powered)

Flask backend for real-time license plate detection, now powered by Ultralytics YOLOv8n with lazy-loading to keep startup fast. Existing dashboard pages and MongoDB/OCR routes remain available for compatibility, while the new detection API and MJPEG streaming are optimized for inference.

What’s new:
- YOLOv8n detector with lazy-loading (model loads only on first use)
- New REST: POST /detect (multipart or base64 JSON)
- New MJPEG streaming: GET /stream with on-frame overlays
- Configurable performance via environment variables (GPU auto if available; optional FP16)
- Graceful fallback to CPU when CUDA is not present
- Lightweight GET /health stays for monitoring

## ⚙️ Features
- Real-time camera feed with YOLO overlay
- YOLOv8n-based plate detection (default yolov8n.pt; can use custom model via env)
- MongoDB integration for vehicle records (existing)
- Admin dashboard and stats (existing)
- OCR preprocessing support remains (legacy flow)

## 🧠 Tech Stack
- Python, Flask
- Ultralytics YOLOv8 (yolov8n)
- OpenCV (headless for servers)
- MongoDB (pymongo)
- Tesseract OCR (legacy OCR flows)
- HTML/CSS (Flask templates)

## 🔧 Environment Configuration
Set these environment variables to control inference behavior (defaults shown):

- YOLO_MODEL_PATH= yolov8n.pt
- YOLO_CONF_THRESH= 0.25
- YOLO_IOU_THRESH= 0.45
- YOLO_IMG_SIZE= 640
- YOLO_DEVICE= auto        # auto | cpu | cuda | 0 | 1 ...
- YOLO_HALF= false         # true to request FP16 (CUDA only)

If using MongoDB features (legacy dashboard/data):
- MONGO_URI=mongodb+srv://<user>:<pass>@<cluster>/<db>?retryWrites=true&w=majority

You may create a .env file with the above variables (the app reads typical environment variables; do not commit secrets).

## 📦 Install
```bash
pip install -r requirements.txt
```
Notes:
- The first detection may trigger an automatic model download (Ultralytics).
- We use opencv-python-headless for server-friendly environments.

## 🚀 Run
Using Flask CLI (recommended):
```bash
export FLASK_APP=app:app    # app.py exposes `app`
flask run --host 0.0.0.0 --port 3001
```

Or directly:
```bash
python app.py
```

Startup is fast because the YOLO model is not loaded until first use.

## ✅ Health
- GET http://localhost:3001/health
  - Response: {"status":"ok"}

## 🧪 Detect API
- POST http://localhost:3001/detect

Send either:
1) multipart/form-data
   - field: image (file)
2) application/json
   - { "image": "<base64_image_data>" }
3) raw body with image bytes

Response:
```json
{
  "success": true,
  "detections": [
    { "bbox": [x1,y1,x2,y2], "confidence": 0.91, "class_id": 0, "class_name": "plate" }
  ],
  "count": 1
}
```

Curl examples:
```bash
# multipart
curl -X POST http://localhost:3001/detect \
  -F "image=@/path/to/image.jpg"

# base64 JSON
python - <<'PY'
import base64, requests
with open("sample.jpg","rb") as f:
    b64 = base64.b64encode(f.read()).decode()
r = requests.post("http://localhost:3001/detect", json={"image": b64})
print(r.status_code, r.json())
PY
```

## 📺 Streaming
- GET http://localhost:3001/stream

Returns an MJPEG stream with YOLO overlays. The camera is opened lazily when the endpoint is requested. If OpenCV is unavailable, a synthetic placeholder stream is provided (no detections).

Embedding example:
```html
<img src="/stream" />
```

## ⚙️ Performance Tips
- GPU is auto-selected if available (YOLO_DEVICE=auto). Otherwise CPU is used.
- Enable FP16 on CUDA with YOLO_HALF=true.
- Tune thresholds via YOLO_CONF_THRESH / YOLO_IOU_THRESH.
- Adjust YOLO_IMG_SIZE (e.g., 640) for latency vs. accuracy trade-offs.
- The stream throttles frames for steady performance and lower CPU usage.

## 🧰 Model
Default model: yolov8n.pt (Ultralytics).
To use a custom fine-tuned model (e.g., trained for Indian plates), set:
```
YOLO_MODEL_PATH=/path/to/custom.pt
```
The model is loaded lazily on first detection or when warmup() is called.

## 📁 Project Structure (relevant parts)
- app.py                          # Flask entrypoint with /health, /detect, /stream
- detection/
  - __init__.py
  - yolo_detector.py              # Lazy-loaded YOLOv8n detector (env-configurable)
- utils/
  - video.py                      # Lightweight MJPEG generator with overlays
- templates/                      # Existing dashboard pages
- static/                         # Static assets
- requirements.txt                # Includes ultralytics and opencv-python-headless

## 📝 Legacy components (kept)
- Existing OCR and MongoDB flows, templates, and routes remain in the repo for compatibility (e.g., myapp.py, templates/*).
- The new detection endpoints do not require MongoDB to operate.

## 🔒 Security
Do not expose this service openly without proper authentication, authorization, and rate limiting in production.

## 🧰 Troubleshooting
- If ultralytics is not installed, /detect and /stream will return clear errors. Install using:
  ```
  pip install ultralytics
  ```
- If CUDA is not available, the app falls back to CPU. You can force CPU by setting `YOLO_DEVICE=cpu`.
- For headless environments, ensure `opencv-python-headless` is installed (provided in requirements.txt).
