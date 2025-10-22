# 🚗 Indian License Plate Recognition System (Advanced YOLOv8n)

Flask backend for real-time license plate detection with an advanced YOLOv8n stack:
- Env-driven config and lazy model warmup
- Optional GPU + half precision when available
- Async capture with frame queue and skip strategy for low latency streams
- Improved Indian-plate postprocessing and a 2-pass Tesseract OCR pipeline
- Temporal smoothing/cache and resilient Mongo writes
- Graceful degradation when dependencies are missing

What’s new:
- Advanced detector (detection/advanced_yolo.py) with warmup and OCR improvements
- Env-tunable thresholds (conf, iou, imgsz, max_det, classes), device/FP16
- Non-blocking MongoDB insert queue
- Streaming that doesn’t block startup

Endpoints:
- /, /health, /routes
- /detect (POST image)
- /stream (GET MJPEG)
- /video_feed (alias of /stream)
- /favicon.ico

## 🧠 Tech Stack
- Python, Flask
- Ultralytics YOLOv8n (lazy load)
- OpenCV headless
- Tesseract OCR (optional, pip + system tesseract recommended)
- MongoDB (optional)

## 🔧 Environment Configuration
See .env.example for all keys. Common:
- MODEL_PATH=yolov8n.pt
- YOLO_DEVICE=auto           # cpu | cuda | index
- YOLO_HALF=false
- YOLO_IMG_SIZE=640
- YOLO_CONF=0.25
- YOLO_IOU=0.45
- YOLO_MAX_DET=300
- YOLO_CLASSES=
- VIDEO_SOURCE=0
- FRAME_SKIP=1
- QUEUE_SIZE=4
- OCR_PSM=7
- OCR_OEM=3
- OCR_WHITELIST=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789
- MONGO_URI=mongodb+srv://user:pass@cluster/vehicle_database?retryWrites=true&w=majority

Create a .env file populated from .env.example (do not commit secrets).

## 📦 Install
```bash
pip install -r requirements.txt
```
Notes:
- First inference may auto-download a YOLO model (Ultralytics).
- We use opencv-python-headless to avoid GUI libs in servers.

## 🚀 Run
Using Flask CLI:
```bash
export FLASK_APP=app:app
flask run --host 0.0.0.0 --port 3001
```

With gunicorn:
```bash
gunicorn -b 0.0.0.0:3001 wsgi:application
```

Direct:
```bash
python app.py
# or
python wsgi.py
```

Server starts fast; model loads on first use (or warmup).

## ✅ Health
- GET /health -> {"status":"ok"}

## 🧭 Routes overview
- /, /health, /routes
- /detect (POST)
- /stream (GET), /video_feed (alias)
- /favicon.ico

## 🧪 Detect API
POST /detect with:
1) multipart/form-data: image=@file
2) application/json: { "image": "<base64>" }
3) raw image bytes

Response:
```json
{
  "success": true,
  "detections": [
    { "bbox": [x1,y1,x2,y2], "confidence": 0.91, "class_id": 0, "class_name": "plate", "plate_text": "KA01AB1234" }
  ],
  "count": 1
}
```

## 📺 Streaming
- GET /stream (preferred)
- GET /video_feed (alias)
MJPEG with YOLO overlays. Camera opens lazily. If OpenCV is unavailable, a synthetic stream is served.

Embed:
```html
<img src="/stream" />
```

## ⚙️ Performance Tuning
- Device: YOLO_DEVICE=auto/cpu/cuda (indices allowed)
- Half precision: YOLO_HALF=true (CUDA only)
- Image size: YOLO_IMG_SIZE (e.g., 640 for balance; 416 for lower latency)
- Thresholds: YOLO_CONF, YOLO_IOU
- Max detections: YOLO_MAX_DET
- Class filter: YOLO_CLASSES=0,1
- Streaming: FRAME_SKIP (>=1), QUEUE_SIZE (>=1)

Tips:
- On CPU-only environments, reduce YOLO_IMG_SIZE to 416 or 480.
- On GPUs, enable YOLO_HALF=true for faster inference.
- For RTSP sources, set VIDEO_SOURCE to RTSP URL.

## 🧰 OCR
- Requires pytesseract (pip) and system tesseract-ocr.
- Two-pass OCR with denoise/sharpen.
- Whitelist tuned for Indian plate charset.
- Temporal smoothing caches multiple reads.

## 🗄️ MongoDB Writes
- Non-blocking queue with retry.
- Set MONGO_URI to enable.
- Data stored: bbox, confidence, class info, plate_text, timestamp.

## 🔒 Security
Do not expose endpoints publicly without auth, rate-limiting, and TLS.

## 🧰 Troubleshooting
- Ultralytics missing: /detect and /stream will return clear guidance. Install:
  ```bash
  pip install ultralytics
  ```
- pytesseract missing: OCR will be skipped; install:
  ```bash
  pip install pytesseract
  # plus system: sudo apt-get install tesseract-ocr
  ```
- No GPU: app runs on CPU. Force CPU with YOLO_DEVICE=cpu.
- Headless: ensure opencv-python-headless is installed (provided).
- Model loading issues: set MODEL_PATH to an accessible .pt file.
- Camera latency: increase FRAME_SKIP and set QUEUE_SIZE small (e.g., 2-4).

## 📁 Structure (relevant)
- app.py
- detection/
  - advanced_yolo.py        # Advanced detector (this)
  - yolo_detector.py        # Legacy/basic detector
- utils/
  - video.py                # MJPEG generator (compatible)
  - async_video.py          # Async camera helper
- templates/, static/
- requirements.txt
- .env.example
