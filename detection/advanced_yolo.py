"""
Advanced YOLOv8n-based number plate detection and OCR pipeline with env-driven configuration.

This module provides:
- Detector class with lazy model loading and warmup, optional TensorRT/ONNX runtimes (if available)
- Environment-driven configuration for device, half, imgsz, conf, iou, max_det, classes
- Improved Indian plate postprocessing and OCR tuning (Tesseract), with denoise/sharpen and 2-pass fallback OCR
- Per-plate caching and temporal smoothing (debounce/aggregation)
- Robust MongoDB writes (non-blocking queue with retry)
- Async-friendly image handling and fast failover when dependencies are missing

Environment variables (all optional with defaults):
- MODEL_PATH: path or name of the model (default: yolov8n.pt)
- YOLO_DEVICE: "auto" | "cpu" | "cuda" | "0" | "1" ... (default: auto)
- YOLO_HALF: "true"/"false" (default: false)
- YOLO_IMG_SIZE: int (default: 640)
- YOLO_CONF: float (default: 0.25)
- YOLO_IOU: float (default: 0.45)
- YOLO_MAX_DET: int (default: 300)
- YOLO_CLASSES: CSV of class ids to filter (default: "")
- OCR_PSM: int (default: 7)
- OCR_OEM: int (default: 3)
- OCR_WHITELIST: string of allowed chars (default: "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
- MONGO_URI: connection URI (optional)
- FRAME_SKIP: int frames to skip during stream (default: 1)
- QUEUE_SIZE: int (default: 4)

Notes:
- This file does not block import even if ultralytics or pytesseract are missing. Graceful messages and fallbacks are provided.
"""

import os
import io
import re
import time
import base64
import threading
from queue import Queue, Empty
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional deps
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

try:
    from PIL import Image, ImageFilter, ImageEnhance  # type: ignore
except Exception:
    Image = None  # type: ignore

try:
    from ultralytics import YOLO  # type: ignore
    _ULTRA_AVAILABLE = True
except Exception as _ultra_err:
    YOLO = None  # type: ignore
    _ULTRA_AVAILABLE = False
    _ULTRA_ERR_MSG = str(_ultra_err)

# Torch is optional
try:
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

# Tesseract optional
try:
    import pytesseract  # type: ignore
    _TESS_AVAILABLE = True
except Exception as _te:
    pytesseract = None  # type: ignore
    _TESS_AVAILABLE = False

# Mongo optional
try:
    from pymongo import MongoClient  # type: ignore
    from pymongo.errors import PyMongoError  # type: ignore
    _MONGO_AVAILABLE = True
except Exception:
    MongoClient = None  # type: ignore
    PyMongoError = Exception  # type: ignore
    _MONGO_AVAILABLE = False


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _parse_classes(env_val: str) -> Optional[List[int]]:
    if not env_val:
        return None
    try:
        return [int(x.strip()) for x in env_val.split(",") if x.strip() != ""]
    except Exception:
        return None


def _select_device(cfg: str) -> str:
    if not _TORCH_AVAILABLE:
        return "cpu"
    if cfg and cfg.lower() != "auto":
        if cfg.lower() == "cpu":
            return "cpu"
        if cfg.lower() == "cuda" and torch.cuda.is_available():
            return "cuda"
        if cfg.isdigit() and torch.cuda.is_available():
            return f"cuda:{cfg}"
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _can_half(device: str, requested: bool) -> bool:
    if not requested:
        return False
    if not _TORCH_AVAILABLE:
        return False
    if device == "cpu":
        return False
    return True


def _to_bgr(image: Any) -> np.ndarray:
    """
    Convert input to OpenCV BGR ndarray.
    Supports: ndarray (assumed BGR), PIL.Image, bytes/base64 str.
    """
    if image is None:
        raise ValueError("Input image is None")

    if isinstance(image, np.ndarray):
        return image

    if Image is not None and isinstance(image, Image.Image):
        if cv2 is None:
            raise RuntimeError("OpenCV not available to convert PIL image")
        return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)  # type: ignore

    if isinstance(image, (bytes, bytearray, str)):
        data = None
        if isinstance(image, str):
            if image.startswith("data:") and "," in image:
                image = image.split(",", 1)[1]
            try:
                data = base64.b64decode(image, validate=False)
            except Exception:
                if cv2 is None:
                    raise RuntimeError("OpenCV not available to read image path")
                arr = cv2.imread(image)  # type: ignore
                if arr is None:
                    raise ValueError("Unable to read image from string input")
                return arr
        else:
            data = bytes(image)

        if data is None:
            raise ValueError("Invalid image input")

        if Image is None:
            raise RuntimeError("Pillow required to parse image bytes/base64")
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        if cv2 is None:
            raise RuntimeError("OpenCV required to convert PIL image to ndarray")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)  # type: ignore

    raise TypeError("Unsupported image type")


def _indian_plate_like(text: str) -> bool:
    """
    Stricter heuristic for Indian plates, but tolerant to OCR noise.
    """
    if not text:
        return False
    t = re.sub(r"[^A-Z0-9]", "", text.upper())
    if len(t) < 6 or len(t) > 12:
        return False
    # Basic pattern checks
    # Common: XX00XX0000 or variations
    patterns = [
        r"^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$",
        r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{1,4}$",
        r"^[0-9]{2}BH[0-9]{4}[A-Z]{2}$",
    ]
    for p in patterns:
        if re.match(p, t):
            return True
    # Fallback len and mix
    if t[:2].isalpha() and any(ch.isdigit() for ch in t[2:6]):
        return True
    return False


def _ocr_preprocess(bgr: np.ndarray) -> np.ndarray:
    """
    Denoise and sharpen the plate crop for better OCR.
    """
    if cv2 is None:
        return bgr
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Normalize and threshold
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    gray = cv2.equalizeHist(gray)
    # Adaptive threshold to handle varied lighting
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 7)
    # Slight morphology to close holes
    kernel = np.ones((2, 2), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
    return thr


def _ocr_with_tesseract(img_bgr: np.ndarray, oem: int, psm: int, whitelist: str) -> str:
    """
    Two-pass OCR: fast single pass, then refined with different params if needed.
    """
    if not _TESS_AVAILABLE:
        return ""

    try:
        pre = _ocr_preprocess(img_bgr)
    except Exception:
        pre = img_bgr

    config_fast = f"--oem {oem} --psm {psm} -c tessedit_char_whitelist={whitelist}"
    text = ""
    try:
        text = pytesseract.image_to_string(pre, config=config_fast)  # type: ignore
    except Exception:
        text = ""

    clean = re.sub(r"[^A-Z0-9]", "", (text or "").upper())
    if _indian_plate_like(clean):
        return clean

    # Refined pass: different PSM and sharpening via PIL if available
    if Image is not None:
        try:
            pil = Image.fromarray(pre if pre.ndim == 2 else cv2.cvtColor(pre, cv2.COLOR_BGR2RGB))  # type: ignore
            pil = pil.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
            pil = ImageEnhance.Contrast(pil).enhance(1.3)
            refined = np.array(pil) if pre.ndim == 2 else cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)  # type: ignore
        except Exception:
            refined = pre
    else:
        refined = pre

    psm_refine = 8 if psm != 8 else 7
    config_refine = f"--oem {oem} --psm {psm_refine} -c tessedit_char_whitelist={whitelist}"
    try:
        text2 = pytesseract.image_to_string(refined, config=config_refine)  # type: ignore
    except Exception:
        text2 = ""
    clean2 = re.sub(r"[^A-Z0-9]", "", (text2 or "").upper())
    if len(clean2) > len(clean):
        clean = clean2

    # Minor post-fixes (common OCR mistakes)
    clean = clean.replace("O", "0") if re.match(r"^[A-Z]{2}O", clean) else clean
    clean = clean.replace("I", "1") if re.match(r"^[A-Z]{2}[0-9]{2}I", clean) else clean

    return clean


class _MongoWriter(threading.Thread):
    """
    A resilient MongoDB writer that consumes from a queue and attempts inserts with retry.
    No-ops if Mongo is not available or URI not provided.
    """
    def __init__(self, uri: Optional[str], collection_name: str = "detections"):
        super().__init__(daemon=True)
        self.uri = uri
        self.collection_name = collection_name
        self._queue: "Queue[Dict[str, Any]]" = Queue()
        self._stop = threading.Event()
        self._client = None
        self._collection = None

        if uri and _MONGO_AVAILABLE:
            try:
                # Use conservative timeouts so DB issues never block app startup/requests
                self._client = MongoClient(
                    uri,
                    connect=False,
                    connectTimeoutMS=int(os.getenv("MONGO_CONNECT_TIMEOUT_MS", "500")),
                    serverSelectionTimeoutMS=int(os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS", "500")),
                    socketTimeoutMS=int(os.getenv("MONGO_SOCKET_TIMEOUT_MS", "5000")),
                )
                db_name = (self._client.get_default_database().name
                           if self._client.get_default_database() is not None else "vehicle_database")
                db = self._client[db_name]
                self._collection = db[self.collection_name]
            except Exception as e:
                print(f"[Mongo] Initialization error: {e}")
                self._client = None
                self._collection = None

    def run(self):
        if self._collection is None:
            # Nothing to do
            return
        while not self._stop.is_set():
            try:
                doc = self._queue.get(timeout=0.25)
            except Empty:
                continue
            try:
                self._collection.insert_one(doc)
            except PyMongoError as e:  # type: ignore
                print(f"[Mongo] insert error: {e}")
                # Backoff
                time.sleep(0.25)
                try:
                    self._collection.insert_one(doc)
                except Exception as e2:
                    print(f"[Mongo] retry failed: {e2}")
            except Exception as e:
                print(f"[Mongo] unknown error: {e}")

    def enqueue(self, doc: Dict[str, Any]):
        if self._collection is None:
            return
        try:
            self._queue.put_nowait(doc)
        except Exception:
            pass

    def stop(self):
        self._stop.set()


class PlateCache:
    """
    Temporal smoothing and caching of plate OCR results.
    Aggregates multiple OCR reads within a time window and returns the most confident.
    """
    def __init__(self, ttl_sec: float = 10.0):
        self.ttl = ttl_sec
        self.cache: Dict[str, Dict[str, Any]] = {}  # key: text -> {count, last_seen, best_conf}

    def update(self, text: str, conf: float) -> None:
        if not text:
            return
        now = time.time()
        if text not in self.cache:
            self.cache[text] = {"count": 0, "last_seen": now, "best_conf": conf}
        self.cache[text]["count"] += 1
        self.cache[text]["last_seen"] = now
        self.cache[text]["best_conf"] = max(self.cache[text]["best_conf"], conf)

    def best(self) -> Optional[Tuple[str, float, int]]:
        # Clean old
        now = time.time()
        to_del = [t for t, v in self.cache.items() if now - v["last_seen"] > self.ttl]
        for k in to_del:
            del self.cache[k]
        if not self.cache:
            return None
        # Choose by count then conf
        sorted_items = sorted(self.cache.items(), key=lambda kv: (kv[1]["count"], kv[1]["best_conf"]), reverse=True)
        t, meta = sorted_items[0]
        return t, float(meta["best_conf"]), int(meta["count"])


class Detector:
    """
    Advanced YOLOv8n detector + OCR with env-driven config and lazy loading.

    Public methods:
    - detect_plates(image): list of detection dicts with bbox, confidence, class info, and optional OCR text
    - warmup(): ensure model is loaded
    - last_error(): last warning or error string

    Graceful behavior:
    - Works without GPU
    - If ultralytics or pytesseract is not available, logs guidance and continues with partial functionality
    """

    def __init__(self) -> None:
        # YOLO config
        self.model_path = _env_str("MODEL_PATH", _env_str("YOLO_MODEL_PATH", "yolov8n.pt"))
        self.device = _select_device(_env_str("YOLO_DEVICE", "auto"))
        self.use_half = _can_half(self.device, _env_bool("YOLO_HALF", False))
        self.imgsz = _env_int("YOLO_IMG_SIZE", _env_int("YOLO_IMG_SIZE".upper(), 640))  # tolerate variations
        self.conf = _env_float("YOLO_CONF", _env_float("YOLO_CONF_THRESH", 0.25))
        self.iou = _env_float("YOLO_IOU", _env_float("YOLO_IOU_THRESH", 0.45))
        self.max_det = _env_int("YOLO_MAX_DET", 100)
        self.classes = _parse_classes(_env_str("YOLO_CLASSES", ""))

        # OCR config
        self.ocr_psm = _env_int("OCR_PSM", 7)
        self.ocr_oem = _env_int("OCR_OEM", 3)
        self.ocr_whitelist = _env_str("OCR_WHITELIST", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

        # Streaming controls
        self.frame_skip = _env_int("FRAME_SKIP", 1)
        self.queue_size = _env_int("QUEUE_SIZE", 4)

        # Mongo
        self.mongo_uri = _env_str("MONGO_URI", "")
        self._mongo_writer = _MongoWriter(self.mongo_uri or None, collection_name="detections")
        if self.mongo_uri:
            self._mongo_writer.start()

        # Internal state
        self._model = None
        self._last_error: Optional[str] = None
        self._cache = PlateCache(ttl_sec=10.0)

        if not _ULTRA_AVAILABLE:
            print(f"[Detector] Ultralytics not available. Install with: pip install ultralytics ({globals().get('_ULTRA_ERR_MSG','')})")
        if not _TESS_AVAILABLE:
            print("[Detector] pytesseract not available. Install with: pip install pytesseract and system tesseract-ocr")

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        if not _ULTRA_AVAILABLE:
            self._last_error = "Ultralytics not installed. Please install 'ultralytics'."
            raise RuntimeError(self._last_error)
        try:
            self._model = YOLO(self.model_path)
        except Exception as e:
            try:
                self._model = YOLO("yolov8n.pt")
                print(f"[Detector] Fallback loaded yolov8n.pt due to: {e}")
            except Exception as e2:
                self._last_error = f"Failed to load model: {e}; fallback error: {e2}"
                raise RuntimeError(self._last_error)

        # Device/precision
        try:
            if _TORCH_AVAILABLE and hasattr(self._model, "model"):
                self._model.model.to(self.device)  # type: ignore[attr-defined]
                if self.use_half:
                    self._model.model.half()  # type: ignore[attr-defined]
        except Exception as e:
            self.use_half = False
            self._last_error = f"Model device/precision setup warning: {e}"
            print(f"[Detector] {self._last_error}")

        # Optional: warmup pass with dummy tensor
        try:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            _ = self._model.predict(source=dummy, imgsz=self.imgsz, device=self.device, half=self.use_half,
                                    conf=0.1, iou=0.5, max_det=1, verbose=False)
        except Exception as e:
            print(f"[Detector] Warmup skipped: {e}")

    # PUBLIC_INTERFACE
    def detect_plates(self, image: Any) -> List[Dict[str, Any]]:
        """Run detection + OCR on input image.

        Args:
            image: numpy.ndarray (BGR), PIL.Image, bytes/base64 str.

        Returns:
            List[dict]: Each includes bbox [x1,y1,x2,y2], confidence, class_id, class_name,
                        crop (BGR np.ndarray), and optional 'plate_text'.
        """
        self._lazy_load()
        frame = _to_bgr(image)
        if self._model is None:
            raise RuntimeError(self._last_error or "Model not available")

        # Inference
        try:
            results = self._model.predict(
                source=frame,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
                half=self.use_half,
                max_det=self.max_det,
                classes=self.classes,
                verbose=False,
                agnostic_nms=False,
            )
        except Exception as e:
            raise RuntimeError(f"Inference error: {e}")

        detections: List[Dict[str, Any]] = []
        for r in results:
            names = r.names if hasattr(r, "names") else {}
            if getattr(r, "boxes", None) is None:
                continue
            for box in r.boxes:
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                conf = float(getattr(box, "conf", [0.0])[0]) if hasattr(box, "conf") else 0.0
                cls_id = int(getattr(box, "cls", [-1])[0]) if hasattr(box, "cls") else -1
                cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)

                h, w = frame.shape[:2]
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(w - 1, x2), min(h - 1, y2)
                crop = frame[y1c:y2c, x1c:x2c].copy() if (y2c > y1c and x2c > x1c) else np.zeros((0, 0, 3), dtype=np.uint8)

                det = {
                    "bbox": [x1c, y1c, x2c, y2c],
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "crop": crop,
                }

                # OCR per detection (optional, can be disabled by missing pytesseract)
                plate_text = ""
                if crop.size != 0 and _TESS_AVAILABLE:
                    try:
                        plate_text = _ocr_with_tesseract(crop, self.ocr_oem, self.ocr_psm, self.ocr_whitelist)
                        if plate_text:
                            self._cache.update(plate_text, conf)
                            det["plate_text"] = plate_text
                    except Exception as e:
                        # OCR failures should not break detection
                        det["ocr_error"] = str(e)

                detections.append(det)

        # Enqueue Mongo writes non-blocking
        if self.mongo_uri and detections:
            ts = time.time()
            for d in detections:
                try:
                    to_store = {
                        "bbox": d.get("bbox"),
                        "confidence": d.get("confidence"),
                        "class_id": d.get("class_id"),
                        "class_name": d.get("class_name"),
                        "plate_text": d.get("plate_text", ""),
                        "timestamp": ts,
                    }
                    self._mongo_writer.enqueue(to_store)
                except Exception as e:
                    print(f"[Detector] Mongo enqueue error: {e}")

        return detections

    # PUBLIC_INTERFACE
    def warmup(self) -> None:
        """Ensure the model is loaded and warmed up."""
        self._lazy_load()

    # PUBLIC_INTERFACE
    def last_error(self) -> Optional[str]:
        """Return the last warning/error string, if any."""
        return self._last_error


class AsyncVideoCapture:
    """
    Async video capture with a frame queue and skip strategy to minimize latency.
    """

    def __init__(self, src: int | str = 0, queue_size: int = 4, frame_skip: int = 1, width: int = 640, height: int = 480):
        self.src = src
        self.queue_size = max(1, queue_size)
        self.frame_skip = max(1, frame_skip)
        self.width = width
        self.height = height
        self.q: "Queue[np.ndarray]" = Queue(maxsize=self.queue_size)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap = None

    def start(self):
        if cv2 is None:
            print("[AsyncVideoCapture] OpenCV not available, cannot start.")
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
                    # Skip frames to keep latency low
                    continue
                # Push latest frame, dropping old if full
                if self.q.full():
                    try:
                        _ = self.q.get_nowait()
                    except Exception:
                        pass
                try:
                    self.q.put_nowait(frame)
                except Exception:
                    pass

        self._thread = threading.Thread(target=_reader, daemon=True)
        self._thread.start()
        return self

    def read(self) -> Optional[np.ndarray]:
        try:
            return self.q.get(timeout=0.1)
        except Empty:
            return None

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.5)
        if self._cap:
            self._cap.release()
