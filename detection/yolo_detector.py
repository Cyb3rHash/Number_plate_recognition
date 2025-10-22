"""
YOLOv8n-based number plate detector with lazy-loading and performance configuration.

This module provides a lazy-loaded Ultralytics YOLO detector tailored for number plate
detection. It reads configuration from environment variables, selects GPU if available,
and supports half precision when compatible.

Environment variables:
- YOLO_MODEL_PATH: Path or model name (default: 'yolov8n.pt')
- YOLO_CONF_THRESH: Confidence threshold (default: 0.25)
- YOLO_IOU_THRESH: IoU threshold for NMS (default: 0.45)
- YOLO_IMG_SIZE: Inference image size (default: 640)
- YOLO_DEVICE: 'auto' (default), 'cpu', 'cuda', or explicit device like '0'
- YOLO_HALF: 'true' or 'false' (default: false)

Notes:
- Model is not loaded at import time. It loads on the first call to detect_plates or warmup().
- If CUDA is not available or incompatible with half precision, it gracefully falls back to CPU/full precision.
"""

import os
import base64
import io
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore

# Optional import of ultralytics, handled gracefully
try:
    from ultralytics import YOLO  # type: ignore
    _ULTRA_AVAILABLE = True
except Exception as _e:  # pragma: no cover
    YOLO = None  # type: ignore
    _ULTRA_AVAILABLE = False

# Torch is optional but used if available to choose device/half
try:
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


def _get_env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "y", "on")


def _get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _get_env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _select_device(cfg_device: str) -> str:
    """
    Selects appropriate device string.
    """
    if not _TORCH_AVAILABLE:
        return "cpu"

    if cfg_device and cfg_device.lower() != "auto":
        # Respect explicit device request
        if cfg_device.lower() == "cuda" and torch.cuda.is_available():
            return "cuda"
        if cfg_device.lower() == "cpu":
            return "cpu"
        # Could be a device idx like '0'
        if cfg_device.isdigit() and torch.cuda.is_available():
            return f"cuda:{cfg_device}"
        return "cpu"

    # Auto select
    return "cuda" if torch.cuda.is_available() else "cpu"


def _can_use_half(device: str, requested_half: bool) -> bool:
    """
    Determine if half precision is safe to enable.
    """
    if not requested_half:
        return False
    if not _TORCH_AVAILABLE:
        return False
    if device == "cpu":
        return False
    # For CUDA devices, half precision is supported
    return True


def _ensure_bgr(image: Any) -> np.ndarray:
    """
    Ensure input is an OpenCV BGR numpy array.
    Accepts:
      - numpy array (BGR or RGB; we assume BGR if 3-channel)
      - PIL Image
      - bytes/base64 encoded image
    """
    if image is None:
        raise ValueError("Input image is None")

    if isinstance(image, np.ndarray):
        # Assume BGR (cv2)
        return image

    if Image is not None and isinstance(image, Image.Image):
        # Convert PIL to BGR
        return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)  # type: ignore

    if isinstance(image, (bytes, bytearray, str)):
        # Try to parse as bytes or base64 string
        data = None
        if isinstance(image, str):
            # Strip potential data URL prefix
            if "," in image and image.strip().startswith("data:"):
                image = image.split(",", 1)[1]
            try:
                data = base64.b64decode(image, validate=False)
            except Exception:
                # Maybe it's a raw path; try to read
                if cv2 is None:
                    raise ValueError("OpenCV not available to read image path")
                arr = cv2.imread(image)  # type: ignore
                if arr is None:
                    raise ValueError("Unable to read image from string input")
                return arr
        else:
            data = bytes(image)

        if data is None:
            raise ValueError("Unable to parse image bytes")

        if Image is None:
            raise RuntimeError("Pillow is required to parse image bytes/base64")
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)  # type: ignore

    raise TypeError("Unsupported image type for detection")


class YoloPlateDetector:
    """
    YOLOv8n-based detector for number plate detection with lazy loading.

    Usage:
        detector = YoloPlateDetector()
        boxes = detector.detect_plates(image)  # image can be np.ndarray (BGR), PIL, or base64/bytes

    Returns:
        List of dicts:
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": float,
                "class_id": int,
                "class_name": str,
                "crop": np.ndarray (BGR)  # cropped plate region
            }
    """

    def __init__(self) -> None:
        # Configuration
        self.model_path = _get_env_str("YOLO_MODEL_PATH", "yolov8n.pt")
        self.conf = _get_env_float("YOLO_CONF_THRESH", 0.25)
        self.iou = _get_env_float("YOLO_IOU_THRESH", 0.45)
        self.imgsz = _get_env_int("YOLO_IMG_SIZE", 640)
        self.device = _select_device(_get_env_str("YOLO_DEVICE", "auto"))
        self.use_half = _can_use_half(self.device, _get_env_bool("YOLO_HALF", False))

        # Internal state
        self._model = None
        self._last_error: Optional[str] = None

    def _lazy_load(self) -> None:
        """
        Lazily load the YOLO model on first use.
        """
        if self._model is not None:
            return

        if not _ULTRA_AVAILABLE:
            self._last_error = "Ultralytics not installed. Please install 'ultralytics'."
            raise RuntimeError(self._last_error)

        # Try to load the specified model; allow auto-download
        try:
            self._model = YOLO(self.model_path)
        except Exception as e:
            # Fallback to yolov8n.pt
            try:
                self._model = YOLO("yolov8n.pt")
            except Exception as e2:
                self._last_error = f"Failed to load model: {e}; fallback error: {e2}"
                raise RuntimeError(self._last_error)

        # Attempt device and precision settings
        try:
            # move to device if torch available
            if _TORCH_AVAILABLE and hasattr(self._model, "model"):
                self._model.model.to(self.device)  # type: ignore[attr-defined]
                if self.use_half:
                    self._model.model.half()  # type: ignore[attr-defined]
        except Exception as e:
            # Gracefully disable half if it fails
            self.use_half = False
            self._last_error = f"Model device/precision setup warning: {e}"

    # PUBLIC_INTERFACE
    def detect_plates(self, image: Any) -> List[Dict[str, Any]]:
        """Run YOLOv8 detection on an image and return plate detections.

        Args:
            image: Input image. Can be:
                   - numpy.ndarray in BGR format (OpenCV)
                   - PIL.Image
                   - bytes/base64 string
        Returns:
            List[Dict]: Each dict contains:
                        - bbox: [x1, y1, x2, y2]
                        - confidence: float
                        - class_id: int
                        - class_name: str
                        - crop: numpy.ndarray (BGR crop of detection)
        Raises:
            RuntimeError: If the model cannot be loaded or Ultralytics/Torch is unavailable.
            ValueError: If the image cannot be parsed.
        """
        self._lazy_load()
        frame_bgr = _ensure_bgr(image)

        if self._model is None:
            raise RuntimeError(self._last_error or "Model not available")

        try:
            results = self._model.predict(
                source=frame_bgr,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
                half=self.use_half,
                verbose=False,
            )
        except Exception as e:
            raise RuntimeError(f"Inference error: {e}")

        detections: List[Dict[str, Any]] = []
        for r in results:
            names = r.names if hasattr(r, "names") else {}
            if getattr(r, "boxes", None) is None:
                continue
            for box in r.boxes:
                # xyxy format
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                conf = float(box.conf[0].item()) if hasattr(box, "conf") else 0.0
                cls_id = int(box.cls[0].item()) if hasattr(box, "cls") else -1
                cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)

                # Bounds and crop
                h, w = frame_bgr.shape[:2]
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(w - 1, x2), min(h - 1, y2)
                crop = frame_bgr[y1c:y2c, x1c:x2c].copy() if y2c > y1c and x2c > x1c else np.zeros((0, 0, 3), dtype=np.uint8)

                detections.append(
                    {
                        "bbox": [x1c, y1c, x2c, y2c],
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "crop": crop,
                    }
                )

        return detections

    # PUBLIC_INTERFACE
    def warmup(self) -> None:
        """Warm up the model by ensuring it's loaded and ready."""
        self._lazy_load()

    # PUBLIC_INTERFACE
    def last_error(self) -> Optional[str]:
        """Get the last warning/error message from the detector, if any."""
        return self._last_error
