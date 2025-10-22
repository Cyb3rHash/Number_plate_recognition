import os
import cv2
import numpy as np
from datetime import datetime, timedelta, timezone
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from flask import Flask, jsonify, redirect, render_template, Response, request, session, url_for
import threading
from pymongo import MongoClient
import re
from collections import deque
import time

# Use Ultralytics YOLOv8 instead of YOLO11 for better performance
# Make import optional to ensure the Flask app can start even if ultralytics isn't installed.
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception as _ultra_err:
    YOLO = None  # type: ignore
    ULTRALYTICS_AVAILABLE = False
    print(f"Ultralytics not available: {_ultra_err}. Video detection will be disabled until installed.")

app = Flask(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
db = client['vehicle_database']
vehicles_collection = db['vehicle']
history_collection = db['history']

class FastPlateDetector:
    def __init__(self, model_path='best.pt'):
        """
        Initialize optimized license plate detector.

        Important: We avoid loading YOLO at import time to ensure the Flask app
        can start even if 'ultralytics' or models are not available. The actual
        model is loaded lazily on first detection request.
        """
        # Defer YOLO model loading to detection time
        self.model = None
        self.model_path = model_path

        # Optimized parameters for speed
        self.confidence_threshold = 0.5  # Higher confidence to reduce false positives
        self.iou_threshold = 0.4

        # Indian license plate patterns
        self.indian_patterns = [
            r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$',  # KA01AB1234
            r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{1,4}$',  # Variations
            r'^[0-9]{2}BH[0-9]{4}[A-Z]{2}$',  # BH series
        ]

        # Confidence tracking with time-based decay
        self.plate_confidence = {}
        self.recognition_threshold = 2  # Reduced from 3 to 2 for faster recognition

        # Frame skipping for performance
        self.frame_skip = 2  # Process every 2nd frame
        self.frame_count = 0

        # Cache for recent detections
        self.detection_cache = {}
        self.cache_timeout = 30  # seconds

    def fast_preprocess_plate(self, img):
        """Optimized preprocessing for speed"""
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Only resize if too small
        width, height = img.size
        if width < 150:
            img = img.resize((width * 2, height * 2), Image.LANCZOS)
        
        # Convert to grayscale
        img = img.convert('L')
        
        # Fast contrast enhancement
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)  # Reduced from 2.0
        
        return img

    def fast_ocr(self, img):
        """Optimized OCR with minimal processing"""
        processed_img = self.fast_preprocess_plate(img)
        
        # Single optimized OCR config
        config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        try:
            text = pytesseract.image_to_string(processed_img, config=config)
            text = self.clean_ocr_result(text)
            if text and self.is_valid_indian_plate(text):
                return text
        except Exception as e:
            print(f"OCR error: {e}")
        
        return None

    def clean_ocr_result(self, text):
        """Fast cleaning of OCR output"""
        if not text:
            return None
            
        # Remove special characters and spaces
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Quick validation
        if len(text) < 6 or len(text) > 12:
            return None
            
        return text

    def is_valid_indian_plate(self, plate_text):
        """Fast validation"""
        if not plate_text or len(plate_text) < 6:
            return False
            
        # Quick format check (first 2 should be letters, next 2 should be numbers)
        if len(plate_text) >= 4:
            if not (plate_text[0:2].isalpha() and plate_text[2:4].isdigit()):
                return False
        
        return True

    def detect_plates_fast(self, frame):
        """Optimized detection with frame skipping"""
        self.frame_count += 1

        # Skip frames for performance
        if self.frame_count % self.frame_skip != 0:
            return []

        # Lazily load the YOLO model if available and not yet loaded
        if self.model is None:
            if ULTRALYTICS_AVAILABLE:
                try:
                    # Try custom trained model first
                    self.model = YOLO(self.model_path)
                    print(f"✓ YOLO model loaded lazily: {self.model_path}")
                except Exception as e:
                    # Fallback to a small pretrained model (may trigger download)
                    print(f"⚠ Could not load custom model '{self.model_path}': {e}. Falling back to yolov8n.pt")
                    try:
                        self.model = YOLO('yolov8n.pt')
                        print("✓ YOLO fallback model loaded lazily: yolov8n.pt")
                    except Exception as e2:
                        print(f"✗ YOLO fallback load failed: {e2}. Detection disabled.")
                        return []
            else:
                # Ultralytics not installed; cannot detect
                return []

        plates = []

        try:
            # Use smaller inference size for speed
            results = self.model(frame, conf=self.confidence_threshold, iou=self.iou_threshold,
                                 imgsz=640, verbose=False, half=True)  # half precision for speed

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()

                    # Filter by confidence
                    if conf < self.confidence_threshold:
                        continue

                    x, y = int(x1), int(y1)
                    w, h = int(x2 - x1), int(y2 - y1)

                    # Fast aspect ratio check
                    if w > 0 and h > 0:
                        aspect_ratio = w / float(h)
                        if 1.8 < aspect_ratio < 5.0 and w > 50 and h > 15:
                            plates.append((x, y, w, h, float(conf)))

        except Exception as e:
            print(f"Detection error: {e}")

        return plates

# Initialize optimized detector
plate_detector = FastPlateDetector()
vehicle_database = {}
recognized_plates = []
recent_detections = deque(maxlen=50)  # Smaller buffer

def load_vehicle_database():
    """Load vehicle database"""
    database = {}
    try:
        for vehicle in vehicles_collection.find():
            plate_number = vehicle['plate_number']
            database[plate_number] = {
                "owner_name": vehicle['owner_name'],
                "make": vehicle['make'],
                "model": vehicle['model'],
                "color": vehicle['color']
            }
        print(f"✓ Database loaded: {len(database)} vehicles")
    except Exception as e:
        print(f"Database error: {e}")
    return database

def optimized_video_processing():
    """Highly optimized video processing"""
    cap = cv2.VideoCapture(0)
    
    # Optimized camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduced resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for stability
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Single buffer to reduce latency
    
    # Warm-up camera
    for _ in range(5):
        cap.read()
    
    print("🚀 Starting optimized video processing...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Fast detection
        plates = plate_detector.detect_plates_fast(frame)
        
        for detection in plates:
            x, y, w, h, conf = detection
            
            # Extract plate region
            plate_roi = frame[y:y + h, x:x + w]
            
            # Fast OCR
            plate_number = plate_detector.fast_ocr(plate_roi)
            
            if plate_number:
                # Update confidence
                current_time = time.time()
                if plate_number not in plate_detector.plate_confidence:
                    plate_detector.plate_confidence[plate_number] = {'count': 0, 'last_seen': current_time}
                
                plate_detector.plate_confidence[plate_number]['count'] += 1
                plate_detector.plate_confidence[plate_number]['last_seen'] = current_time
                
                conf_data = plate_detector.plate_confidence[plate_number]
                
                # Check if plate is confirmed
                if (conf_data['count'] >= plate_detector.recognition_threshold 
                    and plate_number in vehicle_database):
                    
                    vehicle_info = vehicle_database[plate_number]
                    
                    # Check if this is a new detection
                    is_new = True
                    for det in recent_detections:
                        if (det['plate_number'] == plate_number and 
                            current_time - det['timestamp'].timestamp() < 30):
                            is_new = False
                            break
                    
                    if is_new:
                        plate_info = {
                            "plate_number": plate_number,
                            "owner_name": vehicle_info['owner_name'],
                            "make": vehicle_info['make'],
                            "model": vehicle_info['model'],
                            "color": vehicle_info['color'],
                            "timestamp": datetime.now(),
                            "confidence": conf_data['count'],
                            "detection_confidence": float(conf)
                        }
                        
                        recognized_plates.append(plate_info)
                        recent_detections.append(plate_info)
                        
                        # Save to database (non-blocking)
                        try:
                            history_collection.insert_one(plate_info.copy())
                            print(f"✓ Detected: {plate_number} - {vehicle_info['owner_name']}")
                        except Exception as e:
                            print(f"Database error: {e}")
                    
                    # Draw on frame
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    info_text = f"{plate_number} - {vehicle_info['owner_name']}"
                    cv2.putText(frame, info_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    conf_text = f"Conf: {conf_data['count']} | Det: {conf:.2f}"
                    cv2.putText(frame, conf_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                else:
                    # Show detection in progress
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                    status_text = f"{plate_number} ({conf_data['count']})"
                    cv2.putText(frame, status_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            
            else:
                # Show scanning
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Scanning...", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Clean old confidence data (memory management)
        current_time = time.time()
        expired_plates = []
        for plate, data in plate_detector.plate_confidence.items():
            if current_time - data['last_seen'] > 300:  # 5 minutes
                expired_plates.append(plate)
        
        for plate in expired_plates:
            del plate_detector.plate_confidence[plate]
        
        # Add performance info
        fps_text = f"FPS: {cap.get(cv2.CAP_PROP_FPS):.1f} | Active: {len([p for p in plate_detector.plate_confidence.values() if p['count'] >= 2])}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Encode frame
        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])  # Lower quality for speed
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
    
    cap.release()

# Update your video feed route
@app.route('/video_feed')
def video_feed():
    return Response(optimized_video_processing(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Keep your existing Flask routes the same...
app.secret_key = 'admin'
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'admin'

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error=True)
    else:
        return render_template('login.html', error=False)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/add_vehicle', methods=['POST'])
def add_vehicle():
    if 'logged_in' not in session:
        return jsonify({"success": False, "error": "Not logged in"})

    plate_number = request.form['plate_number'].upper()
    owner_name = request.form['owner_name']
    make = request.form['make']
    model = request.form['model']
    color = request.form['color']

    if not plate_detector.is_valid_indian_plate(plate_number):
        return jsonify({"success": False, "error": "Invalid Indian license plate format"})

    try:
        vehicles_collection.insert_one({
            "plate_number": plate_number,
            "owner_name": owner_name,
            "make": make,
            "model": model,
            "color": color,
            "created_at": datetime.now()
        })
        vehicle_database[plate_number] = {
            "owner_name": owner_name,
            "make": make,
            "model": model,
            "color": color
        }
        return jsonify({"success": True, "message": f"Vehicle {plate_number} added!"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/recognized_plates')
def recognized_plates_page():
    return render_template('recognized_plates.html', recognized_plates=recognized_plates)

@app.route('/index')
def index():
    try:
        past_24_hours = datetime.now() - timedelta(hours=24)
        data = history_collection.find({"timestamp": {"$gte": past_24_hours}})
        return render_template('index.html', data=data)
    except Exception as e:
        app.logger.error(f"Database error: {e}")
        return "Database error occurred"

@app.route('/records')
def records():
    try:
        database = vehicles_collection.find()
        return render_template('records.html', rec=database)
    except Exception as e:
        app.logger.error(f"Database error: {e}")
        return "Database error occurred"

@app.route('/api/stats')
def api_stats():
    """API endpoint for real-time statistics"""
    active_plates = len([p for p in plate_detector.plate_confidence.values() if p['count'] >= 2])
    return jsonify({
        'total_vehicles': len(vehicle_database),
        'detected_today': len([p for p in recognized_plates if (datetime.now() - p['timestamp']).days == 0]),
        'active_detections': active_plates,
        'performance': {
            'frame_skip': plate_detector.frame_skip,
            'confidence_threshold': plate_detector.confidence_threshold
        }
    })

@app.route('/api/history')
def api_history():
    """API endpoint for 24-hour detection history"""
    try:
        past_24_hours = datetime.now() - timedelta(hours=24)
        cursor = history_collection.find({"timestamp": {"$gte": past_24_hours}}).sort("timestamp", -1)
        history_data = []
        
        for record in cursor:
            record['_id'] = str(record['_id'])
            if isinstance(record.get('timestamp'), datetime):
                record['timestamp'] = record['timestamp'].isoformat()
            history_data.append(record)
        
        return jsonify({
            "success": True,
            "history": history_data,
            "count": len(history_data)
        })
    except Exception as e:
        print(f"Error fetching history: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/records')
def api_records():
    """API endpoint for all vehicle records"""
    try:
        cursor = vehicles_collection.find().sort("plate_number", 1)
        records_data = []
        
        for record in cursor:
            record['_id'] = str(record['_id'])
            if isinstance(record.get('created_at'), datetime):
                record['created_at'] = record['created_at'].isoformat()
            records_data.append(record)
        
        return jsonify({
            "success": True,
            "records": records_data,
            "count": len(records_data)
        })
    except Exception as e:
        print(f"Error fetching records: {e}")
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    # Load database
    vehicle_database = load_vehicle_database()
    
    print("🚀 Starting Optimized License Plate Recognition System...")
    print(f"✓ Loaded {len(vehicle_database)} vehicles")
    print("✓ Fast detector initialized")
    print("✓ System optimized for real-time performance!")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)