import os
import cv2
import numpy as np
from datetime import datetime, timedelta, timezone
import pytesseract
import requests
from PIL import Image, ImageEnhance, ImageFilter
from flask import Flask, jsonify, redirect, render_template, Response, request, session, url_for
import threading
from pymongo import MongoClient
import re
from collections import deque
import time

app = Flask(__name__)

# Connect to MongoDB
client = MongoClient("mongodb+srv://vehicle:1234@cluster0.ygqnodr.mongodb.net/")
db = client['vehicle_database']

vehicles_collection = db['vehicle']
history_collection = db['history']

class IndianPlateDetector:
    def __init__(self):
        # Indian license plate patterns
        self.indian_patterns = [
            r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$',  # Standard format: KA01AB1234
            r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{1,4}$',  # Variations
            r'^[0-9]{2}BH[0-9]{4}[A-Z]{2}$',  # BH series
        ]
        
        # Confidence tracking for plates
        self.plate_confidence = {}
        self.recognition_threshold = 3  # Minimum detections before confirming
        
    def preprocess_plate_image(self, img):
        """Enhanced preprocessing for Indian license plates"""
        # Convert to PIL for better processing
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Resize for better OCR
        width, height = img.size
        if width < 200:
            img = img.resize((width * 3, height * 3), Image.LANCZOS)
        
        # Convert to grayscale
        img = img.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2.0)
        
        # Apply slight blur to reduce noise
        img = img.filter(ImageFilter.MedianFilter(size=3))
        
        return img
    
    def perform_ocr(self, img):
        """Optimized OCR for Indian license plates"""
        processed_img = self.preprocess_plate_image(img)
        
        # Multiple OCR configurations for better accuracy
        configs = [
            r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            r'--oem 3 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        ]
        
        results = []
        for config in configs:
            try:
                text = pytesseract.image_to_string(processed_img, config=config)
                text = self.clean_ocr_result(text)
                if text and self.is_valid_indian_plate(text):
                    results.append(text)
            except:
                continue
        
        # Return most common result
        if results:
            return max(set(results), key=results.count)
        return None
    
    def clean_ocr_result(self, text):
        """Clean and normalize OCR output"""
        if not text:
            return None
            
        # Remove special characters and spaces
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Common OCR corrections for Indian context
        corrections = {
            'O': '0', 'I': '1', 'S': '5', 'B': '8',
            'G': '6', 'Z': '2', 'T': '7'
        }
        
        # Apply corrections selectively based on position
        corrected = ""
        for i, char in enumerate(text):
            if i < 2 or (i >= 4 and i < 6):  # State code and series positions
                if char.isdigit():
                    # Convert numbers to letters in state code positions
                    digit_to_letter = {'0': 'O', '1': 'I', '5': 'S', '8': 'B'}
                    corrected += digit_to_letter.get(char, char)
                else:
                    corrected += char
            elif i >= 2 and i < 4:  # District code positions
                if char.isalpha():
                    # Convert letters to numbers in district code positions
                    corrected += corrections.get(char, char)
                else:
                    corrected += char
            else:
                corrected += char
        
        return corrected
    
    def is_valid_indian_plate(self, plate_text):
        """Validate if text matches Indian license plate format"""
        if not plate_text or len(plate_text) < 8:
            return False
            
        # Check against Indian patterns
        for pattern in self.indian_patterns:
            if re.match(pattern, plate_text):
                return True
        
        return False
    
    def detect_license_plates(self, frame):
        """Improved license plate detection using multiple methods"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Contour-based detection
        plates = self.detect_by_contours(gray, frame)
        
        # Method 2: Cascade classifier (backup)
        try:
            plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
            cascade_plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 30))
            plates.extend([(x, y, w, h) for (x, y, w, h) in cascade_plates])
        except:
            pass
        
        return plates
    
    def detect_by_contours(self, gray, frame):
        """Detect license plates using contour analysis"""
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Find edges
        edges = cv2.Canny(filtered, 30, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        plates = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
            # Approximate contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if contour has 4 corners (rectangular)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio typical for Indian license plates
                aspect_ratio = w / float(h)
                if 2.0 < aspect_ratio < 5.0 and w > 80 and h > 20:
                    plates.append((x, y, w, h))
        
        return plates

def load_vehicle_database():
    """Load vehicle database from MongoDB"""
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
        print(f"Vehicle database loaded successfully. {len(database)} vehicles found.")
    except Exception as e:
        print(f"Error loading database: {e}")
    return database

# Initialize components
vehicle_database = load_vehicle_database()
plate_detector = IndianPlateDetector()
recognized_plates = []
recent_detections = deque(maxlen=100)  # Keep track of recent detections

def process_video():
    """Enhanced video processing with better detection"""
    cap = cv2.VideoCapture(0)
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process every 3rd frame for better performance
        if frame_count % 3 != 0:
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
            continue
        
        # Detect license plates
        plates = plate_detector.detect_license_plates(frame)
        
        for (x, y, w, h) in plates:
            # Extract and enhance plate region
            plate_roi = frame[y:y + h, x:x + w]
            
            # Perform OCR
            plate_number = plate_detector.perform_ocr(plate_roi)
            
            if plate_number:
                # Update confidence tracking
                if plate_number not in plate_detector.plate_confidence:
                    plate_detector.plate_confidence[plate_number] = 0
                plate_detector.plate_confidence[plate_number] += 1
                
                # Check if plate is confirmed and in database
                if (plate_detector.plate_confidence[plate_number] >= plate_detector.recognition_threshold 
                    and plate_number in vehicle_database):
                    
                    vehicle_info = vehicle_database[plate_number]
                    current_time = datetime.now()
                    
                    # Check if this is a new detection (not detected in last 30 seconds)
                    is_new_detection = True
                    for detection in recent_detections:
                        if (detection['plate_number'] == plate_number and 
                            (current_time - detection['timestamp']).seconds < 30):
                            is_new_detection = False
                            break
                    
                    if is_new_detection:
                        # Store the recognized plate information
                        plate_info = {
                            "plate_number": plate_number,
                            "owner_name": vehicle_info['owner_name'],
                            "make": vehicle_info['make'],
                            "model": vehicle_info['model'],
                            "color": vehicle_info['color'],
                            "timestamp": current_time,
                            "confidence": plate_detector.plate_confidence[plate_number]
                        }
                        
                        recognized_plates.append(plate_info)
                        recent_detections.append(plate_info)
                        
                        # Insert into history collection
                        try:
                            history_collection.insert_one(plate_info.copy())
                            print(f"✓ Detected: {plate_number} - {vehicle_info['owner_name']}")
                        except Exception as e:
                            print(f"Error inserting to database: {e}")
                    
                    # Display information on frame
                    info_text = f"{plate_number} - {vehicle_info['owner_name']}"
                    confidence_text = f"Conf: {plate_detector.plate_confidence[plate_number]}"
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, info_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, confidence_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                else:
                    # Show detection in progress
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                    if plate_number:
                        conf_text = f"{plate_number} ({plate_detector.plate_confidence[plate_number]})"
                        cv2.putText(frame, conf_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            else:
                # Show unrecognized plate region
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Scanning...", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Add frame info
        cv2.putText(frame, f"Active Plates: {len([p for p in plate_detector.plate_confidence.values() if p >= plate_detector.recognition_threshold])}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Encode and yield frame
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
    
    cap.release()

# Flask routes (keeping your existing routes)
app.secret_key = 'admin'
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'admin'

@app.route('/video_feed')
def video_feed():
    return Response(process_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

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
    return jsonify({
        'total_vehicles': len(vehicle_database),
        'detected_today': len([p for p in recognized_plates 
                              if (datetime.now() - p['timestamp']).days == 0]),
        'active_detections': len([p for p in plate_detector.plate_confidence.values() 
                                 if p >= plate_detector.recognition_threshold])
    })

@app.route('/api/history')
def api_history():
    """API endpoint for 24-hour detection history"""
    try:
        past_24_hours = datetime.now() - timedelta(hours=24)
        cursor = history_collection.find({"timestamp": {"$gte": past_24_hours}}).sort("timestamp", -1)
        history_data = []
        
        for record in cursor:
            # Convert ObjectId to string for JSON serialization
            record['_id'] = str(record['_id'])
            # Convert datetime to ISO string
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
            # Convert ObjectId to string for JSON serialization
            record['_id'] = str(record['_id'])
            # Convert datetime to ISO string if exists
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
    return jsonify({
        'total_vehicles': len(vehicle_database),
        'detected_today': len([p for p in recognized_plates 
                              if (datetime.now() - p['timestamp']).days == 0]),
        'active_detections': len([p for p in plate_detector.plate_confidence.values() 
                                 if p >= plate_detector.recognition_threshold])
    })


if __name__ == '__main__':
    print("🚗 Starting Enhanced Indian License Plate Recognition System...")
    print(f"✓ Loaded {len(vehicle_database)} vehicles from database")
    print("✓ System ready!")
  
    app.run(debug=True, host='0.0.0.0', port=5000)