from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import math
import threading
import queue
import time
import logging
import os
from dotenv import load_dotenv
import sys
from flask_sqlalchemy import SQLAlchemy
from models import db, User, Detection
from datetime import datetime
import pytz

load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Global variables
MODEL_PATH = os.getenv('MODEL_PATH', 'yolo11m.pt')
KNOWN_WIDTH = {
    # People
    'person': 0.5,        # Average shoulder width
    
    # Vehicles
    'bicycle': 0.6,       # Average bicycle width
    'car': 1.8,          # Average car width
    'motorcycle': 0.7,    # Average motorcycle width
    'airplane': 40.0,     # Average commercial airplane width
    'bus': 2.5,          # Average bus width
    'train': 3.0,        # Average train width
    'truck': 2.5,        # Average truck width
    'boat': 2.5,         # Average small boat width
    
    # Common objects
    'traffic light': 0.3, # Average traffic light width
    'fire hydrant': 0.3,  # Average fire hydrant width
    'stop sign': 0.6,     # Average stop sign width
    'parking meter': 0.2, # Average parking meter width
    'bench': 1.5,        # Average bench width
    
    # Animals
    'bird': 0.2,         # Average small bird width
    'cat': 0.3,          # Average cat width
    'dog': 0.4,          # Average dog width
    'horse': 0.6,        # Average horse width (shoulder width)
    'sheep': 0.5,        # Average sheep width
    'cow': 0.8,          # Average cow width
    'elephant': 3.0,     # Average elephant width
    'bear': 1.0,         # Average bear width
    'zebra': 0.7,        # Average zebra width
    'giraffe': 1.5,      # Average giraffe width (body)
    
    # Accessories
    'backpack': 0.3,     # Average backpack width
    'umbrella': 1.0,     # Average umbrella width when open
    'handbag': 0.3,      # Average handbag width
    'tie': 0.1,          # Average tie width
    'suitcase': 0.5,     # Average suitcase width
    
    # Sports equipment
    'frisbee': 0.25,     # Average frisbee diameter
    'skis': 0.1,         # Average ski width
    'snowboard': 0.3,    # Average snowboard width
    'sports ball': 0.2,  # Average sports ball diameter
    'kite': 1.0,         # Average kite width
    'baseball bat': 0.07, # Average baseball bat width
    'baseball glove': 0.25, # Average baseball glove width
    'skateboard': 0.2,   # Average skateboard width
    'surfboard': 0.5,    # Average surfboard width
    'tennis racket': 0.3, # Average tennis racket width
    
    # Indoor objects
    'bottle': 0.1,       # Average bottle width
    'wine glass': 0.08,  # Average wine glass width
    'cup': 0.1,          # Average cup width
    'fork': 0.03,        # Average fork width
    'knife': 0.03,       # Average knife width
    'spoon': 0.03,       # Average spoon width
    'bowl': 0.15,        # Average bowl width
    'banana': 0.03,      # Average banana width
    'apple': 0.08,       # Average apple width
    'sandwich': 0.15,    # Average sandwich width
    'orange': 0.08,      # Average orange width
    'broccoli': 0.15,    # Average broccoli width
    'carrot': 0.03,      # Average carrot width
    'hot dog': 0.15,     # Average hot dog width
    'pizza': 0.3,        # Average pizza width
    'donut': 0.1,        # Average donut width
    'cake': 0.25,        # Average cake width
    
    # Electronics
    'chair': 0.5,        # Average chair width
    'couch': 2.0,        # Average couch width
    'potted plant': 0.3, # Average potted plant width
    'bed': 1.5,          # Average bed width
    'dining table': 1.5, # Average dining table width
    'toilet': 0.6,       # Average toilet width
    'tv': 1.2,           # Average TV width
    'laptop': 0.35,      # Average laptop width
    'mouse': 0.06,       # Average computer mouse width
    'remote': 0.05,      # Average remote control width
    'keyboard': 0.4,     # Average keyboard width
    'cell phone': 0.07,  # Average cell phone width
    
    # Appliances
    'microwave': 0.5,    # Average microwave width
    'oven': 0.6,         # Average oven width
    'toaster': 0.3,      # Average toaster width
    'sink': 0.6,         # Average sink width
    'refrigerator': 0.8, # Average refrigerator width
    
    # Other
    'book': 0.15,        # Average book width
    'clock': 0.3,        # Average wall clock width
    'vase': 0.2,         # Average vase width
    'scissors': 0.1,     # Average scissors width
    'teddy bear': 0.3,   # Average teddy bear width
    'hair drier': 0.1,   # Average hair drier width
    'toothbrush': 0.02,  # Average toothbrush width
}

FOCAL_LENGTH = 1000   # This is an approximate value, might need calibration

# Initialize YOLO model globally
try:
    model = YOLO(MODEL_PATH)
    logger.info(f"YOLO model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading YOLO model: {e}")
    sys.exit(1)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///detectoo.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

with app.app_context():
    db.create_all()
    
class AudioFeedback:
    def __init__(self):
        self.engine = None
        self.speech_queue = queue.Queue(maxsize=1)
        self.speaking_thread = threading.Thread(target=self._speak_worker, daemon=True)
        self.speaking_thread.start()
        self.engine_lock = threading.Lock()
        self.is_speaking = threading.Event()

    def init_engine(self):
        if self.engine is not None:
            try:
                self.engine.stop()
            except:
                pass
            self.engine = None

        try:
            with self.engine_lock:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 175)
                voices = self.engine.getProperty('voices')
                for voice in voices:
                    if "david" in voice.id.lower() or "samantha" in voice.id.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
                self.engine.setProperty('volume', 0.9)
        except Exception as e:
            logger.error(f"Error initializing speech engine: {e}")
            self.engine = None

    def speak(self, text):
        if not text or self.is_speaking.is_set():
            return
        
        try:
            self.speech_queue.put_nowait(text)
        except queue.Full:
            pass

    def clear_queue(self):
        """Clear the speech queue and stop current speech"""
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except queue.Empty:
                break
        
        if self.engine is not None:
            try:
                self.engine.stop()
            except:
                pass

    def _speak_worker(self):
        while True:
            try:
                text = self.speech_queue.get()
                if self.engine is None:
                    self.init_engine()
                
                if self.engine is not None:
                    self.is_speaking.set()
                    with self.engine_lock:
                        self.engine.say(text)
                        self.engine.runAndWait()
                    self.is_speaking.clear()
                
                self.speech_queue.task_done()
            except Exception as e:
                logger.error(f"Speech worker error: {e}")
                self.is_speaking.clear()
                time.sleep(1)
                self.init_engine()

class ObjectDetector:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.camera = None
        self.audio_feedback = AudioFeedback()
        self.detection_data = []
        self.is_running = False

    def start_camera(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
            self.is_running = True

    def stop_camera(self):
        self.is_running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        # Clear speech queue when stopping
        self.audio_feedback.clear_queue()
        self.detection_data = [] 

    def calculate_clock_position(self, x, y, frame_width, frame_height):
        center_x = frame_width / 2
        center_y = frame_height / 2
        angle = math.atan2(y - center_y, x - center_x)
        clock_position = int(((angle + math.pi) * 6 / math.pi + 3) % 12)
        return 12 if clock_position == 0 else clock_position

    def calculate_distance(self, pixel_width, object_class):
        """
        Calculate the distance of an object using the triangle similarity formula
        Distance = (Known width × Focal length) / Pixel width
        """
        try:
            # Get the known width for the object class, default to 0.5 if not found
            known_width = KNOWN_WIDTH.get(object_class, 0.5)
            distance = (known_width * FOCAL_LENGTH) / pixel_width
            # Limit distance to reasonable range (0.1 to 20 meters)
            return max(0.1, min(20, distance))
        except ZeroDivisionError:
            return 0.0

    def process_frame(self):
        while self.is_running:
            if self.camera is None:
                continue

            ret, frame = self.camera.read()
            if not ret:
                continue

            results = self.model(frame, conf=0.5)
            detected_objects = []

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    label = self.model.names[cls]
                    
                    # Calculate distance using pixel width and object class
                    pixel_width = x2 - x1
                    distance = self.calculate_distance(pixel_width, label.lower())
                    
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    clock_pos = self.calculate_clock_position(
                        center_x, center_y, frame.shape[1], frame.shape[0]
                    )
                    
                    # Draw bounding box and label with distance
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_text = f"{label} {distance:.1f}m"
                    cv2.putText(
                        frame, 
                        label_text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, 
                        (0, 255, 0), 
                        2
                    )
                    
                    # Add detection data
                    detected_objects.append({
                        'label': label,
                        'position': clock_pos,
                        'distance': f"{distance:.1f}",
                        'bbox': {
                            'x': x1,
                            'y': y1,
                            'width': x2 - x1,
                            'height': y2 - y1
                        }
                    })

            self.detection_data = detected_objects
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_clock_reference():
    size = 400
    center = (size // 2, size // 2)
    radius = size // 3
    
    clock_img = np.zeros((size, size, 3), dtype=np.uint8)
    clock_img[:] = [42, 42, 42]  # Dark gray background
    
    # Draw clock circle
    cv2.circle(clock_img, center, radius, (0, 255, 0), 2)
    
    # Draw hour marks and numbers
    for hour in range(1, 13):
        angle = math.radians(30 * hour - 90)
        # Calculate points for hour marks
        start_pt = (
            int(center[0] + (radius - 20) * math.cos(angle)),
            int(center[1] + (radius - 20) * math.sin(angle))
        )
        
        # Add numbers
        text_pt = (
            int(center[0] + (radius - 40) * math.cos(angle)) - 10,
            int(center[1] + (radius - 40) * math.sin(angle)) + 10
        )
        cv2.putText(clock_img, str(hour), text_pt, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Convert to jpg
    _, buffer = cv2.imencode('.jpg', clock_img)
    return buffer.tobytes()

detector = ObjectDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    detector.start_camera()
    return Response(
        detector.process_frame(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/get_detections')
def get_detections():
    return jsonify(detector.detection_data)

@app.route('/start')
def start():
    detector.start_camera()
    return jsonify({"status": "success", "message": "Camera started"})

@app.route('/stop')
def stop():
    try:
        detector.stop_camera()
        return jsonify({
            "status": "success", 
            "message": "Camera and detection stopped"
        })
    except Exception as e:
        logger.error(f"Error stopping detection: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/clock_reference')
def clock_reference():
    return Response(generate_clock_reference(), mimetype='image/jpeg')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        user_id = request.form.get('user_id')  # Get user_id from request
        if not user_id:
            return jsonify({'error': 'User ID is required'}), 400

        file = request.files['image']
        nparr = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Failed to decode image")
            
        results = model(frame, conf=0.5)
        detected_objects = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                
                # Calculate distance using pixel width and object class
                pixel_width = x2 - x1
                distance = (KNOWN_WIDTH.get(label.lower(), 0.5) * FOCAL_LENGTH) / pixel_width
                distance = max(0.1, min(20, distance))  # Limit to reasonable range
                
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                clock_pos = calculate_clock_position(
                    center_x, center_y, frame.shape[1], frame.shape[0]
                )
                
                detected_objects.append({
                    'label': label,
                    'position': clock_pos,
                    'distance': f"{distance:.1f}",
                    'confidence': float(box.conf[0]),
                    'bbox': {
                        'x': x1,
                        'y': y1,
                        'width': x2 - x1,
                        'height': y2 - y1
                    }
                })
        
        # Save detections to database
        for obj in detected_objects:
            detection = Detection(
                user_id=user_id,
                object_label=obj['label'],
                position=obj['position'],
                distance=float(obj['distance'])
            )
            db.session.add(detection)
        
        db.session.commit()
        
        return jsonify(detected_objects)
    
    except Exception as e:
        logger.error(f"Error processing detection: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/user', methods=['POST'])
def create_user():
    try:
        data = request.get_json()
        name = data.get('name')
        if not name:
            return jsonify({'error': 'Name is required'}), 400
        
        user = User(name=name)
        db.session.add(user)
        db.session.commit()
        
        return jsonify({
            'id': user.id,
            'name': user.name,
            'created_at': user.created_at.isoformat()
        })
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def history():
    return render_template('history.html')


@app.route('/api/history')
def get_history():
    try:
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({'error': 'User ID required'}), 400
            
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        query = Detection.query.filter_by(user_id=user_id)
        
        ist_tz = pytz.timezone('Asia/Kolkata')
        
        if start_date:
            # Convert start_date to IST for comparison
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            start_dt = start_dt.astimezone(ist_tz)
            query = query.filter(Detection.detected_at >= start_dt)
            
        if end_date:
            # Convert end_date to IST for comparison
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            end_dt = end_dt.astimezone(ist_tz)
            query = query.filter(Detection.detected_at <= end_dt)
        
        detections = query.order_by(Detection.detected_at.desc()).all()
        
        return jsonify([{
            'id': d.id,
            'user_name': d.user.name,
            'object_label': d.object_label,
            'position': d.position,
            'distance': d.distance,
            'detected_at': d.detected_at.astimezone(ist_tz).isoformat()
        } for d in detections])
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/api/auth', methods=['POST'])
def auth():
    try:
        data = request.get_json()
        name = data.get('name')
        action = data.get('action')
        
        if not name:
            return jsonify({'error': 'Name is required'}), 400
        
        if action == 'login':
            user = User.query.filter_by(name=name).first()
            if not user:
                return jsonify({'error': 'User not found'}), 404
        else:  # register
            if User.query.filter_by(name=name).first():
                return jsonify({'error': 'Name already taken'}), 400
            user = User(name=name)
            db.session.add(user)
            db.session.commit()
        
        return jsonify({
            'id': user.id,
            'name': user.name,
            'created_at': user.created_at.isoformat()
        })
    except Exception as e:
        logger.error(f"Auth error: {e}")
        return jsonify({'error': str(e)}), 500

# Add this helper function outside of any class or route
def calculate_clock_position(x, y, frame_width, frame_height):
    """Calculate clock position (1-12) based on coordinates"""
    center_x = frame_width / 2
    center_y = frame_height / 2
    angle = math.atan2(y - center_y, x - center_x)
    clock_position = int(((angle + math.pi) * 6 / math.pi + 3) % 12)
    return 12 if clock_position == 0 else clock_position

if __name__ == '__main__':
    try:
        # Get port from Azure's environment variable or use 8000 as default
        port = int(os.environ.get('WEBSITES_PORT', 8000))
        app.run(debug=False,host='0.0.0.0', port=port)
    except Exception as e:
        print(f"Error starting application: {e}", file=sys.stderr)