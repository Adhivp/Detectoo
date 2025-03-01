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
load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Global variables
MODEL_PATH = os.getenv('MODEL_PATH', 'yolo11m.pt')
KNOWN_WIDTH = float(os.getenv('KNOWN_WIDTH', 0.6))
FOCAL_LENGTH = int(os.getenv('FOCAL_LENGTH', 1000))

# Initialize YOLO model globally
try:
    model = YOLO(MODEL_PATH)
    logger.info(f"YOLO model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading YOLO model: {e}")
    sys.exit(1)

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

    def calculate_distance(self, pixel_width):
        return (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width

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
                    
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    clock_pos = self.calculate_clock_position(
                        center_x, center_y, frame.shape[1], frame.shape[0]
                    )
                    distance = self.calculate_distance(x2 - x1)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame, 
                        f"{label} {distance:.1f}m", 
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, 
                        (0, 255, 0), 
                        2
                    )
                    
                    detected_objects.append({
                        'label': label,
                        'position': clock_pos,
                        'distance': f"{distance:.1f}"
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
                
                # Calculate center and position
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Calculate clock position
                frame_height, frame_width = frame.shape[:2]
                clock_pos = calculate_clock_position(
                    center_x, center_y, frame_width, frame_height
                )
                
                # Calculate distance
                distance = (KNOWN_WIDTH * FOCAL_LENGTH) / (x2 - x1)
                
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
        
        return jsonify(detected_objects)
    
    except Exception as e:
        logger.error(f"Error processing detection: {e}", exc_info=True)
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