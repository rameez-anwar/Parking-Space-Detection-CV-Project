#!/usr/bin/env python3
"""
Flask Web Application for Live Parking Space Detection
Professional parking management system with database storage
"""

from flask import Flask, render_template, jsonify, request, session, redirect, url_for, flash
import numpy as np
from functools import wraps
"""Heavy modules are imported lazily to avoid boot failures on Azure
   (e.g., OpenCV libGL issues)."""
cv2 = None  # will be imported lazily
torch = None  # will be imported lazily
YOLO = None  # will be imported lazily
requests = None  # will be imported lazily
Image = None  # will be imported lazily
np_frombuffer = None  # lazy reference helper
import io
import json
import time
import threading
import base64
import os
from database import ParkingDatabase

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'
app.config['PERMANENT_SESSION_LIFETIME'] = 86400  # 24 hours
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Database instance
db = ParkingDatabase()

# Admin credentials (hardcoded as requested)
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'pass'

def admin_required(f):
    """Decorator to require admin authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Detection system components
model = None
parking_lot_detectors = {}  # Dictionary to store detectors for each parking lot
detection_started = False
current_sessions = {}  # Dictionary to store session IDs for each parking lot
_live_captures = {}  # Dictionary to store captures for each parking lot
_mjpeg_streams = {}  # Dictionary to store MJPEG streams for each parking lot
latest_images = {}  # Dictionary to store latest images for each parking lot
latest_processed_images = {}  # Dictionary to store latest processed images for each parking lot
init_in_progress = False  # guard to prevent duplicate heavy init

def load_regions_from_file(file_path='regions_new.json'):
    """Load regions from JSON file into dict[name]->np.array(points)."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

        regions_dict = {}
        
        # Handle new format: list of {id, name, points:[[x,y], ...], ...}
        if isinstance(data, list):
            for item in data:
                name = str(item.get('name') or item.get('id'))
                pts = np.array(item.get('points', []), dtype=np.int32)
                if pts.size == 0:
                    continue
                regions_dict[name] = pts
        
        # Handle old format: dict with string keys and point arrays
        elif isinstance(data, dict):
            for key, pts in data.items():
                if isinstance(pts, list) and len(pts) > 0:
                    # Convert to numpy array and ensure proper format
                    try:
                        pts_array = np.array(pts, dtype=np.int32)
                        if pts_array.size > 0:
                            regions_dict[str(key)] = pts_array
                    except (ValueError, TypeError) as e:
                        print(f"Error converting points for region {key}: {e}")
                        continue

        print(f"Successfully loaded {len(regions_dict)} regions from {file_path}")
        for name, pts in regions_dict.items():
            print(f"  Region {name}: {len(pts)} points")
        return regions_dict
    except Exception as e:
        print(f"Error loading regions from {file_path}: {e}")
        return {}

 

class _MjpegStream:
    """Minimal MJPEG multipart/x-mixed-replace stream reader using requests."""
    def __init__(self, url: str):
        self.url = url
        self.session = None
        self.response = None
        self.boundary = None
        self.buffer = bytearray()
        self.last_ok = 0.0

    def open(self):
        global requests
        if requests is None:
            import importlib as _imp
            requests = _imp.import_module('requests')
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; ParkingSpace/1.0)'
        }
        self.session = requests.Session()
        self.response = self.session.get(self.url, headers=headers, stream=True, timeout=20)
        self.response.raise_for_status()
        ctype = self.response.headers.get('Content-Type', '')
        # Example: multipart/x-mixed-replace; boundary=--myboundary
        boundary = None
        if 'boundary=' in ctype:
            boundary = ctype.split('boundary=')[-1].strip()
        if boundary and not boundary.startswith('--'):
            boundary = '--' + boundary
        self.boundary = boundary.encode('utf-8') if boundary else b'--'
        self.buffer.clear()

    def close(self):
        try:
            if self.response is not None:
                self.response.close()
        except Exception:
            pass
        try:
            if self.session is not None:
                self.session.close()
        except Exception:
            pass
        self.session = None
        self.response = None

    def read_frame(self):
        """Read next JPEG frame from the stream and return as numpy BGR image or None."""
        global np_frombuffer, np
        if self.response is None:
            self.open()
        # lazily import np helper
        if np_frombuffer is None:
            try:
                np_frombuffer = np.frombuffer
            except Exception:
                pass
        # Read chunks until we find a full JPEG (FFD8 ... FFD9)
        for _ in range(100):  # limit work per call
            chunk = next(self.response.iter_content(chunk_size=4096), None)
            if not chunk:
                break
            self.buffer.extend(chunk)
            start = self.buffer.find(b'\xff\xd8')
            end = self.buffer.find(b'\xff\xd9')
            if start != -1 and end != -1 and end > start:
                jpeg = self.buffer[start:end+2]
                # trim buffer to after this frame
                del self.buffer[:end+2]
                try:
                    arr = np_frombuffer(jpeg, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        self.last_ok = time.time()
                        return img
                except Exception:
                    return None
        # If no full frame yet, allow caller to retry later
        return None


def fetch_live_image(url, capture_key=None):
    """Fetch a single frame as an image.

    Supports snapshot URLs and continuous streams like MJPG/RTSP by
    maintaining a persistent cv2.VideoCapture and returning one frame per call.
    """
    global _live_captures, _mjpeg_streams, cv2, requests, Image
    try:
        # Lazy import dependencies
        if cv2 is None:
            import importlib
            try:
                cv2 = importlib.import_module('cv2')
            except Exception as e:
                print(f"OpenCV import error: {e}")
                return None
        if requests is None:
            import importlib as _imp
            requests = _imp.import_module('requests')
        if Image is None:
            from PIL import Image as _PILImage
            Image = _PILImage
        
        # Use capture_key to maintain separate captures for different URLs
        if capture_key is None:
            capture_key = url
        
        # Treat mjpg/rtsp/http video streams as VideoCapture sources
        lower = url.lower()
        if any(x in lower for x in ['.mjpg', 'rtsp://', 'http://', 'https://']) and 'webcapture.jpg' not in lower:
            # First try OpenCV with FFMPEG backend where available
            if capture_key not in _live_captures or not _live_captures[capture_key].isOpened():
                try:
                    _live_captures[capture_key] = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                except Exception:
                    _live_captures[capture_key] = cv2.VideoCapture(url)
                try:
                    _live_captures[capture_key].set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
            
            if _live_captures[capture_key] is not None and _live_captures[capture_key].isOpened():
                ok, frame = _live_captures[capture_key].read()
                if ok and frame is not None:
                    return frame
            
            # If OpenCV failed or intermittent, fall back to a raw MJPEG HTTP reader
            if '.mjpg' in lower or 'mjpeg' in lower:
                if capture_key not in _mjpeg_streams:
                    _mjpeg_streams[capture_key] = _MjpegStream(url)
                    try:
                        _mjpeg_streams[capture_key].open()
                    except Exception as e:
                        print(f"MJPEG open error: {e}")
                        _mjpeg_streams[capture_key] = None
                
                if _mjpeg_streams[capture_key] is not None:
                    try:
                        frame = _mjpeg_streams[capture_key].read_frame()
                        if frame is not None:
                            return frame
                    except Exception as e:
                        print(f"MJPEG read error: {e}")
                        try:
                            _mjpeg_streams[capture_key].close()
                        except Exception:
                            pass
                        _mjpeg_streams[capture_key] = None
            
            # Reconnect OpenCV capture briefly before giving up this cycle
            try:
                if capture_key in _live_captures and _live_captures[capture_key] is not None:
                    _live_captures[capture_key].release()
            except Exception:
                pass
            time.sleep(0.2)
            try:
                _live_captures[capture_key] = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            except Exception:
                _live_captures[capture_key] = cv2.VideoCapture(url)
            try:
                _live_captures[capture_key].set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            ok, frame = _live_captures[capture_key].read() if _live_captures[capture_key] is not None else (False, None)
            if ok and frame is not None:
                return frame
            return None

        # Fallback to HTTP snapshot
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return opencv_image
    except Exception as e:
        print(f"Error fetching image: {e}")
        return None

def detect_vehicles_robust(frame, model):
    """
    Robust vehicle detection using multiple strategies
    """
    # Strategy 1: YOLO with vehicle classes only (current approach)
    result1 = model(frame, conf=0.15, classes=[2, 3, 7], imgsz=(1088, 1920))
    r1 = result1[0]
    
    # Strategy 2: YOLO with all classes but filter for vehicles
    result2 = model(frame, conf=0.1, imgsz=(1088, 1920))
    r2 = result2[0]
    
    # Strategy 3: YOLO with very low confidence for vehicles
    result3 = model(frame, conf=0.05, classes=[2, 3, 7], imgsz=(1088, 1920))
    r3 = result3[0]
    
    # Combine all detections
    all_boxes = []
    all_masks = []
    
    # Add detections from strategy 1
    if r1.boxes is not None and r1.masks is not None:
        for box, mask in zip(r1.boxes, r1.masks.data):
            all_boxes.append(box)
            all_masks.append(mask)
    
    # Add detections from strategy 2 (filter for vehicles)
    if r2.boxes is not None and r2.masks is not None:
        for box, mask in zip(r2.boxes, r2.masks.data):
            class_id = int(box.cls[0].cpu().numpy())
            if class_id in [2, 3, 7]:  # car, motorcycle, truck
                all_boxes.append(box)
                all_masks.append(mask)
    
    # Add detections from strategy 3
    if r3.boxes is not None and r3.masks is not None:
        for box, mask in zip(r3.boxes, r3.masks.data):
            all_boxes.append(box)
            all_masks.append(mask)
    
    # Remove duplicate detections (same area)
    unique_boxes = []
    unique_masks = []
    
    for i, (box, mask) in enumerate(zip(all_boxes, all_masks)):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        confidence = float(box.conf[0].cpu().numpy())
        
        # Check if this detection overlaps significantly with existing ones
        is_duplicate = False
        for existing_box in unique_boxes:
            ex1, ey1, ex2, ey2 = existing_box.xyxy[0].cpu().numpy().astype(int)
            
            # Calculate overlap
            overlap_x1 = max(x1, ex1)
            overlap_y1 = max(y1, ey1)
            overlap_x2 = min(x2, ex2)
            overlap_y2 = min(y2, ey2)
            
            if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                box1_area = (x2 - x1) * (y2 - y1)
                box2_area = (ex2 - ex1) * (ey2 - ey1)
                
                # If overlap is more than 50% of either box, consider it duplicate
                if overlap_area > 0.5 * min(box1_area, box2_area):
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            unique_boxes.append(box)
            unique_masks.append(mask)
    
    return unique_boxes, unique_masks

def process_frame_improved(frame, regions, model):
    """
    Improved frame processing with robust vehicle detection
    """
    
    # Get robust vehicle detections
    vehicle_boxes, vehicle_masks = detect_vehicles_robust(frame, model)
    
    # Create combined vehicle mask
    vehicle_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    
    if vehicle_masks:
        for mask in vehicle_masks:
            binary_mask = mask.cpu().numpy().astype(np.uint8)
            if binary_mask.shape != (frame.shape[0], frame.shape[1]):
                binary_mask = cv2.resize(binary_mask, (frame.shape[1], frame.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
            vehicle_mask = cv2.bitwise_or(vehicle_mask, binary_mask)
    
    # Process each region
    labeled_image = frame.copy()
    total_empty_spaces = 0
    region_counts = {}
    
    for region_name, region in regions.items():
        # Create a mask for this specific region
        region_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        cv2.fillPoly(region_mask, [region], 255)
        
        # Count total pixels in region
        total_region_pixels = cv2.countNonZero(region_mask)
        
        # Check vehicle coverage in this region
        region_vehicle_mask = cv2.bitwise_and(vehicle_mask, region_mask)
        covered_pixels = cv2.countNonZero(region_vehicle_mask)
        
        # Calculate coverage percentage
        if total_region_pixels > 0:
            coverage_percentage = (covered_pixels / total_region_pixels) * 100
        else:
            coverage_percentage = 0
        
        # Determine if region is occupied (30% or more covered by vehicles)
        is_occupied = coverage_percentage >= 30
        
        # Visualize region
        if is_occupied:
            # Red for occupied regions
            boundary_color = (0, 0, 255)
            fill_color = (0, 0, 255)
            alpha = 0.3
            status = "Occupied"
        else:
            # Green for empty regions
            boundary_color = (0, 255, 0)
            fill_color = (0, 255, 0)
            alpha = 0.3
            status = "Empty"
        
        # Draw filled region with transparency
        overlay = labeled_image.copy()
        cv2.fillPoly(overlay, [region], fill_color)
        labeled_image = cv2.addWeighted(labeled_image, 1-alpha, overlay, alpha, 0)
        
        # Draw region boundary
        cv2.polylines(labeled_image, [region], True, boundary_color, 2)
        
        # Add label with coverage percentage
        M = cv2.moments(region)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            
            # Add region name and status
            label = f"{region_name}: {status}"
            cv2.putText(labeled_image, label, (center_x - 50, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, boundary_color, 2)
            
            # Add coverage percentage
            coverage_label = f"{coverage_percentage:.1f}% covered"
            cv2.putText(labeled_image, coverage_label, (center_x - 50, center_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, boundary_color, 1)
        
        # Count empty spaces
        if not is_occupied:
            total_empty_spaces += 1
        
        region_counts[region_name] = 0 if is_occupied else 1  # 0 = occupied, 1 = empty
    
    return labeled_image, total_empty_spaces, region_counts

 

def detection_loop_for_lot(parking_lot_id, camera_url, regions_file):
    """Background detection loop for a specific parking lot"""
    global model, parking_lot_detectors, current_sessions, latest_images, latest_processed_images
    
    print(f"Detection loop started for parking lot {parking_lot_id}")
    
    # Load regions for this parking lot
    regions = {}
    if regions_file and os.path.exists(regions_file):
        try:
            regions = load_regions_from_file(regions_file)
            print(f"Loaded {len(regions)} regions for parking lot {parking_lot_id}")
        except Exception as e:
            print(f"Error loading regions for parking lot {parking_lot_id}: {e}")
    else:
        print(f"No regions file configured for parking lot {parking_lot_id}: {regions_file}")
    
    # Start new session for this parking lot
    session_id = db.start_session(parking_lot_id)
    current_sessions[parking_lot_id] = session_id
    print(f"Started detection session {session_id} for parking lot {parking_lot_id}")
    
    frame_count = 0
    while True:
        try:
            frame_count += 1
            print(f"Parking lot {parking_lot_id} - Detection iteration {frame_count}")
            
            # Fetch live image
            frame = fetch_live_image(camera_url, f"lot_{parking_lot_id}")
            if frame is None:
                print(f"Failed to fetch frame for parking lot {parking_lot_id}, retrying...")
                time.sleep(5.0)
                continue
            
            print(f"Frame fetched for parking lot {parking_lot_id}, shape: {frame.shape}")
            
            # Convert image to base64 for live serving (in-memory)
            _, buffer = cv2.imencode('.jpg', frame)
            latest_images[parking_lot_id] = base64.b64encode(buffer).decode('utf-8')
            
            # Process frame with improved detection (robust to errors)
            try:
                if regions and model:
                    processed_frame, total_spaces, region_counts = process_frame_improved(
                        frame, regions, model
                    )
                    print(f"Frame processed for parking lot {parking_lot_id} - {len(region_counts)} regions detected")
                    if len(region_counts) == 0:
                        print(f"Warning: No regions detected for parking lot {parking_lot_id}")
                        print(f"Available regions: {list(regions.keys())}")
                else:
                    processed_frame = frame.copy()
                    if not regions:
                        cv2.putText(processed_frame, f'No regions configured for lot {parking_lot_id}', (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        print(f"No regions loaded for parking lot {parking_lot_id} (file: {regions_file})")
                    if not model:
                        cv2.putText(processed_frame, f'Model not loaded for lot {parking_lot_id}', (20, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        print(f"Model not loaded for parking lot {parking_lot_id}")
                    total_spaces = 0
                    region_counts = {}
            except Exception as e:
                # Graceful fallback so UI continues updating
                print(f"Processing error for parking lot {parking_lot_id}: {e}")
                processed_frame = frame.copy()
                cv2.putText(processed_frame, 'Processing error', (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                total_spaces = 0
                region_counts = {}
            
            # Convert processed image to base64 for live serving (in-memory)
            _, processed_buffer = cv2.imencode('.jpg', processed_frame)
            latest_processed_images[parking_lot_id] = base64.b64encode(processed_buffer).decode('utf-8')
            
            # Calculate empty and filled spaces from region_counts
            empty_spaces = sum(1 for status in region_counts.values() if status > 0)
            filled_spaces = len(region_counts) - empty_spaces
            total_spaces_count = len(region_counts) if region_counts else 0
            
            # Save statistics to database (omit images for real-time serving)
            db.save_detection_data(
                parking_lot_id=parking_lot_id,
                session_id=session_id,
                total_spaces=total_spaces_count,
                empty_spaces=empty_spaces,
                filled_spaces=filled_spaces,
                region_data=region_counts,
                image_data=None,
                processed_image_data=None
            )
            
            print(f"Parking lot {parking_lot_id} - Saved stats: {total_spaces_count} total, {empty_spaces} empty, {filled_spaces} filled")
            
            # Wait before next frame (15 seconds per requirement)
            print(f"Parking lot {parking_lot_id} - Waiting 15 seconds for next cycle...")
            for _ in range(15):
                time.sleep(1.0)
            
        except Exception as e:
            print(f"Detection error for parking lot {parking_lot_id}: {e}")
            time.sleep(5.0)

def init_detection_system():
    """Initialize the detection system for multiple parking lots"""
    global model, detection_started, torch, YOLO, cv2, init_in_progress
    
    try:
        if detection_started or init_in_progress:
            return
        init_in_progress = True
        print("Starting multi-parking lot detection system initialization...")
        
        # Lazy import heavy deps
        if torch is None:
            import importlib
            torch = importlib.import_module('torch')
        if YOLO is None:
            from ultralytics import YOLO as _YOLO
            YOLO = _YOLO
        if cv2 is None:
            import importlib as _imp
            try:
                cv2 = _imp.import_module('cv2')
            except Exception as e:
                print(f"OpenCV import error during init: {e}")
                # Continue without cv2; endpoints will handle gracefully
        
        # Set CUDA benchmarking for performance (safe on CPU too)
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        
        device = 'cuda' if hasattr(torch, 'cuda') and torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Initialize YOLO model with robust weight selection
        print("Loading YOLO model...")
        weights_path = os.getenv('YOLO_WEIGHTS_PATH')
        weights_url = os.getenv('YOLO_WEIGHTS_URL')
        try:
            # Prefer explicit path if provided
            if weights_path and os.path.exists(weights_path):
                print(f"Loading YOLO weights from path: {weights_path}")
                model = YOLO(weights_path)
            else:
                # If a URL is provided and file is missing, download to a default path
                if weights_url and weights_path:
                    try:
                        from pathlib import Path
                        import requests as _requests
                        dest = Path(weights_path)
                        if not dest.exists():
                            print(f"Downloading YOLO weights from URL to {dest}...")
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            with _requests.get(weights_url, stream=True, timeout=120) as r:
                                r.raise_for_status()
                                with open(dest, 'wb') as f:
                                    for chunk in r.iter_content(1024 * 1024):
                                        if chunk:
                                            f.write(chunk)
                    except Exception as de:
                        print(f"Weights download failed, continuing with built-in model name: {de}")
                # Fallback to built-in model name available via ultralytics hub/cache
                model_name = os.getenv('YOLO_MODEL_NAME', 'yolov8n-seg.pt')
                print(f"Loading YOLO model by name: {model_name}")
                model = YOLO(model_name)
            model.to(device)
            print("YOLO model loaded successfully")
        except Exception as me:
            print(f"Failed to load YOLO model: {me}")
            raise
        
        print("Detection system initialized successfully")
        
        # Start detection loops for all active parking lots
        if not detection_started:
            print("Starting detection loops for all parking lots...")
            parking_lots = db.get_parking_lots()
            
            for lot in parking_lots:
                if lot['is_active']:
                    print(f"Starting detection for parking lot: {lot['name']}")
                    detection_thread = threading.Thread(
                        target=detection_loop_for_lot, 
                        args=(lot['id'], lot['camera_url'], lot['regions_file']),
                        daemon=True
                    )
                    detection_thread.start()
                    print(f"Detection thread started for parking lot {lot['id']}")
            
            detection_started = True
            print(f"Detection loops started for {len([l for l in parking_lots if l['is_active']])} parking lots")
        
        # Wait a moment to ensure the threads start
        time.sleep(2)
        
    except Exception as e:
        print(f"Initialization error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        init_in_progress = False

@app.before_request
def _ensure_detection_started():
    """Ensure detection starts; runs once guarded by flag on first incoming request.

    Skip for health endpoints so Azure health probes do not block on heavy initialization.
    """
    global detection_started
    # Do not trigger initialization for lightweight health endpoints
    if request.path in ("/healthz", "/health", "/status"):
        return
    if not detection_started and not init_in_progress:
        try:
            threading.Thread(target=init_detection_system, daemon=True).start()
        except Exception as e:
            print(f"Failed to spawn detection init: {e}")

@app.route('/')
def index():
    """Redirect to login page"""
    return redirect(url_for('login'))

@app.route('/login')
def login():
    """Login page"""
    return render_template('main_login.html')

@app.route('/dashboard')
@admin_required
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/session-check')
def session_check():
    """Check session status for debugging"""
    return jsonify({
        'logged_in': session.get('admin_logged_in', False),
        'session_id': session.get('_id', 'no-id'),
        'permanent': session.permanent
    })

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Admin login page"""
    if request.method == 'POST':
        # Handle JSON requests (from new login page)
        if request.is_json:
            data = request.json
            username = data.get('username')
            password = data.get('password')
            
            if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                session['admin_logged_in'] = True
                session.permanent = True
                return jsonify({'success': True, 'redirect': '/dashboard'})
            else:
                return jsonify({'error': 'Invalid credentials'}), 401
        
        # Handle form requests (from old admin login page)
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            return redirect(url_for('admin_panel'))
        else:
            flash('Invalid credentials', 'error')
    
    return render_template('admin_login.html')

@app.route('/admin/logout')
def admin_logout():
    """Admin logout"""
    session.clear()  # Clear all session data
    return redirect(url_for('login'))

@app.route('/admin')
@admin_required
def admin_panel():
    """Admin panel main page"""
    parking_lots = db.get_parking_lots()
    return render_template('admin_panel.html', parking_lots=parking_lots)

@app.route('/admin/calibration/<int:lot_id>')
@admin_required
def admin_calibration(lot_id):
    """Calibration tool for specific parking lot"""
    parking_lot = None
    parking_lots = db.get_parking_lots()
    for lot in parking_lots:
        if lot['id'] == lot_id:
            parking_lot = lot
            break
    
    if not parking_lot:
        flash('Parking lot not found', 'error')
        return redirect(url_for('admin_panel'))
    
    return render_template('admin_calibration.html', parking_lot=parking_lot)

@app.route('/api/parking-lots', methods=['GET'])
def api_parking_lots():
    """Public API for getting parking lots list"""
    return jsonify(db.get_parking_lots())

@app.route('/admin/api/parking-lots', methods=['GET', 'POST'])
@admin_required
def admin_api_parking_lots():
    """API for managing parking lots"""
    if request.method == 'GET':
        return jsonify(db.get_parking_lots())
    
    elif request.method == 'POST':
        data = request.json
        name = data.get('name')
        camera_url = data.get('camera_url')
        regions_file = data.get('regions_file')
        
        if not name or not camera_url:
            return jsonify({'error': 'Name and camera URL are required'}), 400
        
        try:
            lot_id = db.add_parking_lot(name, camera_url, regions_file)
            return jsonify({'id': lot_id, 'message': 'Parking lot added successfully'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/admin/api/parking-lots/<int:lot_id>', methods=['PUT', 'DELETE'])
@admin_required
def admin_api_parking_lot(lot_id):
    """API for updating/deleting specific parking lot"""
    if request.method == 'PUT':
        data = request.json
        try:
            db.update_parking_lot(
                lot_id,
                name=data.get('name'),
                camera_url=data.get('camera_url'),
                regions_file=data.get('regions_file'),
                is_active=data.get('is_active')
            )
            return jsonify({'message': 'Parking lot updated successfully'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    elif request.method == 'DELETE':
        try:
            db.delete_parking_lot(lot_id)
            return jsonify({'message': 'Parking lot deleted successfully'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/admin/api/save-regions/<int:lot_id>', methods=['POST'])
@admin_required
def admin_api_save_regions(lot_id):
    """API for saving regions for a parking lot"""
    try:
        data = request.json
        regions_data = data.get('regions')
        
        if not regions_data:
            return jsonify({'error': 'No regions data provided'}), 400
        
        # Get parking lot info
        parking_lots = db.get_parking_lots()
        parking_lot = None
        for lot in parking_lots:
            if lot['id'] == lot_id:
                parking_lot = lot
                break
        
        if not parking_lot:
            return jsonify({'error': 'Parking lot not found'}), 404
        
        # Save regions to file
        regions_file = parking_lot.get('regions_file', f'regions_lot_{lot_id}.json')
        print(f"Saving regions to file: {regions_file}")
        print(f"Regions data: {regions_data}")
        
        with open(regions_file, 'w') as f:
            import json
            json.dump(regions_data, f, indent=2)
        
        print(f"Regions saved successfully to {regions_file}")
        
        # Update parking lot with regions file
        db.update_parking_lot(lot_id, regions_file=regions_file)
        
        # Restart detection for this parking lot
        restart_detection_for_lot(lot_id)
        
        return jsonify({'message': 'Regions saved successfully', 'file': regions_file})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/api/capture-frame/<int:lot_id>')
@admin_required
def admin_api_capture_frame(lot_id):
    """API for capturing a frame from a specific parking lot camera"""
    try:
        # Get parking lot info
        parking_lots = db.get_parking_lots()
        parking_lot = None
        for lot in parking_lots:
            if lot['id'] == lot_id:
                parking_lot = lot
                break
        
        if not parking_lot:
            return jsonify({'error': 'Parking lot not found'}), 404
        
        # Fetch frame from camera
        frame = fetch_live_image(parking_lot['camera_url'], f"calibration_{lot_id}")
        
        if frame is None:
            return jsonify({'error': 'Failed to capture frame from camera'}), 500
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({'image': frame_b64, 'success': True})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def restart_detection_for_lot(lot_id):
    """Restart detection for a specific parking lot"""
    try:
        parking_lots = db.get_parking_lots()
        parking_lot = None
        for lot in parking_lots:
            if lot['id'] == lot_id:
                parking_lot = lot
                break
        
        if parking_lot and parking_lot['is_active']:
            print(f"Restarting detection for parking lot {lot_id} with regions file: {parking_lot['regions_file']}")
            
            # End current session if exists
            if lot_id in current_sessions:
                try:
                    db.end_session(current_sessions[lot_id])
                except:
                    pass
            
            # Start new detection thread
            detection_thread = threading.Thread(
                target=detection_loop_for_lot, 
                args=(lot_id, parking_lot['camera_url'], parking_lot['regions_file']),
                daemon=True
            )
            detection_thread.start()
            print(f"Detection restarted for parking lot {lot_id}")
    except Exception as e:
        print(f"Error restarting detection for lot {lot_id}: {e}")

@app.route('/healthz')
def healthz():
    """Lightweight health check endpoint for Azure App Service."""
    return jsonify({"status": "ok"}), 200

@app.route('/api/statistics')
def get_statistics():
    """API endpoint to get parking statistics from database"""
    # Get parking lot ID from query parameter
    lot_id = request.args.get('lot_id')
    
    if lot_id and lot_id != 'all':
        # Get statistics for specific parking lot
        try:
            lot_id = int(lot_id)
            all_stats = db.get_all_latest_statistics()
            print(f"API: Requested statistics for lot {lot_id}")
            print(f"API: Available lots: {[lot['id'] for lot in all_stats]}")
            
            specific_lot = None
            for lot in all_stats:
                if lot['id'] == lot_id:
                    specific_lot = lot
                    break
            
            if specific_lot:
                print(f"API: Found lot {lot_id}: {specific_lot['name']}")
                print(f"API: Total spaces: {specific_lot['total_spaces']}, Empty: {specific_lot['empty_spaces']}")
                print(f"API: Region breakdown: {specific_lot['region_breakdown']}")
                
                # Clean up region names to be simple zone numbers
                cleaned_regions = {}
                zone_counter = 1
                for region_name, status in specific_lot['region_breakdown'].items():
                    cleaned_regions[f"Zone {zone_counter}"] = status
                    zone_counter += 1
                
                return jsonify({
                    'total_spaces': specific_lot['total_spaces'],
                    'empty_spaces': specific_lot['empty_spaces'],
                    'filled_spaces': specific_lot['filled_spaces'],
                    'region_breakdown': cleaned_regions,
                    'last_update': specific_lot['last_update'],
                    'parking_lot_name': specific_lot['name']
                })
            else:
                print(f"API: No data found for lot {lot_id}")
                return jsonify({
                    'total_spaces': 0,
                    'empty_spaces': 0,
                    'filled_spaces': 0,
                    'region_breakdown': {},
                    'last_update': 'No data available',
                    'parking_lot_name': 'Unknown'
                })
        except ValueError:
            pass  # Fall through to all parking lots
    
    # Get statistics for all parking lots
    all_stats = db.get_all_latest_statistics()
    
    if all_stats:
        # Calculate totals across all parking lots
        total_spaces = sum(lot['total_spaces'] for lot in all_stats)
        total_empty = sum(lot['empty_spaces'] for lot in all_stats)
        total_filled = sum(lot['filled_spaces'] for lot in all_stats)
        
        # Combine region breakdowns with simple zone names
        combined_regions = {}
        for lot in all_stats:
            lot_name = lot['name']
            zone_counter = 1
            for region, status in lot['region_breakdown'].items():
                combined_regions[f"{lot_name} - Zone {zone_counter}"] = status
                zone_counter += 1
        
        return jsonify({
            'total_spaces': total_spaces,
            'empty_spaces': total_empty,
            'filled_spaces': total_filled,
            'region_breakdown': combined_regions,
            'last_update': max(lot['last_update'] for lot in all_stats if lot['last_update'] != 'No data'),
            'parking_lots': all_stats
        })
    
    return jsonify({
        'total_spaces': 0,
        'empty_spaces': 0,
        'filled_spaces': 0,
        'region_breakdown': {},
        'last_update': 'No data available',
        'parking_lots': []
    })

@app.route('/api/live-image')
def get_live_image():
    """API endpoint to get the current live image from memory (real-time)."""
    # Return the first available image from any parking lot
    for lot_id, image_b64 in latest_images.items():
        if image_b64:
            return jsonify({'image': image_b64, 'parking_lot_id': lot_id})
    return jsonify({'error': 'No image available'})

@app.route('/api/processed-image')
def get_processed_image():
    """API endpoint to get the processed image with detection results from memory (real-time)."""
    # Return the first available processed image from any parking lot
    for lot_id, image_b64 in latest_processed_images.items():
        if image_b64:
            return jsonify({'image': image_b64, 'parking_lot_id': lot_id})
    # Fallback: if processed not ready yet, serve the live frame so UI still updates
    for lot_id, image_b64 in latest_images.items():
        if image_b64:
            return jsonify({'image': image_b64, 'parking_lot_id': lot_id})
    return jsonify({'error': 'No processed image available'})

@app.route('/api/live-image/<int:lot_id>')
def get_live_image_for_lot(lot_id):
    """API endpoint to get the current live image for a specific parking lot."""
    if lot_id in latest_images and latest_images[lot_id]:
        return jsonify({'image': latest_images[lot_id], 'parking_lot_id': lot_id})
    return jsonify({'error': f'No image available for parking lot {lot_id}'})

@app.route('/api/processed-image/<int:lot_id>')
def get_processed_image_for_lot(lot_id):
    """API endpoint to get the processed image for a specific parking lot."""
    if lot_id in latest_processed_images and latest_processed_images[lot_id]:
        return jsonify({'image': latest_processed_images[lot_id], 'parking_lot_id': lot_id})
    # Fallback to live image
    if lot_id in latest_images and latest_images[lot_id]:
        return jsonify({'image': latest_images[lot_id], 'parking_lot_id': lot_id})
    return jsonify({'error': f'No processed image available for parking lot {lot_id}'})

@app.route('/api/history')
def get_history():
    """API endpoint to get historical statistics"""
    limit = request.args.get('limit', 50, type=int)
    history = db.get_statistics_history(limit)
    return jsonify(history)

@app.route('/api/session-summary')
def get_session_summary():
    """API endpoint to get current session summary"""
    if current_session_id:
        summary = db.get_session_summary(current_session_id)
        return jsonify(summary or {})
    return jsonify({'error': 'No active session'})

@app.route('/api/detection-config', methods=['GET', 'POST'])
def detection_config():
    """API endpoint to get or update detection configuration"""
    if request.method == 'POST':
        config = request.json
        if 'overlap_threshold' in config:
            db.update_system_config('overlap_threshold', str(config['overlap_threshold']))
        if 'min_vehicle_size' in config:
            db.update_system_config('min_vehicle_size', str(config['min_vehicle_size']))
        return jsonify({'status': 'updated'})
    
    # GET request - return current config
    overlap_threshold = db.get_system_config('overlap_threshold') or '15'
    min_vehicle_size = db.get_system_config('min_vehicle_size') or '500'
    
    return jsonify({
        'overlap_threshold': float(overlap_threshold),
        'min_vehicle_size': int(min_vehicle_size)
    })

@app.route('/api/test')
def test_detection():
    """Test endpoint to check if detection system is working"""
    all_stats = db.get_all_latest_statistics()
    parking_lots = db.get_parking_lots()
    return jsonify({
        'model_loaded': model is not None,
        'database_connected': True,
        'parking_lots_count': len(parking_lots),
        'active_parking_lots': len([l for l in parking_lots if l['is_active']]),
        'detection_started': detection_started,
        'current_sessions': current_sessions,
        'latest_data_available': len(all_stats) > 0,
        'latest_images_count': len(latest_images),
        'latest_processed_images_count': len(latest_processed_images)
    })

if __name__ == '__main__':
    # Initialize detection system
    init_detection_system()
    
    # Wait a moment for initialization to complete
    time.sleep(2)
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
