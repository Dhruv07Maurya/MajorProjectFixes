from flask import Flask, render_template, Response, jsonify, request
import cv2
import threading
import json
import sqlite3
from datetime import datetime, timedelta
import time
import os
import numpy as np
import base64

# --- WEED DETECTION INTEGRATION ---
# Imported from the self-contained weed_detection package (sourced from WeedIoTNew).
# This adds weed analysis capabilities to the camera module.
from weed_detection import run_weed_detection

# --- CONFIGURATION ---
STATE_FILE = "state.json"
DB_FILE = "greenhouse.db"
PLANT_PROFILES_FILE = "plant_profiles.json"
PLACEHOLDER_IMAGE = "placeholder.jpg"

app = Flask(__name__)

# --- CAMERA SETUP ---
camera_lock = threading.Lock()
camera_thread = None
output_frame = None
streaming = False

def video_stream_thread():
    """Reads frames from the camera as long as the streaming flag is set."""
    global output_frame, streaming
    
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Camera Error: Could not open video stream.")
        streaming = False
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(1)
    print("Camera thread started successfully.")

    while streaming:
        success, frame = camera.read()
        if success:
            _, buffer = cv2.imencode('.jpg', frame)
            with camera_lock:
                output_frame = buffer.tobytes()
        time.sleep(0.05)

    camera.release()
    with camera_lock:
        output_frame = None
    print("Camera thread stopped and camera released.")

def generate_frames():
    """Generator function to stream frames to the client."""
    while streaming:
        with camera_lock:
            if output_frame is None:
                with open(PLACEHOLDER_IMAGE, "rb") as f: frame = f.read()
            else:
                frame = output_frame
        
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.05)


# --- HELPER FUNCTIONS ---
def load_json_file(filename, default_data={}):
    try:
        with open(filename, 'r') as f: return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError): return default_data

def save_json_file(filename, data):
    with open(filename, 'w') as f: json.dump(data, f, indent=4)

def query_db(query, args=(), one=False):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(query, args)
    rv = cur.fetchall()
    conn.close()
    return (rv[0] if rv else None) if one else rv

def log_to_db(table, data_dict):
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        if table == 'action_logs':
            cursor.execute('INSERT INTO action_logs (timestamp, source, action) VALUES (?, ?, ?)',
                           (timestamp, data_dict.get('source'), data_dict.get('action')))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"WebApp: Database log error: {e}")

def format_timestamps(rows):
    results = []
    for row in rows:
        row_dict = dict(row)
        if 'timestamp' in row_dict and row_dict['timestamp']:
            try:
                ts = datetime.strptime(row_dict['timestamp'], '%Y-%m-%d %H:%M:%S')
                row_dict['timestamp'] = ts.isoformat() + "Z"
            except (ValueError, TypeError): pass
        results.append(row_dict)
    return results

# --- MAIN ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

# --- API ENDPOINTS ---

@app.route('/api/status')
def get_status():
    state = load_json_file(STATE_FILE, {})
    profiles = load_json_file(PLANT_PROFILES_FILE, {})
    return jsonify({
        'live_data': state.get('live_data', {}),
        'actuator_states': state.get('live_actuator_states', {}),
        'system_mode': state.get('system_mode', 'AUTO'),
        'active_profile_name': state.get('active_profile_name'),
        'plant_profiles': profiles
    })

@app.route('/api/historical-data')
def get_historical_data():
    hours = request.args.get('hours', 24, type=int)
    time_threshold = datetime.utcnow() - timedelta(hours=hours)
    readings = query_db( 'SELECT timestamp, temperature, humidity, soil_moisture, light_intensity FROM sensor_readings WHERE timestamp >= ? ORDER BY timestamp ASC', [time_threshold] )
    return jsonify({'readings': format_timestamps(readings)})
    
@app.route('/api/logs')
def get_logs():
    logs = query_db('SELECT timestamp, source, action FROM action_logs ORDER BY id DESC LIMIT 50')
    return jsonify(format_timestamps(logs))

# --- PROFILE MANAGEMENT API ---
@app.route('/api/profiles', methods=['GET', 'POST'])
def handle_profiles():
    profiles = load_json_file(PLANT_PROFILES_FILE, {})
    if request.method == 'POST':
        data = request.json
        name = data.pop("profile_name", None)
        if not name: return jsonify({'success': False, 'message': 'Profile name is required.'}), 400
        if name in profiles: return jsonify({'success': False, 'message': f'Profile "{name}" already exists.'}), 409
        profiles[name] = data
        save_json_file(PLANT_PROFILES_FILE, profiles)
        log_to_db('action_logs', {'source': 'WEB UI', 'action': f"Created profile: {name}"})
        return jsonify({'success': True, 'message': f'Profile "{name}" created.'}), 201
    return jsonify(profiles)

@app.route('/api/profiles/<profile_name>', methods=['GET', 'PUT', 'DELETE'])
def handle_single_profile(profile_name):
    profiles = load_json_file(PLANT_PROFILES_FILE, {})
    if profile_name not in profiles: return jsonify({'success': False, 'message': 'Profile not found.'}), 404
    
    if request.method == 'GET': return jsonify(profiles[profile_name])
    
    if request.method == 'PUT':
        profiles[profile_name] = request.json
        save_json_file(PLANT_PROFILES_FILE, profiles)
        log_to_db('action_logs', {'source': 'WEB UI', 'action': f"Updated profile: {profile_name}"})
        return jsonify({'success': True, 'message': f'Profile "{profile_name}" updated.'})

    if request.method == 'DELETE':
        del profiles[profile_name]
        state = load_json_file(STATE_FILE, {})
        if state.get('active_profile_name') == profile_name:
            state['active_profile_name'] = None
            save_json_file(STATE_FILE, state)
        save_json_file(PLANT_PROFILES_FILE, profiles)
        log_to_db('action_logs', {'source': 'WEB UI', 'action': f"Deleted profile: {profile_name}"})
        return jsonify({'success': True, 'message': f'Profile "{profile_name}" deleted.'})

@app.route('/api/active-profile', methods=['POST'])
def set_active_profile():
    data = request.json
    state = load_json_file(STATE_FILE, {})
    profile_name = data.get('active_profile_name')
    state['active_profile_name'] = profile_name if profile_name != 'none' else None
    save_json_file(STATE_FILE, state)
    log_to_db('action_logs', {'source': 'WEB UI', 'action': f"Active profile set to: {state['active_profile_name']}"})
    return jsonify({'success': True})

# --- ACTUATOR & CAMERA API ---
@app.route('/api/actuators/mode', methods=['POST'])
def set_system_mode():
    data = request.json
    state = load_json_file(STATE_FILE, {})
    state['system_mode'] = data.get('mode', 'AUTO')
    save_json_file(STATE_FILE, state)
    log_to_db('action_logs', {'source': 'WEB UI', 'action': f"System mode set to {state['system_mode']}."})
    return jsonify({'success': True})
    
@app.route('/api/actuators/<actuator_name>', methods=['POST'])
def set_manual_actuator(actuator_name):
    data = request.json
    state = load_json_file(STATE_FILE, {})
    if state.get('system_mode') != 'MANUAL': return jsonify({'success': False, 'message': 'System must be in MANUAL mode.'}), 400
    state[f"manual_{actuator_name}"] = data.get('state', False)
    save_json_file(STATE_FILE, state)
    log_to_db('action_logs', {'source': 'WEB UI', 'action': f"Manual override: {actuator_name} set to {data.get('state')}."})
    return jsonify({'success': True})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera/start', methods=['POST'])
def start_stream():
    global camera_thread, streaming
    with camera_lock:
        if not streaming:
            streaming = True
            camera_thread = threading.Thread(target=video_stream_thread)
            camera_thread.start()
            print("Camera stream requested by client.")
    return jsonify({'success': True})

@app.route('/camera/stop', methods=['POST'])
def stop_stream():
    global camera_thread, streaming
    with camera_lock:
        if streaming: streaming = False
    if camera_thread: camera_thread.join()
    print("Camera stream stopped by client.")
    return jsonify({'success': True})

# ============================================================
# WEED DETECTION INTEGRATION — API ENDPOINT
# ============================================================
# This endpoint captures the current camera frame and runs
# weed detection analysis from the WeedIoTNew model.
# It does NOT modify the camera streaming logic in any way.
# ============================================================

@app.route('/api/weed-detect', methods=['POST'])
def weed_detect():
    """
    Capture current camera frame and run weed detection.
    
    Expects JSON body (all optional):
        - method: str ('color', 'ndvi', 'texture', 'size_filter') — default 'color'
        - threshold: float (0.0-1.0) — default 0.12
    
    Returns JSON with detection results including annotated images as base64.
    """
    try:
        # 1. Grab the current frame from the camera stream
        with camera_lock:
            current_frame = output_frame
        
        if current_frame is None:
            return jsonify({
                'success': False,
                'error': 'Camera is not active. Please start the camera stream first.'
            }), 400
        
        # 2. Decode JPEG bytes back to a BGR numpy array
        frame_array = np.frombuffer(current_frame, dtype=np.uint8)
        img_bgr = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            return jsonify({
                'success': False,
                'error': 'Failed to decode camera frame. Try again.'
            }), 500
        
        # 3. Get detection parameters from request
        data = request.get_json(silent=True) or {}
        method = data.get('method', 'color')
        threshold = float(data.get('threshold', 0.12))
        
        # Validate method
        valid_methods = ['color', 'ndvi', 'texture', 'size_filter']
        if method not in valid_methods:
            return jsonify({
                'success': False,
                'error': f'Invalid method "{method}". Use one of: {valid_methods}'
            }), 400
        
        # 4. Run weed detection
        result = run_weed_detection(img_bgr, method=method, threshold=threshold)
        
        # 5. Also encode the raw captured frame for display
        _, raw_buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        result['captured_image_b64'] = base64.b64encode(raw_buf).decode('utf-8')
        
        # 6. Log the detection event
        log_to_db('action_logs', {
            'source': 'WEED DETECT',
            'action': f"Weed scan: {result['confidence']} confidence, "
                      f"{result['weed_coverage_percent']}% coverage, "
                      f"{result['num_weed_regions']} regions ({method})"
        })
        
        result['success'] = True
        return jsonify(result)
    
    except Exception as e:
        print(f"Weed Detection Error: {e}")
        return jsonify({
            'success': False,
            'error': f'Detection failed: {str(e)}'
        }), 500


if __name__ == '__main__':
    if not os.path.exists(PLACEHOLDER_IMAGE):
        import numpy as np
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, 'Camera Offline', (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(PLACEHOLDER_IMAGE, placeholder)
    
    app.run(host='0.0.0.0', port=5000, threaded=True)


