from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import csv
import os
from datetime import datetime

app = Flask(__name__)

# --- CẤU HÌNH NGƯỠNG (ĐIỀU CHỈNH ĐỘ KHẮT KHE) ---
SIDE_NECK_THRESH = 50             # Góc cổ (Text Neck)
SIDE_SHOULDER_ROUNDING_THRESH = 145 # Góc vai (Gù lưng/Vai cuộn)
FRONT_TILT_THRESH = 20            # Nghiêng đầu
FRONT_OFFSET_Y = 30               # Khoảng cách cúi (Dí mắt)
ALARM_DELAY = 3                   # Giây (Chờ 3s mới đếm lỗi)

# --- BIẾN TOÀN CỤC ---
posture_status = {"front": None, "side": None}
front_ref = {"nose_y": 0, "shoulder_y": 0, "calibrated": False}

# Quản lý phiên học
is_monitoring = False
current_session_start = None
current_user_name = ""
current_user_age = ""
target_duration_min = 0

# Bộ đếm lỗi chi tiết
error_counts = {"neck": 0, "back": 0, "tilt": 0, "close": 0}
current_realtime_errors = set() # Các lỗi đang diễn ra tức thì

# Quản lý logic báo động
bad_posture_start_time = None
current_error_counted = False 

# Quản lý vắng mặt
is_absent = False
absent_start_time = None
total_break_seconds = 0

mp_pose = mp.solutions.pose
LOG_FILE = 'nhat_ky_be_hoc.csv'

# Tạo file CSV
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Thời gian", "Tên Bé", "Tuổi", "Mục tiêu(p)", "Thực học(p)", 
            "Tổng Lỗi", "Gù Cổ", "Gù Lưng", "Nghiêng Đầu", "Dí Mắt"
        ])

# --- HÀM TOÁN HỌC & AI ---
def make_pose_detector():
    return mp_pose.Pose(model_complexity=1, smooth_landmarks=True, min_detection_confidence=0.6, min_tracking_confidence=0.6, static_image_mode=False)

def find_angle(x1, y1, x2, y2):
    v1 = (x2-x1, y2-y1); v2 = (0, -1)
    try:
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        mag = math.sqrt(v1[0]**2 + v1[1]**2) * math.sqrt(v2[0]**2 + v2[1]**2)
        return 180 - (math.degrees(math.acos(dot/mag))) if mag!=0 else 0
    except: return 0

def calculate_tilt(p1, p2):
    if p1[0] == p2[0]: return 90
    angle = abs(math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0])))
    return 180 - angle if angle > 90 else angle

def calculate_3_point_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b; bc = c - b
    try:
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(cos))
    except: return 180

# --- XỬ LÝ CAM TRƯỚC (PHÁT HIỆN VẮNG MẶT + NGHIÊNG/DÍ MẮT) ---
def gen_frames_front():
    global front_ref, posture_status, is_monitoring, current_realtime_errors
    global is_absent, absent_start_time, total_break_seconds
    
    cap = cv2.VideoCapture(1)
    cap.set(3, 1280); cap.set(4, 720)
    pose = make_pose_detector()
    
    while True:
        success, frame = cap.read()
        if not success: break
        
        h, w, _ = frame.shape
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            # CÓ NGƯỜI
            if is_monitoring and is_absent:
                is_absent = False
                if absent_start_time: total_break_seconds += (time.time() - absent_start_time); absent_start_time = None

            lm = results.pose_landmarks.landmark
            l_ear = (int(lm[mp_pose.PoseLandmark.LEFT_EAR.value].x * w), int(lm[mp_pose.PoseLandmark.LEFT_EAR.value].y * h))
            r_ear = (int(lm[mp_pose.PoseLandmark.RIGHT_EAR.value].x * w), int(lm[mp_pose.PoseLandmark.RIGHT_EAR.value].y * h))
            l_sh = (int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w), int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h))
            r_sh = (int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w), int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h))
            nose = (int(lm[mp_pose.PoseLandmark.NOSE.value].x * w), int(lm[mp_pose.PoseLandmark.NOSE.value].y * h))
            
            cv2.line(frame, l_ear, r_ear, (0, 255, 255), 2)
            
            if is_monitoring:
                current_realtime_errors.discard('tilt')
                current_realtime_errors.discard('close')
                detected_issues = []
                
                # Check Nghiêng
                if calculate_tilt(l_ear, r_ear) > FRONT_TILT_THRESH: 
                    detected_issues.append("NGHIENG DAU")
                    current_realtime_errors.add('tilt')

                # Check Dí Mắt
                if front_ref["calibrated"]:
                    cv2.line(frame, (0, front_ref["nose_y"]), (w, front_ref["nose_y"]), (0, 255, 0), 1)
                    if ((l_sh[1] + r_sh[1]) // 2 - nose[1]) < (front_ref["shoulder_y"] - front_ref["nose_y"] - FRONT_OFFSET_Y): 
                        detected_issues.append("DI MAT GAN")
                        current_realtime_errors.add('close')
                else:
                    cv2.putText(frame, "CHUA CALIB!", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

                if detected_issues:
                    posture_status["front"] = False
                    cv2.putText(frame, " | ".join(detected_issues), (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                else:
                    posture_status["front"] = True
                    cv2.putText(frame, "OK", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            else:
                posture_status["front"] = None
                cv2.putText(frame, "CHO...", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
        else:
            # VẮNG MẶT
            posture_status["front"] = None
            if is_monitoring:
                if not is_absent: is_absent = True; absent_start_time = time.time()
                current_realtime_errors.discard('tilt'); current_realtime_errors.discard('close')
                
                overlay = frame.copy(); cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                cv2.putText(frame, "BE VANG MAT", (w//2-100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- XỬ LÝ CAM CẠNH (GÙ CỔ / GÙ LƯNG) ---
def gen_frames_side():
    global posture_status, is_monitoring, current_realtime_errors
    cap = cv2.VideoCapture(0) # Lưu ý ID Cam
    cap.set(3, 1280); cap.set(4, 720)
    pose = make_pose_detector()
    
    while True:
        success, frame = cap.read()
        if not success:
            blank = np.zeros((720, 1280, 3), np.uint8)
            ret, buffer = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            continue
            
        if is_absent and is_monitoring:
            cv2.putText(frame, "TAM DUNG", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            current_realtime_errors.discard('neck'); current_realtime_errors.discard('back')
        else:
            h, w, _ = frame.shape
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks and is_monitoring:
                lm = results.pose_landmarks.landmark
                p_ear = (int(lm[mp_pose.PoseLandmark.LEFT_EAR.value].x * w), int(lm[mp_pose.PoseLandmark.LEFT_EAR.value].y * h))
                p_sh = (int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w), int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h))
                p_hip = (int(lm[mp_pose.PoseLandmark.LEFT_HIP.value].x * w), int(lm[mp_pose.PoseLandmark.LEFT_HIP.value].y * h))
                
                cv2.line(frame, p_ear, p_sh, (0, 255, 255), 3)
                cv2.line(frame, p_sh, p_hip, (0, 255, 255), 3)
                
                current_realtime_errors.discard('neck'); current_realtime_errors.discard('back')
                detected_issues = []
                
                neck = find_angle(p_sh[0], p_sh[1], p_ear[0], p_ear[1])
                sh_round = calculate_3_point_angle(p_ear, p_sh, p_hip)
                
                # Check Gù Cổ & Lưng
                if neck > SIDE_NECK_THRESH: 
                    detected_issues.append(f"GU CO ({int(neck)})")
                    current_realtime_errors.add('neck')
                if sh_round < SIDE_SHOULDER_ROUNDING_THRESH: 
                    detected_issues.append(f"GU LUNG ({int(sh_round)})")
                    current_realtime_errors.add('back')
                
                if detected_issues:
                    posture_status["side"] = False
                    cv2.putText(frame, "|".join(detected_issues), (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                else:
                    posture_status["side"] = True
                    cv2.putText(frame, "OK", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            else:
                posture_status["side"] = None

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- API QUẢN LÝ ---
@app.route('/start_session', methods=['POST'])
def start_session():
    global is_monitoring, current_session_start, current_user_name, current_user_age, target_duration_min
    global total_break_seconds, is_absent, absent_start_time, bad_posture_start_time
    global error_counts, current_error_counted, current_realtime_errors

    data = request.json
    current_user_name = data.get('name', 'Bé')
    current_user_age = data.get('age', '')
    target_duration_min = float(data.get('duration', 0))
    
    current_session_start = datetime.now()
    is_monitoring = True
    
    # Reset
    total_break_seconds = 0
    is_absent = False; absent_start_time = None; bad_posture_start_time = None
    current_error_counted = False; current_realtime_errors = set()
    error_counts = {"neck": 0, "back": 0, "tilt": 0, "close": 0}
    
    return jsonify({"status": "started"})

@app.route('/stop_session', methods=['POST'])
def stop_session():
    global is_monitoring, error_counts
    if is_monitoring and current_session_start:
        end_time = datetime.now()
        if is_absent and absent_start_time:
            global total_break_seconds
            total_break_seconds += (time.time() - absent_start_time)

        total_min = round((end_time - current_session_start).total_seconds() / 60, 2)
        break_min = round(total_break_seconds / 60, 2)
        actual_min = round(total_min - break_min, 2)
        total_errors = sum(error_counts.values())

        # Ghi CSV
        with open(LOG_FILE, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([
                current_session_start.strftime("%Y-%m-%d %H:%M"),
                current_user_name, current_user_age, target_duration_min, actual_min,
                total_errors, error_counts['neck'], error_counts['back'], error_counts['tilt'], error_counts['close']
            ])
        
        is_monitoring = False
        return jsonify({"status": "stopped", "actual": actual_min, "errors": error_counts})
    return jsonify({"status": "error"})

@app.route('/check_status')
def check_status():
    global bad_posture_start_time, error_counts, current_error_counted
    
    if not is_monitoring or is_absent:
        return jsonify({"active": is_monitoring, "absent": is_absent, "alarm": False, "counts": error_counts})

    is_bad = len(current_realtime_errors) > 0
    alarm = False
    
    if is_bad:
        if bad_posture_start_time is None: bad_posture_start_time = time.time()
        elif time.time() - bad_posture_start_time > ALARM_DELAY:
            alarm = True
            if not current_error_counted: # Cộng lỗi vào chi tiết
                for err in current_realtime_errors:
                    if err in error_counts: error_counts[err] += 1
                current_error_counted = True 
    else:
        bad_posture_start_time = None
        current_error_counted = False 
        
    return jsonify({
        "active": True, "absent": False, 
        "front": posture_status["front"], "side": posture_status["side"], 
        "alarm": alarm, "counts": error_counts
    })

# Routes phụ
@app.route('/calibrate_front')
def calibrate_front():
    global front_ref
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280); cap.set(4, 720)
    ret, frame = cap.read()
    cap.release()
    if ret:
        with make_pose_detector() as pose:
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                h, w, _ = frame.shape
                lm = res.pose_landmarks.landmark
                front_ref["nose_y"] = int(lm[mp_pose.PoseLandmark.NOSE.value].y * h)
                l_y = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
                r_y = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
                front_ref["shoulder_y"] = int(((l_y + r_y)/2) * h)
                front_ref["calibrated"] = True
                return jsonify({"status": "success"})
    return jsonify({"status": "failed"})

@app.route('/get_history')
def get_history():
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader: logs.append(row)
    return jsonify(logs[-5:][::-1])

@app.route('/')
def index(): return render_template('index.html')
@app.route('/video_front')
def video_front(): return Response(gen_frames_front(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/video_side')
def video_side(): return Response(gen_frames_side(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5000)