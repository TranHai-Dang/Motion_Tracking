from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import math
import time

app = Flask(__name__)

# --- CẤU HÌNH ---
SIDE_THRESHOLD = 75 
FRONT_OFFSET_Y = 30 
ALARM_DELAY = 3 # Giây

# --- BIẾN TRẠNG THÁI (3 TRẠNG THÁI) ---
# True: Tốt | False: Xấu | None: Không tìm thấy người
posture_status = {
    "front": None, 
    "side": None
}

front_ref = {"nose_y": 0, "shoulder_y": 0, "calibrated": False}
bad_posture_start_time = None

mp_pose = mp.solutions.pose

# --- HÀM HỖ TRỢ ---
def calculate_angle(a, b):
    if not a or not b: return 0
    angle = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
    return abs(angle)

def make_pose_detector():
    return mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- XỬ LÝ CAMERA TRƯỚC (FRONT) ---
def gen_frames_front():
    global front_ref, posture_status
    cap = cv2.VideoCapture(0)
    pose = make_pose_detector()
    
    while True:
        success, frame = cap.read()
        if not success: 
            posture_status["front"] = None # Mất kết nối cam
            break
        
        h, w, _ = frame.shape
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        msg = "Waiting..."
        color = (128, 128, 128) # Xám

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            nose = lm[mp_pose.PoseLandmark.NOSE.value]
            l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            
            nose_y = int(nose.y * h)
            sh_mid_y = int((l_sh.y * h + r_sh.y * h) / 2)
            
            cv2.circle(frame, (int(nose.x * w), nose_y), 5, (255, 0, 0), -1)

            if not front_ref["calibrated"]:
                cv2.putText(frame, "Chua Set Chuan", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                posture_status["front"] = None # Chưa set chuẩn thì coi như chưa giám sát
            else:
                cv2.line(frame, (0, front_ref["nose_y"]), (w, front_ref["nose_y"]), (0, 255, 0), 1)
                ref_dist = front_ref["shoulder_y"] - front_ref["nose_y"]
                curr_dist = sh_mid_y - nose_y
                
                if curr_dist < (ref_dist - FRONT_OFFSET_Y):
                    posture_status["front"] = False
                    color = (0, 0, 255)
                    msg = "Front: BAD!"
                else:
                    posture_status["front"] = True
                    color = (0, 255, 0)
                    msg = "Front: OK"
        else:
            posture_status["front"] = None # Không thấy người
            msg = "No Face Detected"

        cv2.putText(frame, msg, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- XỬ LÝ CAMERA BÊN (SIDE) ---
def gen_frames_side():
    global posture_status
    cap = cv2.VideoCapture(1) # Đổi số này nếu dùng cam khác
    pose = make_pose_detector()
    
    while True:
        success, frame = cap.read()
        if not success: 
            posture_status["side"] = None
            # Gửi ảnh đen báo lỗi
            blank = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(blank, "NO SIGNAL CAM 2", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            ret, buffer = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            continue

        h, w, _ = frame.shape
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        msg = "Waiting..."
        color = (128, 128, 128)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            l_ear = lm[mp_pose.PoseLandmark.LEFT_EAR.value]
            l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            
            p_ear = (int(l_ear.x * w), int(l_ear.y * h))
            p_sh = (int(l_sh.x * w), int(l_sh.y * h))
            
            cv2.line(frame, p_ear, p_sh, (255, 255, 0), 3)
            angle = calculate_angle(p_ear, p_sh)
            
            if angle < SIDE_THRESHOLD or angle > 160:
                posture_status["side"] = False
                color = (0, 0, 255)
                msg = f"Side: BAD ({int(angle)})"
            else:
                posture_status["side"] = True
                color = (0, 255, 0)
                msg = f"Side: OK ({int(angle)})"
        else:
            posture_status["side"] = None # Không thấy người
            msg = "No Body Detected"

        cv2.putText(frame, msg, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- FLASK ROUTES ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_front')
def video_front(): return Response(gen_frames_front(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_side')
def video_side(): return Response(gen_frames_side(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/calibrate_front')
def calibrate_front():
    # ... (Giữ nguyên logic cũ hoặc cải thiện việc bắt frame)
    global front_ref
    cap = cv2.VideoCapture(0)
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

@app.route('/check_status')
def check_status():
    global bad_posture_start_time
    
    # Logic mới: Chỉ báo động khi có người VÀ ngồi sai (False).
    # Nếu là None (không có người) thì bỏ qua.
    
    front_bad = (posture_status["front"] is False)
    side_bad = (posture_status["side"] is False)
    
    is_any_bad = front_bad or side_bad
    
    alarm_trigger = False
    if is_any_bad:
        if bad_posture_start_time is None:
            bad_posture_start_time = time.time()
        elif time.time() - bad_posture_start_time > ALARM_DELAY:
            alarm_trigger = True
    else:
        bad_posture_start_time = None
        
    return jsonify({
        "front": posture_status["front"], # True/False/None
        "side": posture_status["side"],   # True/False/None
        "alarm": alarm_trigger
    })

if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5000)