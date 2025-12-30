from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import math
import time

app = Flask(__name__)

# --- CẤU HÌNH NGƯỠNG (THRESHOLDS) ---
# 1. SIDE VIEW (Góc nhìn bên cạnh)
SIDE_NECK_THRESH = 45             # Góc cổ (tăng lên 45 cho thoải mái hơn chút)
SIDE_TORSO_THRESH = 15            # Góc thân
SIDE_SHOULDER_ROUNDING_THRESH = 155 # Góc cuộn vai (Rounded Shoulder)

# 2. FRONT VIEW (Góc nhìn chính diện)
FRONT_TILT_THRESH = 20  # Nghiêng đầu/vai
FRONT_OFFSET_Y = 30     # Cúi đầu (pixel)

# --- BIẾN TOÀN CỤC ---
posture_status = {"front": None, "side": None}
front_ref = {"nose_y": 0, "shoulder_y": 0, "calibrated": False}
bad_posture_start_time = None
ALARM_DELAY = 3  # Giây

mp_pose = mp.solutions.pose

# --- HÀM CẤU HÌNH ---
def make_pose_detector():
    """
    Cấu hình High-Precision cho MediaPipe:
    - model_complexity=2: Dùng model nặng nhất, chính xác nhất.
    - smooth_landmarks=True: Bật bộ lọc chống rung.
    - confidence=0.7: Chỉ nhận diện khi độ tin cậy > 70%.
    """
    return mp_pose.Pose(
        model_complexity=2,           # 0=Lite, 1=Full, 2=Heavy (Chuẩn nhất)
        smooth_landmarks=True,        # Giảm rung
        min_detection_confidence=0.7, # Lọc bỏ nhiễu
        min_tracking_confidence=0.7,  # Theo dõi sát sao hơn
        static_image_mode=False
    )

# --- HÀM TOÁN HỌC ---
def find_angle(x1, y1, x2, y2):
    """Tính góc so với trục dọc (Vertical Axis)"""
    v1_x, v1_y = x2 - x1, y2 - y1
    v2_x, v2_y = 0, -1 
    try:
        dot_product = v1_x * v2_x + v1_y * v2_y
        m1 = math.sqrt(v1_x**2 + v1_y**2)
        m2 = math.sqrt(v2_x**2 + v2_y**2)
        if m1 * m2 == 0: return 0
        angle = math.degrees(math.acos(dot_product / (m1 * m2)))
        if angle > 90: angle = 180 - angle
        return angle
    except: return 0

def calculate_tilt(p1, p2):
    """Tính góc nghiêng so với trục ngang (Horizontal)"""
    if p1[0] == p2[0]: return 90
    dy = p2[1] - p1[1]
    dx = p2[0] - p1[0]
    angle = abs(math.degrees(math.atan2(dy, dx)))
    if angle > 180: angle -= 180
    if angle > 90: angle = 180 - angle
    return angle

def calculate_3_point_angle(a, b, c):
    """Tính góc kẹp tại b giữa 3 điểm a-b-c"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    try:
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(cosine_angle))
    except: return 180

# --- XỬ LÝ CAM CHÍNH DIỆN (FRONT) ---
def gen_frames_front():
    global front_ref, posture_status
    cap = cv2.VideoCapture(0)
    # [QUAN TRỌNG] Tăng độ phân giải lên HD để nhận diện xa tốt hơn
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    pose = make_pose_detector()
    
    while True:
        success, frame = cap.read()
        if not success:
            posture_status["front"] = None
            break
            
        h, w, _ = frame.shape
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        msg = "Waiting..."
        color = (128, 128, 128)
        errors = []

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            nose = lm[mp_pose.PoseLandmark.NOSE.value]
            l_ear = lm[mp_pose.PoseLandmark.LEFT_EAR.value]
            r_ear = lm[mp_pose.PoseLandmark.RIGHT_EAR.value]
            l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            
            nose_pos = (int(nose.x * w), int(nose.y * h))
            l_ear_pos = (int(l_ear.x * w), int(l_ear.y * h))
            r_ear_pos = (int(r_ear.x * w), int(r_ear.y * h))
            l_sh_pos = (int(l_sh.x * w), int(l_sh.y * h))
            r_sh_pos = (int(r_sh.x * w), int(r_sh.y * h))

            # Logic Check
            if calculate_tilt(l_ear_pos, r_ear_pos) > FRONT_TILT_THRESH:
                errors.append("NGHIENG DAU")
            if calculate_tilt(l_sh_pos, r_sh_pos) > FRONT_TILT_THRESH:
                errors.append("LECH VAI")

            if not front_ref["calibrated"]:
                cv2.putText(frame, "CHUA SET CHUAN", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                posture_status["front"] = None
            else:
                cv2.line(frame, (0, front_ref["nose_y"]), (w, front_ref["nose_y"]), (0, 255, 0), 1)
                sh_mid_y = (l_sh_pos[1] + r_sh_pos[1]) // 2
                if (sh_mid_y - nose_pos[1]) < (front_ref["shoulder_y"] - front_ref["nose_y"] - FRONT_OFFSET_Y):
                    errors.append("CUI DAU")

                if errors:
                    posture_status["front"] = False
                    color = (0, 0, 255)
                    msg = " | ".join(errors)
                else:
                    posture_status["front"] = True
                    color = (0, 255, 0)
                    msg = "Front: OK"

            # Vẽ 
            cv2.line(frame, l_ear_pos, r_ear_pos, (255, 255, 0), 2)
            cv2.line(frame, l_sh_pos, r_sh_pos, (255, 255, 0), 2)
        else:
            posture_status["front"] = None
            msg = "No Face"

        cv2.putText(frame, msg, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- XỬ LÝ CAM BÊN CẠNH (SIDE) ---
def gen_frames_side():
    global posture_status
    cap = cv2.VideoCapture(1) # Thay đổi ID nếu cần (0, 1, 2)
    
    # [QUAN TRỌNG] Tăng độ phân giải lên HD để nhận diện xa tốt hơn
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    pose = make_pose_detector()
    
    while True:
        success, frame = cap.read()
        if not success:
            posture_status["side"] = None
            blank = np.zeros((720, 1280, 3), np.uint8) 
            cv2.putText(blank, "NO CAM 2", (300, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
            ret, buffer = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            continue

        h, w, _ = frame.shape
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        msg = "Waiting..."
        color = (128, 128, 128)
        errors = []

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            l_ear = lm[mp_pose.PoseLandmark.LEFT_EAR.value]
            l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            l_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
            
            p_ear = (int(l_ear.x * w), int(l_ear.y * h))
            p_sh = (int(l_sh.x * w), int(l_sh.y * h))
            p_hip = (int(l_hip.x * w), int(l_hip.y * h))
            
            # Tính toán
            neck_angle = find_angle(p_sh[0], p_sh[1], p_ear[0], p_ear[1])
            torso_angle = find_angle(p_hip[0], p_hip[1], p_sh[0], p_sh[1])
            shoulder_round_angle = calculate_3_point_angle(p_ear, p_sh, p_hip)
            
            # Check lỗi
            if neck_angle > SIDE_NECK_THRESH: errors.append(f"CO RUA ({int(neck_angle)})")
            if torso_angle > SIDE_TORSO_THRESH: errors.append(f"NGA LUNG ({int(torso_angle)})")
            if shoulder_round_angle < SIDE_SHOULDER_ROUNDING_THRESH: errors.append(f"GU VAI ({int(shoulder_round_angle)})")

            if errors:
                posture_status["side"] = False
                color = (0, 0, 255)
                msg = " | ".join(errors)
            else:
                posture_status["side"] = True
                color = (0, 255, 0)
                msg = f"OK (V:{int(shoulder_round_angle)} C:{int(neck_angle)})"

            # Vẽ
            cv2.line(frame, p_ear, p_sh, (0, 255, 255), 3)
            cv2.line(frame, p_sh, p_hip, (0, 255, 255), 3)
            cv2.circle(frame, p_sh, 8, color, -1)
            cv2.putText(frame, str(int(shoulder_round_angle)), (p_sh[0]-50, p_sh[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        else:
            posture_status["side"] = None
            msg = "No Body"

        cv2.putText(frame, msg, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
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
    global front_ref
    cap = cv2.VideoCapture(0)
    # Calibrate cũng cần HD để đồng bộ
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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
    is_any_bad = (posture_status["front"] is False) or (posture_status["side"] is False)
    alarm_trigger = False
    if is_any_bad:
        if bad_posture_start_time is None: bad_posture_start_time = time.time()
        elif time.time() - bad_posture_start_time > ALARM_DELAY: alarm_trigger = True
    else: bad_posture_start_time = None
    return jsonify({"front": posture_status["front"], "side": posture_status["side"], "alarm": alarm_trigger})

if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5000)
