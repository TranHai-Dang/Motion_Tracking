from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import math
import time

app = Flask(__name__)

# --- CẤU HÌNH NGƯỠNG (THRESHOLDS) ---
# 1. SIDE VIEW (Theo LearnOpenCV)
# Góc tính theo trục thẳng đứng (0 độ là thẳng đứng)
SIDE_NECK_THRESH = 40   # Nếu cổ nghiêng quá 40 độ -> Gù cổ (Text Neck)
SIDE_TORSO_THRESH = 10  # Nếu thân nghiêng quá 10 độ -> Gù lưng hoặc ngả quá đà

# 2. FRONT VIEW
FRONT_TILT_THRESH = 20  # Nghiêng đầu/vai quá 20 độ -> Lệch
FRONT_OFFSET_Y = 30     # Sai số khoảng cách mũi-vai (pixel)

# --- BIẾN TOÀN CỤC ---
posture_status = {"front": None, "side": None}
front_ref = {"nose_y": 0, "shoulder_y": 0, "calibrated": False}
bad_posture_start_time = None
ALARM_DELAY = 3  # Giây

mp_pose = mp.solutions.pose

# --- HÀM TOÁN HỌC (GEOMETRY) ---

def find_angle(x1, y1, x2, y2):
    """
    Tính góc của đoạn thẳng nối (x1,y1)-(x2,y2) so với trục dọc (Vertical Axis).
    Output: 0-180 độ.
    """
    # Vector của đoạn thẳng
    v1_x, v1_y = x2 - x1, y2 - y1
    # Vector trục dọc (hướng lên trên để so sánh độ mở)
    v2_x, v2_y = 0, -1 
    
    # Công thức góc giữa 2 vector: cos(theta) = (v1.v2) / (|v1|*|v2|)
    try:
        dot_product = v1_x * v2_x + v1_y * v2_y
        magnitude_v1 = math.sqrt(v1_x**2 + v1_y**2)
        magnitude_v2 = math.sqrt(v2_x**2 + v2_y**2)
        
        if magnitude_v1 * magnitude_v2 == 0: return 0
        
        angle_rad = math.acos(dot_product / (magnitude_v1 * magnitude_v2))
        angle_deg = math.degrees(angle_rad)
        
        # Vì trục y ảnh hướng xuống, ta cần điều chỉnh lại góc nhìn cho thuận mắt
        # Nếu angle > 90 tức là đoạn thẳng đang hướng xuống
        # Ta lấy góc nhọn so với phương thẳng đứng
        if angle_deg > 90:
            angle_deg = 180 - angle_deg
            
        return angle_deg
    except:
        return 0

def calculate_tilt(p1, p2):
    """Tính góc nghiêng so với trục ngang (Horizontal) cho Front View"""
    if p1[0] == p2[0]: return 90
    dy = p2[1] - p1[1]
    dx = p2[0] - p1[0]
    angle = abs(math.degrees(math.atan2(dy, dx)))
    if angle > 180: angle -= 180
    if angle > 90: angle = 180 - angle
    return angle

def make_pose_detector():
    return mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- XỬ LÝ CAM CHÍNH DIỆN (FRONT) ---
def gen_frames_front():
    global front_ref, posture_status
    cap = cv2.VideoCapture(0)
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
            
            # Pixel Coordinates
            nose_pos = (int(nose.x * w), int(nose.y * h))
            l_ear_pos = (int(l_ear.x * w), int(l_ear.y * h))
            r_ear_pos = (int(r_ear.x * w), int(r_ear.y * h))
            l_sh_pos = (int(l_sh.x * w), int(l_sh.y * h))
            r_sh_pos = (int(r_sh.x * w), int(r_sh.y * h))

            # 1. Check Nghiêng Đầu (Head Tilt)
            head_tilt = calculate_tilt(l_ear_pos, r_ear_pos)
            if head_tilt > FRONT_TILT_THRESH:
                errors.append(f"NGHIENG DAU ({int(head_tilt)})")

            # 2. Check Lệch Vai (Shoulder Tilt)
            sh_tilt = calculate_tilt(l_sh_pos, r_sh_pos)
            if sh_tilt > FRONT_TILT_THRESH:
                errors.append(f"LECH VAI ({int(sh_tilt)})")

            # 3. Check Cúi Đầu (Forward Head) - Dùng Calibration
            if not front_ref["calibrated"]:
                cv2.putText(frame, "CHUA SET CHUAN", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                posture_status["front"] = None
            else:
                cv2.line(frame, (0, front_ref["nose_y"]), (w, front_ref["nose_y"]), (0, 255, 0), 1)
                sh_mid_y = (l_sh_pos[1] + r_sh_pos[1]) // 2
                
                ref_dist = front_ref["shoulder_y"] - front_ref["nose_y"]
                curr_dist = sh_mid_y - nose_pos[1]
                
                if curr_dist < (ref_dist - FRONT_OFFSET_Y):
                    errors.append("CUI DAU")

                # Tổng hợp kết quả Front
                if len(errors) > 0:
                    posture_status["front"] = False
                    color = (0, 0, 255) # Đỏ
                    msg = " | ".join(errors)
                else:
                    posture_status["front"] = True
                    color = (0, 255, 0) # Xanh
                    msg = "Front: OK"

            # Vẽ Visualization
            cv2.line(frame, l_ear_pos, r_ear_pos, (255, 255, 0), 2)
            cv2.line(frame, l_sh_pos, r_sh_pos, (255, 255, 0), 2)
            cv2.circle(frame, nose_pos, 5, (255, 0, 0), -1)

        else:
            posture_status["front"] = None
            msg = "No Face"

        cv2.putText(frame, msg, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- XỬ LÝ CAM BÊN CẠNH (SIDE - THEO LEARNOPENCV) ---
def gen_frames_side():
    global posture_status
    cap = cv2.VideoCapture(1) # Thay đổi index nếu cần (0, 1, 2)
    pose = make_pose_detector()
    
    while True:
        success, frame = cap.read()
        if not success:
            posture_status["side"] = None
            blank = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(blank, "NO CAM 2", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
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
            
            # Lấy các điểm quan trọng: Tai, Vai, Hông
            l_ear = lm[mp_pose.PoseLandmark.LEFT_EAR.value]
            l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            l_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
            
            # Convert sang pixel
            p_ear = (int(l_ear.x * w), int(l_ear.y * h))
            p_sh = (int(l_sh.x * w), int(l_sh.y * h))
            p_hip = (int(l_hip.x * w), int(l_hip.y * h))
            
            # --- LOGIC TÍNH GÓC (THEO TRỤC DỌC) ---
            # 1. Neck Inclination (Góc cổ): Đường nối Vai -> Tai so với trục dọc
            neck_angle = find_angle(p_sh[0], p_sh[1], p_ear[0], p_ear[1])
            
            # 2. Torso Inclination (Góc thân): Đường nối Hông -> Vai so với trục dọc
            torso_angle = find_angle(p_hip[0], p_hip[1], p_sh[0], p_sh[1])
            
            # --- KIỂM TRA LỖI ---
            # LearnOpenCV gợi ý: Neck < 40 là tốt. Torso < 10 là tốt.
            if neck_angle > SIDE_NECK_THRESH:
                errors.append(f"CO RUA ({int(neck_angle)})")
            
            if torso_angle > SIDE_TORSO_THRESH:
                errors.append(f"GU LUNG ({int(torso_angle)})")

            if len(errors) > 0:
                posture_status["side"] = False
                color = (0, 0, 255)
                msg = " | ".join(errors)
            else:
                posture_status["side"] = True
                color = (0, 255, 0)
                msg = f"Good (N:{int(neck_angle)} T:{int(torso_angle)})"

            # Vẽ Visualization
            cv2.line(frame, p_ear, p_sh, (0, 255, 255), 2) # Vàng: Cổ
            cv2.line(frame, p_sh, p_hip, (255, 0, 255), 2) # Tím: Lưng
            
            cv2.circle(frame, p_ear, 5, (255, 255, 0), -1)
            cv2.circle(frame, p_sh, 5, (255, 255, 0), -1)
            cv2.circle(frame, p_hip, 5, (255, 255, 0), -1)
            
            # Hiển thị số đo góc ngay cạnh khớp
            cv2.putText(frame, str(int(neck_angle)), (p_sh[0], p_sh[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, str(int(torso_angle)), (p_hip[0], p_hip[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        else:
            posture_status["side"] = None
            msg = "Can't see full body"

        cv2.putText(frame, msg, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- ROUTING GIỮ NGUYÊN ---
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
        if bad_posture_start_time is None:
            bad_posture_start_time = time.time()
        elif time.time() - bad_posture_start_time > ALARM_DELAY:
            alarm_trigger = True
    else:
        bad_posture_start_time = None
        
    return jsonify({
        "front": posture_status["front"],
        "side": posture_status["side"],
        "alarm": alarm_trigger
    })

if __name__ == "__main__":
    app.run(debug=True, threaded=True, port=5000)