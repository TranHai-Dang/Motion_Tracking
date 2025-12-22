from flask import Flask, render_template, Response, request, jsonify
import cv2
import time

# Import các file logic cũ của bạn
from helper import CameraStream
from squat import SquatExercise
from jumping_jack import JumpingJackExercise

app = Flask(__name__)

# --- KHỞI TẠO TOÀN CỤC ---
# Khởi tạo Camera và AI một lần duy nhất
camera = CameraStream(0).start() # Đổi thành 1 nếu dùng Cam rời
time.sleep(1.0)

# Khởi tạo các bài tập
squat_exercise = SquatExercise()
jj_exercise = JumpingJackExercise()

# Biến lưu trạng thái hiện tại (Mặc định là Squat)
current_mode = "Squat"

# Khởi tạo MediaPipe
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(model_complexity=1, smooth_landmarks=True)

def generate_frames():
    global current_mode
    while True:
        frame = camera.read()
        if frame is None: continue

        # 1. Xử lý ảnh
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        # Biến tạm để vẽ
        counter, stage, feedback = 0, "N/A", "Ready"

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # 2. CHỌN LOGIC THEO MODE
            if current_mode == "Squat":
                frame, counter, stage, feedback, _ = squat_exercise.process(frame, landmarks)
            elif current_mode == "Jumping Jack":
                frame, counter, stage, feedback, _ = jj_exercise.process(frame, landmarks)

        # 3. Vẽ UI cơ bản lên video (Optional - vì web đã có UI rồi)
        # Nhưng vẽ lên để debug cũng tốt
        cv2.putText(frame, f"MODE: {current_mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # 4. Mã hóa ảnh thành JPG để gửi về web
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Trả về theo chuẩn MJPEG (Motion JPEG)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Trang chủ hiển thị giao diện"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Đường dẫn video stream"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/change_mode', methods=['POST'])
def change_mode():
    """API để đổi bài tập khi bấm nút"""
    global current_mode
    data = request.json
    new_mode = data.get('mode')
    
    if new_mode in ["Squat", "Jumping Jack"]:
        current_mode = new_mode
        # Reset counter khi đổi bài
        if new_mode == "Squat": squat_exercise.counter = 0
        if new_mode == "Jumping Jack": jj_exercise.counter = 0
        return jsonify({"status": "success", "mode": current_mode})
    
    return jsonify({"status": "error"}), 400

if __name__ == "__main__":
    # debug=True để tự reload khi sửa code
    app.run(host='0.0.0.0', port=5000, debug=True)