import cv2
import mediapipe as mp
import threading
import time
import math

# --- CLASS ĐỌC VIDEO (GIỮ NGUYÊN ĐỂ KHÔNG BỊ DELAY) ---
class UDPVideoStream:
    def __init__(self, src):
        # Tăng buffer size để tránh lỗi mjpeg overread
        self.cap = cv2.VideoCapture(0)
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                continue
            grabbed, frame = self.cap.read()
            if grabbed:
                self.grabbed = grabbed
                self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# --- CẤU HÌNH MEDIAPIPE ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1, 
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- MAIN ---
udp_url = 0
print("Đang kết nối và xử lý 33 điểm Pose...")

stream = UDPVideoStream(udp_url).start()
time.sleep(1.0) 

try:
    while True:
        frame = stream.read()
        if frame is None:
            continue

        h, w, _ = frame.shape
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        image_rgb.flags.writeable = True

        if results.pose_landmarks:
            # 1. Vẽ bộ khung xương nối sẵn (Skeleton)
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # 2. Duyệt qua từng điểm trong 33 điểm để lấy toạ độ
            print("--- New Frame ---")
            for id, lm in enumerate(results.pose_landmarks.landmark):
                # lm.x, lm.y là toạ độ tỉ lệ (0.0 -> 1.0)
                # lm.z là độ sâu (càng âm càng gần camera)
                # lm.visibility là độ tin cậy điểm đó xuất hiện trong hình
                
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                # In toạ độ ra console (bỏ comment nếu muốn xem)
                # print(f"ID {id}: ({cx}, {cy}) - Conf: {lm.visibility:.2f}")

                # Vẽ số ID lên khớp xương nếu độ tin cậy > 0.5
                if lm.visibility > 0.5:
                    cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            # 3. Nhận diện tư thế: Đứng thẳng hoặc Squat
            # Lấy các điểm quan trọng
            left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
            left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]

            right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
            right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

            # Tính góc giữa hip-knee-ankle (trái và phải)

            def calculate_angle(a, b, c):
                """Tính góc tại điểm b"""
                angle = math.degrees(
                    math.atan2(c.y - b.y, c.x - b.x) - 
                    math.atan2(a.y - b.y, a.x - b.x)
                )
                return abs(angle)

            left_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_angle = calculate_angle(right_hip, right_knee, right_ankle)
            avg_angle = (left_angle + right_angle) / 2

            # Phân loại tư thế dựa trên góc gối
            # Góc > 160: Đứng thẳng
            # Góc < 120: Squat
            if avg_angle > 160:
                posture = "DUNG THANG"
                color = (0, 255, 0)  # Xanh lá
            elif avg_angle < 120:
                posture = "SQUAT"
                color = (0, 0, 255)  # Đỏ
            else:
                posture = "TRUNG GIAN"
                color = (0, 165, 255)  # Cam

            # Hiển thị tư thế lên màn hình
            cv2.putText(frame, f"Tu the: {posture}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Goc goi: {avg_angle:.1f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('WSL Pose 33 Landmarks', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopping...")

finally:
    stream.stop()
    pose.close()
    cv2.destroyAllWindows()