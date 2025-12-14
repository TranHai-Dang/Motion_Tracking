import numpy as np
from utils import calculate_angle

class PushUpExercise:
    def __init__(self):
        self.reset()

    def reset(self):
        self.counter = 0
        self.stage = "up" # Bắt đầu ở tư thế tay thẳng
        self.feedback = "Sẵn sàng"

    def process(self, landmarks):
        # Lấy toạ độ Shoulder, Elbow, Wrist
        left_shoulder = [landmarks[11].x, landmarks[11].y]
        left_elbow = [landmarks[13].x, landmarks[13].y]
        left_wrist = [landmarks[15].x, landmarks[15].y]
        
        right_shoulder = [landmarks[12].x, landmarks[12].y]
        right_elbow = [landmarks[14].x, landmarks[14].y]
        right_wrist = [landmarks[16].x, landmarks[16].y]

        # Tính góc khuỷu tay
        angle_l = calculate_angle(left_shoulder, left_elbow, left_wrist)
        angle_r = calculate_angle(right_shoulder, right_elbow, right_wrist)
        avg_angle = (angle_l + angle_r) / 2

        # --- LOGIC ĐẾM NGHIÊM NGẶT ---
        # Xuống: Gập tay < 90 độ
        if avg_angle < 90:
            self.stage = "down"
            self.feedback = "Giữ thân người thẳng"
        
        # Lên: Duỗi tay > 160 độ
        if avg_angle > 160 and self.stage == "down":
            self.stage = "up"
            self.counter += 1
            self.feedback = "Tốt lắm!"
            
        # Cảnh báo lưng võng (Kiểm tra thêm góc vai-hông-gối nếu muốn, ở đây check đơn giản)
        if self.stage == "down" and avg_angle > 100:
             self.feedback = "Xuống sâu hơn nữa!"

        return avg_angle, self.counter, self.feedback, self.stage