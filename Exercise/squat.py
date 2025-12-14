import numpy as np
from utils import calculate_angle

class SquatExercise:
    def __init__(self):
        self.reset()

    def reset(self):
        self.counter = 0
        self.stage = "up" # Bắt đầu ở tư thế đứng
        self.feedback = "Sẵn sàng"

    def process(self, landmarks):
        # Lấy toạ độ Hip, Knee, Ankle
        left_hip = [landmarks[23].x, landmarks[23].y]
        left_knee = [landmarks[25].x, landmarks[25].y]
        left_ankle = [landmarks[27].x, landmarks[27].y]
        
        right_hip = [landmarks[24].x, landmarks[24].y]
        right_knee = [landmarks[26].x, landmarks[26].y]
        right_ankle = [landmarks[28].x, landmarks[28].y]

        # Tính góc đầu gối
        angle_l = calculate_angle(left_hip, left_knee, left_ankle)
        angle_r = calculate_angle(right_hip, right_knee, right_ankle)
        avg_angle = (angle_l + angle_r) / 2

        # --- LOGIC ĐẾM NGHIÊM NGẶT ---
        # 1. Xuống (Down): Góc phải nhỏ hơn 90 độ (Ngồi sâu)
        if avg_angle < 90:
            self.stage = "down"
            self.feedback = "Tốt! Giữ lưng thẳng"

        # 2. Lên (Up): Góc phải lớn hơn 160 độ (Đứng thẳng) & Đã từng xuống
        if avg_angle > 160 and self.stage == "down":
            self.stage = "up"
            self.counter += 1
            self.feedback = "Tuyệt vời!"

        # Cảnh báo nếu xuống chưa đủ sâu (trong khoảng lấp lửng 90-140 khi đang xuống)
        if self.stage == "down" and avg_angle > 100:
             self.feedback = "Hạ thấp hông xuống!"

        return avg_angle, self.counter, self.feedback, self.stage