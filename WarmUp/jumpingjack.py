import numpy as np
from utils import calculate_angle

class JumpingJackExercise:
    def __init__(self):
        self.reset()

    def reset(self):
        self.counter = 0
        self.stage = "down" # Bắt đầu ở tư thế tay xuôi theo thân
        self.feedback = "Sẵn sàng"

    def process(self, landmarks):
        # Lấy toạ độ Hip, Shoulder, Elbow (để đo độ mở nách)
        left_hip = [landmarks[23].x, landmarks[23].y]
        left_shoulder = [landmarks[11].x, landmarks[11].y]
        left_elbow = [landmarks[13].x, landmarks[13].y]
        
        right_hip = [landmarks[24].x, landmarks[24].y]
        right_shoulder = [landmarks[12].x, landmarks[12].y]
        right_elbow = [landmarks[14].x, landmarks[14].y]

        # Tính góc nách (Vai là đỉnh)
        angle_l = calculate_angle(left_hip, left_shoulder, left_elbow)
        angle_r = calculate_angle(right_hip, right_shoulder, right_elbow)
        avg_angle = (angle_l + angle_r) / 2

        # --- LOGIC ĐẾM NGHIÊM NGẶT ---
        # Lên (Tay vỗ cao): Góc nách > 160 độ
        if avg_angle > 160:
            self.stage = "up"
            self.feedback = "Vỗ tay!"
            
        # Xuống (Tay khép sát): Góc nách < 30 độ
        if avg_angle < 30 and self.stage == "up":
            self.stage = "down"
            self.counter += 1
            self.feedback = "Tốt!"

        return avg_angle, self.counter, self.feedback, self.stage