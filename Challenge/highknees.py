import numpy as np
from utils import calculate_angle

class HighKneesExercise:
    def __init__(self):
        self.reset()

    def reset(self):
        self.counter = 0
        self.stage = "down"
        self.feedback = "Chạy tại chỗ!"

    def process(self, landmarks):
        # Đo góc Hông: Vai - Hông - Gối (Đùi càng cao thì góc này càng nhỏ)
        l_shdr = [landmarks[11].x, landmarks[11].y]
        l_hip = [landmarks[23].x, landmarks[23].y]
        l_knee = [landmarks[25].x, landmarks[25].y]
        
        r_shdr = [landmarks[12].x, landmarks[12].y]
        r_hip = [landmarks[24].x, landmarks[24].y]
        r_knee = [landmarks[26].x, landmarks[26].y]

        angle_l = calculate_angle(l_shdr, l_hip, l_knee)
        angle_r = calculate_angle(r_shdr, r_hip, r_knee)

        # Lấy chân đang nâng cao nhất (góc nhỏ nhất)
        active_angle = min(angle_l, angle_r)

        # --- LOGIC ĐẾM ---
        # Nâng cao (Up): Góc hông < 110 độ (Đùi vuông góc hoặc hơn)
        if active_angle < 110:
             if self.stage == "down": # Chỉ tính khi chuyển trạng thái
                 self.stage = "up"
                 self.counter += 1
                 self.feedback = "Cao đùi!"
        
        # Hạ chân (Down): Góc hông > 160 độ (Chân duỗi thẳng)
        elif active_angle > 160:
            self.stage = "down"

        return active_angle, self.counter, self.feedback, self.stage