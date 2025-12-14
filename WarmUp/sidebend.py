import numpy as np
from utils import calculate_angle

class SideBendExercise:
    def __init__(self):
        self.reset()

    def reset(self):
        self.counter = 0
        self.stage = "center" # Bắt đầu đứng thẳng
        self.feedback = "Đứng thẳng"

    def process(self, landmarks):
        # Đo góc nghiêng bên trái và phải: Vai - Hông - Gối
        # Bên Trái
        l_shdr = [landmarks[11].x, landmarks[11].y]
        l_hip = [landmarks[23].x, landmarks[23].y]
        l_knee = [landmarks[25].x, landmarks[25].y]
        angle_l = calculate_angle(l_shdr, l_hip, l_knee)
        
        # Bên Phải
        r_shdr = [landmarks[12].x, landmarks[12].y]
        r_hip = [landmarks[24].x, landmarks[24].y]
        r_knee = [landmarks[26].x, landmarks[26].y]
        angle_r = calculate_angle(r_shdr, r_hip, r_knee)

        # --- LOGIC ĐẾM ---
        # Đứng thẳng: Cả 2 bên đều > 170 độ
        if angle_l > 170 and angle_r > 170:
            # Nếu trước đó đã nghiêng (left/right) thì mới đếm
            if self.stage in ["left", "right"]:
                self.counter += 1
                self.feedback = "Tốt!"
            self.stage = "center"

        # Nghiêng Trái: Góc trái co lại < 155 độ
        elif angle_l < 155:
            self.stage = "left"
            self.feedback = "Nghiêng tốt"
            
        # Nghiêng Phải: Góc phải co lại < 155 độ
        elif angle_r < 155:
            self.stage = "right"
            self.feedback = "Nghiêng tốt"

        return min(angle_l, angle_r), self.counter, self.feedback, self.stage