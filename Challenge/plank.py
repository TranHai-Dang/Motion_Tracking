import numpy as np
import time
from utils import calculate_angle

class PlankExercise:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = None
        self.current_duration = 0
        self.is_holding = False
        self.feedback = "Vào tư thế Plank"

    def process(self, landmarks):
        # Kiểm tra độ thẳng lưng: Vai - Hông - Gối
        l_shdr = [landmarks[11].x, landmarks[11].y]
        l_hip = [landmarks[23].x, landmarks[23].y]
        l_knee = [landmarks[25].x, landmarks[25].y]
        
        angle = calculate_angle(l_shdr, l_hip, l_knee)

        # --- LOGIC TÍNH GIỜ ---
        # Lưng thẳng trong khoảng cho phép (170 - 185 độ)
        if 165 < angle < 185:
            if not self.is_holding:
                self.start_time = time.time()
                self.is_holding = True
                self.feedback = "Giữ nguyên! Good!"
            
            # Tính thời gian đã trôi qua
            self.current_duration = int(time.time() - self.start_time)
            return angle, self.current_duration, "Giữ vững... Good Job!", "Hold"
            
        else:
            # Nếu sai tư thế (Mông quá cao hoặc võng lưng)
            self.is_holding = False
            self.start_time = None
            
            if angle < 165:
                self.feedback = "Hạ thấp mông xuống!"
            elif angle > 185:
                self.feedback = "Nâng mông lên (Đừng võng lưng)!"
                
            return angle, self.current_duration, self.feedback, "FIX FORM"