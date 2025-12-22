import cv2
import time
from helper import calculate_angle, OneEuroFilter

class JumpingJackExercise:
    def __init__(self):
        self.counter = 0
        self.stage = "CLOSE"
        self.feedback = "Start Jumping"
        
        # Bộ lọc
        self.arm_filter = OneEuroFilter(time.time(), 20)
        self.leg_filter = OneEuroFilter(time.time(), 90)
        
        # --- BIẾN CHO ROM ---
        self.max_arm_detected = 0  # Góc tay lớn nhất trong lần nhảy
        self.max_leg_detected = 0  # Góc chân lớn nhất trong lần nhảy
        
        self.last_arm_rom = 0      # Kết quả ROM tay lần gần nhất
        self.last_leg_rom = 0      # Kết quả ROM chân lần gần nhất

    def process(self, image, landmarks):
        h, w, _ = image.shape
        
        # Lấy điểm mốc
        l_shoulder = landmarks[11]
        l_elbow = landmarks[13]
        l_hip = landmarks[23]
        r_hip = landmarks[24]
        l_knee = landmarks[25]

        # 1. Tính góc thô
        raw_arm = calculate_angle(l_hip, l_shoulder, l_elbow)
        raw_leg = calculate_angle(r_hip, l_hip, l_knee)

        # 2. Lọc mượt
        curr_time = time.time()
        smooth_arm = self.arm_filter(curr_time, raw_arm)
        smooth_leg = self.leg_filter(curr_time, raw_leg)

        # 3. Theo dõi MAX ANGLE (Chỉ theo dõi khi đang nhảy lên)
        if self.stage == "OPEN":
            if smooth_arm > self.max_arm_detected:
                self.max_arm_detected = smooth_arm
            if smooth_leg > self.max_leg_detected:
                self.max_leg_detected = smooth_leg

        # 4. LOGIC STATE MACHINE
        
        # Điều kiện Mở (Nhảy lên - JUMP PHASE)
        if smooth_arm > 150 and smooth_leg > 105:
            if self.stage == "CLOSE":
                self.stage = "OPEN"
                self.counter += 1
                self.feedback = "Good!"
                # Reset biến max để bắt đầu đo cho lần nhảy mới này
                self.max_arm_detected = smooth_arm
                self.max_leg_detected = smooth_leg
        
        # Điều kiện Đóng (Hạ xuống - LAND PHASE)
        # Khi người dùng khép tay chân lại, ta chốt sổ ROM của cú nhảy vừa rồi
        if smooth_arm < 45 and smooth_leg < 100:
            if self.stage == "OPEN":
                # TÍNH ROM KHI HOÀN THÀNH
                # Trừ đi góc tự nhiên (tay ~15 độ, chân ~90 độ)
                self.last_arm_rom = self.max_arm_detected - 15 
                self.last_leg_rom = self.max_leg_detected - 90
                
            self.stage = "CLOSE"
            self.feedback = "Jump Up!"

        # 5. HIỂN THỊ UI
        # Vẽ góc hiện tại (Real-time)
        cv2.putText(image, f"Arm: {int(smooth_arm)}", (20, h - 140), cv2.FONT_HERSHEY_PLAIN, 1.5, (200, 200, 200), 2)
        cv2.putText(image, f"Leg: {int(smooth_leg)}", (20, h - 110), cv2.FONT_HERSHEY_PLAIN, 1.5, (200, 200, 200), 2)

        # Vẽ ROM (Kết quả cú nhảy trước) - Màu vàng nổi bật
        cv2.putText(image, f"ROM Tay: {int(self.last_arm_rom)}", (w - 200, h - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(image, f"ROM Chan: {int(self.last_leg_rom)}", (w - 200, h - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Thanh Bar hiển thị độ mở tay (cho đẹp)
        # Max mở tay tầm 170 độ -> mapping ra thanh 100px
        bar_len = int((smooth_arm / 180) * 100)
        cv2.rectangle(image, (20, h-40), (20+bar_len, h-30), (0, 255, 0), -1)
        cv2.rectangle(image, (20, h-40), (120, h-30), (255, 255, 255), 1)

        return image, self.counter, self.stage, self.feedback, smooth_arm