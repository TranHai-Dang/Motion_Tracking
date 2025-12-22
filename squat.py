import cv2
import time
from helper import calculate_angle, OneEuroFilter

class SquatExercise:
    def __init__(self):
        self.counter = 0
        self.stage = "UP"
        self.feedback = "Stand Ready"
        self.angle_filter = OneEuroFilter(time.time(), 180)
        
        # --- BIẾN CHO ROM ---
        self.min_angle_detected = 180 # Góc thấp nhất trong 1 rep
        self.current_rom = 0          # Kết quả ROM của rep vừa xong

    def process(self, image, landmarks):
        h, w, _ = image.shape
        
        # Hông(23) - Gối(25) - Cổ chân(27)
        hip = landmarks[23]
        knee = landmarks[25]
        ankle = landmarks[27]

        # 1. Tính góc & Lọc
        raw_angle = calculate_angle(hip, knee, ankle)
        smooth_angle = self.angle_filter(time.time(), raw_angle)

        # 2. Logic ROM & State Machine
        if self.stage == "DOWN":
            # Khi đang ngồi xuống, liên tục cập nhật góc thấp nhất
            if smooth_angle < self.min_angle_detected:
                self.min_angle_detected = smooth_angle

        # --- CHUYỂN TRẠNG THÁI ---
        if smooth_angle > 160: # Đứng lên (UP)
            if self.stage == "DOWN":
                # Vừa hoàn thành 1 Rep -> Tính toán ROM ngay
                # ROM = Góc đứng (180) - Góc thấp nhất đã xuống được
                self.current_rom = 180 - self.min_angle_detected
                
                self.counter += 1
                self.feedback = "Good Job!"
                
                # Reset biến min để chuẩn bị cho Rep sau
                self.min_angle_detected = 180 
                
            self.stage = "UP"
            self.feedback = "Squat Down!"
            
        if smooth_angle < 90 and self.stage == "UP": # Bắt đầu xuống
            self.stage = "DOWN"
            self.feedback = "Lower..."

        # 3. HIỂN THỊ LÊN MÀN HÌNH
        # Vẽ góc hiện tại cạnh đầu gối
        cv2.putText(image, str(int(smooth_angle)), 
                   (int(knee.x * w) + 10, int(knee.y * h)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Vẽ Chỉ số ROM (Góc gập tối đa)
        # Hiển thị ở góc màn hình hoặc ngay cạnh người
        cv2.putText(image, f"Last ROM: {int(self.current_rom)} deg", 
                   (20, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Thanh Bar hiển thị độ sâu trực quan (Visual Feedback)
        # Map góc từ 180->70 tương ứng với 0->100% thanh
        # Nếu đang ở state DOWN thì hiển thị min_angle hiện tại, nếu UP thì hiển thị kết quả
        display_rom = (180 - self.min_angle_detected) if self.stage == "DOWN" else self.current_rom
        
        # Vẽ thanh năng lượng ROM (nhỏ nhỏ bên cạnh)
        bar_h = int((display_rom / 110) * 100) # Giả sử max ROM là 110 độ
        cv2.rectangle(image, (w-30, h-150), (w-10, h-50), (100, 100, 100), 1)
        cv2.rectangle(image, (w-30, h-50-bar_h), (w-10, h-50), (0, 255, 255), -1)
        cv2.putText(image, "ROM", (w-45, h-30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

        return image, self.counter, self.stage, self.feedback, smooth_angle