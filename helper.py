import cv2
import threading
import time
import math
import numpy as np

# --- 1. CLASS CAMERA ĐA LUỒNG (Không Delay) ---
class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened(): continue
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

# --- 2. BỘ LỌC ONE EURO (Chống Rung) ---
class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=0.01, beta=0.1):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.x_prev = float(x0)
        self.dx_prev = 0.0
        self.t_prev = float(t0)

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0: return self.x_prev
        
        # Hàm mũ làm mượt
        a_d = self.smoothing_factor(t_e, 1.0)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

# --- 3. HÀM TÍNH GÓC (Chuẩn hóa 0-180) ---
def calculate_angle(a, b, c):
    """Tính góc tại điểm b (a-b-c)"""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0: angle = 360 - angle
    return angle