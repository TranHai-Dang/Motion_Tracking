import numpy as np

def calculate_angle(a, b, c):
    """
    Tính góc giữa 3 điểm a, b, c (trong đó b là đỉnh góc).
    Đầu vào: a, b, c là các list hoặc array [x, y].
    Đầu ra: Góc (độ) từ 0 đến 180.
    """
    a = np.array(a) # Điểm đầu
    b = np.array(b) # Đỉnh
    c = np.array(c) # Điểm cuối
    
    # Tính góc bằng arctan2
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle