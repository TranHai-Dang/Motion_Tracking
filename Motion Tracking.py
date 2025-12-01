import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pygame
import time
import sys

# --- CẤU HÌNH ---
SKIP_FRAMES = 2
CAMERA_INDEX = 0 # Chỉnh sửa nếu bạn có nhiều camera

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
YELLOW = (255, 255, 0)

# --- KHỞI TẠO HỆ THỐNG ---
pygame.init()

# 1. Tự động lấy độ phân giải màn hình hiện tại
infoObject = pygame.display.Info()
SCREEN_WIDTH = infoObject.current_w
SCREEN_HEIGHT = infoObject.current_h

# 2. Bật chế độ Fullscreen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Move & Smile (Fullscreen)")
clock = pygame.time.Clock()

# Font chữ to hơn cho dễ nhìn
font = pygame.font.SysFont("arial", 40, bold=True)
big_font = pygame.font.SysFont("arial", 80, bold=True)

# Load Model
print("Loading MoveNet...")
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures['serving_default']
print("Ready!")

EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), 
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

# --- HÀM XỬ LÝ ---
def get_movenet_keypoints(frame):
    img = tf.image.resize_with_pad(np.expand_dims(frame, axis=0), 192, 192)
    img = tf.cast(img, dtype=tf.int32)
    return movenet(img)['output_0'].numpy()[0][0]

def draw_skeleton_cv2(frame, keypoints):
    """Vẽ xương bằng OpenCV lên ảnh gốc"""
    h, w, _ = frame.shape
    for edge in EDGES:
        p1, p2 = edge
        y1, x1, c1 = keypoints[p1]
        y2, x2, c2 = keypoints[p2]
        if c1 > 0.3 and c2 > 0.3:
            # Vẽ đường viền đen cho dễ nhìn trên nền camera
            cv2.line(frame, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), BLACK, 5)
            cv2.line(frame, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), YELLOW, 3)
    
    for kp in keypoints:
        y, x, conf = kp
        if conf > 0.3:
            cv2.circle(frame, (int(x*w), int(y*h)), 8, RED, -1)
            cv2.circle(frame, (int(x*w), int(y*h)), 8, WHITE, 2)

def frame_to_fullscreen(frame):
    """Resize ảnh camera cho vừa khít màn hình"""
    # Resize to screen resolution
    frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)
    return pygame.surfarray.make_surface(frame.swapaxes(0,1))

def draw_text_hud(text, x, y, color=WHITE, size="normal", align="center"):
    """Vẽ chữ có viền đen (Shadow) để nổi bật trên nền video"""
    use_font = big_font if size == "big" else font
    
    # Vẽ viền đen (Shadow)
    shadow = use_font.render(text, True, BLACK)
    text_surf = use_font.render(text, True, color)
    
    if align == "center":
        rect_s = shadow.get_rect(center=(x+2, y+2))
        rect_t = text_surf.get_rect(center=(x, y))
    elif align == "left":
        rect_s = shadow.get_rect(topleft=(x+2, y+2))
        rect_t = text_surf.get_rect(topleft=(x, y))
    elif align == "right":
        rect_s = shadow.get_rect(topright=(x+2, y+2))
        rect_t = text_surf.get_rect(topright=(x, y))

    screen.blit(shadow, rect_s)
    screen.blit(text_surf, rect_t)

# --- GAME LOGIC ---
def run_game(game_type, difficulty):
    score = 0
    start_time = time.time()
    game_duration = 45 + (difficulty - 1) * 15  # Tăng thời gian theo độ khó
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened(): return 0
    
    frame_count = 0
    last_keypoints = None

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # AI Processing
        if frame_count % SKIP_FRAMES == 0:
            last_keypoints = get_movenet_keypoints(frame)
        keypoints = last_keypoints
        frame_count += 1
        
        # Vẽ xương lên frame
        draw_skeleton_cv2(frame, keypoints)
        
        # 1. Đưa Camera lên Full màn hình nền
        bg_surf = frame_to_fullscreen(frame)
        screen.blit(bg_surf, (0, 0))

        # 2. Vẽ Giao diện HUD (Đè lên trên Camera)
        
        # Tiêu đề game ở giữa trên cùng
        title = "HAND RAISE" if game_type == "hand" else "TORSO TWIST"
        draw_text_hud(title, SCREEN_WIDTH//2, 50, YELLOW, "big")
        
        # Thời gian (Góc trái trên)
        remain = int(game_duration - (time.time() - start_time))
        color_time = RED if remain < 10 else WHITE
        draw_text_hud(f"TIME: {remain}", 50, 50, color_time, "normal", "left")
        
        # Điểm số (Góc phải trên)
        draw_text_hud(f"SCORE: {score}", SCREEN_WIDTH - 50, 50, WHITE, "normal", "right")

        # Logic Game
        detected = False
        if keypoints is not None:
            if game_type == "hand":
                ls, lw = keypoints[5][0], keypoints[9][0]
                rs, rw = keypoints[6][0], keypoints[10][0]
                detected = (lw < ls - 0.02) and (rw < rs - 0.02)
            else:
                diff = abs(keypoints[5][1] - keypoints[6][1])
                detected = diff > (0.10 + 0.03 * difficulty)

        if detected:
            score += 1 if frame_count % 10 == 0 else 0
            # Hiện chữ GOOD JOB to đùng giữa màn hình
            draw_text_hud("EXCELLENT!", SCREEN_WIDTH//2, SCREEN_HEIGHT//2, GREEN, "big")
        
        # Hướng dẫn thoát
        draw_text_hud("Press Q to Quit", SCREEN_WIDTH//2, SCREEN_HEIGHT - 50, WHITE, "normal")

        if remain <= 0:
            cap.release()
            return score

        # Xử lý sự kiện thoát
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    cap.release()
                    return score

        pygame.display.update()
        clock.tick(30)

def main_menu():
    while True:
        # Màn hình Menu đơn giản (Nền đen)
        screen.fill(BLACK)
        draw_text_hud("MOVE & SMILE", SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 100, YELLOW, "big")
        draw_text_hud("1: Hand Raise Game", SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 50)
        draw_text_hud("2: Torso Twist Game", SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 120)
        draw_text_hud("Q: Quit Application", SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 190)
        
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1: 
                    score = run_game("hand", 1)
                    show_result(score)
                if event.key == pygame.K_2: 
                    score = run_game("torso", 1)
                    show_result(score)
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE: 
                    pygame.quit(); sys.exit()

def show_result(score):
    start_wait = time.time()
    while time.time() - start_wait < 5: # Hiện kết quả trong 5 giây
        screen.fill(BLACK)
        draw_text_hud("TIME'S UP!", SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 50, RED, "big")
        draw_text_hud(f"FINAL SCORE: {score}", SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 50, WHITE, "big")
        pygame.display.update()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN: return

if __name__ == "__main__":
    main_menu()