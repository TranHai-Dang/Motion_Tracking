import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- IMPORT CÁC BÀI TẬP ---
try:
    from WarmUp.jumpingjack import JumpingJackExercise
    from WarmUp.sidebend import SideBendExercise
    from Exercise.squat import SquatExercise
    from Exercise.pushup import PushUpExercise
    from Challenge.plank import PlankExercise
    from Challenge.highknees import HighKneesExercise
except ImportError as e:
    st.error(f"❌ Lỗi Import: {e}")
    st.stop()

# --- CLASS XỬ LÝ HÌNH ẢNH ---
class PoseProcessor(VideoTransformerBase):
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.exercise = None 

    def set_exercise(self, exercise_class):
        if exercise_class:
            self.exercise = exercise_class()
            self.exercise.reset()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        
        info_text = "Waiting..."
        status_color = (255, 255, 255)

        if results.pose_landmarks and self.exercise:
            try:
                angle, count, feedback, stage = self.exercise.process(results.pose_landmarks.landmark)
                info_text = f"Count: {count} | {stage} | {feedback}"
                if "Good" in feedback or "Hold" in feedback: status_color = (0, 255, 0)
                elif "FIX" in feedback or "Lower" in feedback: status_color = (0, 0, 255)
                self.mp_drawing.draw_landmarks(img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            except Exception as e:
                info_text = f"Error: {e}"

        cv2.rectangle(img, (0,0), (640, 60), (245, 117, 16), -1)
        cv2.putText(img, info_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
        return img

# --- GIAO DIỆN CHÍNH (STREAMLIT UI) ---
def main():
    st.set_page_config(page_title="Virtual Rehab AI", layout="wide")
    
    # --- CHÈN CSS ĐỂ PHÓNG TO CAMERA ---
    st.markdown(
        """
        <style>
        video {
            width: 100% !important;
            height: auto !important;
            border-radius: 10px; 
            box-shadow: 0 4px 8px rgba(0,0,0,0.5);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🏋️ Virtual Rehab - AI Trainer")

    CLASS_MAP = {
        "Jumping Jack": JumpingJackExercise,
        "Side Bend": SideBendExercise,
        "Squat": SquatExercise,
        "Push Up": PushUpExercise,
        "Plank": PlankExercise,
        "High Knees": HighKneesExercise
    }

    MENU_STRUCTURE = {
        "Warm Up (Khởi động)": ["Jumping Jack", "Side Bend"],
        "Training (Tập luyện)": ["Squat", "Push Up"],
        "Challenge (Thử thách)": ["Plank", "High Knees"]
    }

    # Chia cột: Cột 1 nhỏ (Menu), Cột 2 to (Camera)
    col1, col2 = st.columns([1, 4]) # Đổi tỉ lệ từ 1:3 sang 1:4 để cam to hơn nữa

    with col1:
        st.header("🎛️ Menu")
        selected_mode = st.selectbox("Chọn Chế Độ:", list(MENU_STRUCTURE.keys()))
        available_exercises = MENU_STRUCTURE[selected_mode]
        selected_exercise_name = st.selectbox("Chọn Bài Tập:", available_exercises)
        
        st.info(f"👉 Bạn đang chọn: **{selected_exercise_name}**")
        st.warning("💡 Đứng cách Camera 2-3 mét.")
        
        current_exercise_class = CLASS_MAP.get(selected_exercise_name)

    with col2:
        st.header(f"🎥 Camera: {selected_exercise_name}")
        
        ctx = webrtc_streamer(
            key="rehab-cam",
            video_processor_factory=PoseProcessor,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        if ctx.video_processor and current_exercise_class:
            ctx.video_processor.set_exercise(current_exercise_class)

if __name__ == "__main__":
    main()