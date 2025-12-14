import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode

# --- 1. IMPORT CÁC BÀI TẬP ---
try:
    from WarmUp.jumpingjack import JumpingJackExercise
    from WarmUp.sidebend import SideBendExercise
    from Exercise.squat import SquatExercise
    from Exercise.pushup import PushUpExercise
    from Challenge.plank import PlankExercise
    from Challenge.highknees import HighKneesExercise
except ImportError as e:
    st.error(f"❌ Lỗi Import: {e}. Hãy kiểm tra lại cấu trúc thư mục.")
    st.stop()

# --- 2. DỮ LIỆU HƯỚNG DẪN ---
GUIDE_VIETNAMESE = {
    "Jumping Jack": """
**🔥 Cách thực hiện:**

1. Đứng thẳng, hai chân khép, tay xuôi theo thân.
2. Bật nhảy, dang hai chân rộng hơn vai, đồng thời vung hai tay lên cao qua đầu đập vào nhau.
3. Bật nhảy trở về tư thế ban đầu.

👉 *Mẹo: Giữ nhịp thở đều, tiếp đất bằng mũi chân.*
    """,
    "Side Bend": """
**🔥 Cách thực hiện:**

1. Đứng thẳng, hai chân rộng bằng vai, tay để dọc thân hoặc sau gáy.
2. Nghiêng lườn sang trái sâu hết mức có thể.
3. Trở về giữa rồi nghiêng sang phải.

👉 *Mẹo: Không cúi người về trước, chỉ nghiêng sang ngang.*
    """,
    "Squat": """
**🔥 Cách thực hiện:**

1. Đứng thẳng, chân rộng bằng vai.
2. Hạ hông xuống như đang ngồi trên ghế (đùi song song sàn).
3. Đứng thẳng dậy trở về vị trí đầu.

👉 *Mẹo: Giữ lưng thẳng, đầu gối không vượt quá mũi chân.*
    """,
    "Push Up": """
**🔥 Cách thực hiện:**

1. Chống tay xuống sàn, thân người tạo thành đường thẳng.
2. Hạ ngực xuống gần chạm sàn (khuỷu tay gập).
3. Đẩy người lên thẳng tay.

👉 *Mẹo: Gồng bụng, không để võng lưng.*
    """,
    "Plank": """
**🔥 Cách thực hiện:**

1. Chống khuỷu tay xuống sàn, giữ người thẳng tắp.
2. Gồng chặt bụng và giữ nguyên tư thế.

👉 *Mẹo: Đừng đẩy mông quá cao hoặc để lưng bị võng.*
    """,
    "High Knees": """
**🔥 Cách thực hiện:**

1. Chạy tại chỗ.
2. Cố gắng nâng đùi cao vuông góc với thân người.

👉 *Mẹo: Đánh tay mạnh theo nhịp chạy.*
    """
}

# --- 3. CLASS XỬ LÝ HÌNH ẢNH ---
class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.exercise = None 
        self.flip = True  
        self.rotate_type = "Không xoay" 

    def set_exercise(self, exercise_class):
        if exercise_class:
            self.exercise = exercise_class()
            self.exercise.reset()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # 1. Xử lý xoay/lật
        if self.flip:
            img = cv2.flip(img, 1)
            
        if self.rotate_type == "Xoay trái 90°":
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.rotate_type == "Xoay phải 90°":
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotate_type == "Xoay 180°":
            img = cv2.rotate(img, cv2.ROTATE_180)

        # 2. Xử lý AI
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        
        info_text = "San sang..."
        status_color = (255, 255, 255)

        if results.pose_landmarks and self.exercise:
            try:
                angle, count, feedback, stage = self.exercise.process(results.pose_landmarks.landmark)
                info_text = f"Count: {count} | {stage} | {feedback}"
                
                if "Good" in feedback or "Tot" in feedback: 
                    status_color = (0, 255, 0)
                elif "FIX" in feedback or "Ha" in feedback: 
                    status_color = (0, 0, 255)
                
                self.mp_drawing.draw_landmarks(img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            except Exception as e:
                info_text = f"Loi: {e}"

        # 3. Vẽ bảng thông báo
        cv2.rectangle(img, (0,0), (img.shape[1], 80), (245, 117, 16), -1)
        cv2.putText(img, info_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2, cv2.LINE_AA)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. GIAO DIỆN CHÍNH ---
def main():
    st.set_page_config(page_title="Virtual Rehab AI", layout="wide")
    
    st.markdown(
        """
        <style>
        video {
            width: 100% !important;
            border-radius: 10px;
        }
        div.stWebrtc {
            width: 100% !important;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🏋️ Virtual Rehab - Trợ Lý Tập Gym AI")

    CLASS_MAP = {
        "Jumping Jack": JumpingJackExercise,
        "Side Bend": SideBendExercise,
        "Squat": SquatExercise,
        "Push Up": PushUpExercise,
        "Plank": PlankExercise,
        "High Knees": HighKneesExercise
    }

    MENU_STRUCTURE = {
        "Khởi động (Warm Up)": ["Jumping Jack", "Side Bend"],
        "Tập luyện (Training)": ["Squat", "Push Up"],
        "Thử thách (Challenge)": ["Plank", "High Knees"]
    }

    # Sidebar
    st.sidebar.header("📷 Cài đặt Camera")
    flip_cam = st.sidebar.checkbox("Lật gương (Mirror)", value=True)
    rotate_option = st.sidebar.radio(
        "Xoay khung hình:",
        ("Không xoay", "Xoay trái 90°", "Xoay phải 90°", "Xoay 180°")
    )
    st.sidebar.info("💡 Dùng 'Xoay' nếu bạn dùng điện thoại làm Webcam.")

    # Layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🎛️ Chọn Bài Tập")
        selected_mode = st.selectbox("1. Chế độ:", list(MENU_STRUCTURE.keys()))
        available_exercises = MENU_STRUCTURE[selected_mode]
        selected_exercise_name = st.selectbox("2. Bài tập:", available_exercises)
        
        current_exercise_class = CLASS_MAP.get(selected_exercise_name)
        
        st.markdown("---")
        st.subheader(f"📖 Hướng dẫn: {selected_exercise_name}")
        guide_text = GUIDE_VIETNAMESE.get(selected_exercise_name, "Chưa có hướng dẫn.")
        st.info(guide_text)

    with col2:
        st.subheader("🎥 Màn hình AI")
        
        rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        
        ctx = webrtc_streamer(
            key="rehab-cam",
            video_processor_factory=PoseProcessor,
            mode=WebRtcMode.SENDRECV, # <--- QUAN TRỌNG: Sửa thành Enum thay vì string
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        if ctx.video_processor:
            ctx.video_processor.set_exercise(current_exercise_class)
            ctx.video_processor.flip = flip_cam
            ctx.video_processor.rotate_type = rotate_option

    st.markdown("---")
    st.caption("💡 Mẹo: Bấm vào biểu tượng ⛶ (Full Screen) ở góc dưới video để phóng to toàn màn hình.")

if __name__ == "__main__":
    main()