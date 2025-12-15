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
    "Jumping Jack": "1. Đứng thẳng, hai chân khép, tay xuôi theo thân.\n\n2. Bật nhảy, dang hai chân rộng hơn vai, vung tay lên cao đập vào nhau.\n\n3. Bật nhảy trở về tư thế ban đầu.\n\n👉 *Mẹo: Giữ nhịp thở đều, tiếp đất bằng mũi chân.*",
    "Side Bend": "1. Đứng thẳng, hai chân rộng bằng vai.\n\n2. Nghiêng lườn sang trái sâu hết mức có thể.\n\n3. Trở về giữa rồi nghiêng sang phải.\n\n👉 *Mẹo: Không cúi người về trước, chỉ nghiêng sang ngang.*",
    "Squat": "1. Đứng thẳng, chân rộng bằng vai.\n\n2. Hạ hông xuống như đang ngồi trên ghế (đùi song song sàn).\n\n3. Đứng thẳng dậy.\n\n👉 *Mẹo: Giữ lưng thẳng, đầu gối không vượt quá mũi chân.*",
    "Push Up": "1. Chống tay xuống sàn, thân người thẳng.\n\n2. Hạ ngực xuống gần chạm sàn.\n\n3. Đẩy người lên thẳng tay.\n\n👉 *Mẹo: Gồng bụng, không để võng lưng.*",
    "Plank": "1. Chống khuỷu tay xuống sàn.\n\n2. Giữ người thẳng tắp, gồng chặt bụng.\n\n3. Giữ nguyên tư thế càng lâu càng tốt.\n\n👉 *Mẹo: Đừng đẩy mông quá cao hoặc để lưng bị võng.*",
    "High Knees": "1. Chạy tại chỗ.\n\n2. Nâng đùi cao vuông góc với thân người.\n\n3. Đánh tay mạnh theo nhịp.\n\n👉 *Mẹo: Cố gắng nâng đùi càng cao càng tốt.*"
}

# --- 3. CLASS XỬ LÝ AI ---
class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.exercise = None 
        self.flip = True  
        self.rotate_type = "Không xoay"
        
        # Lưu lịch sử
        self.total_reps = 0
        self.error_log = [] 

    def set_exercise(self, exercise_class):
        if exercise_class:
            self.exercise = exercise_class()
            self.exercise.reset()
            self.total_reps = 0
            self.error_log = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
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
            
            status_color = (0, 165, 255) # Cam
            info_text = "AI Ready..."
            
            # Khung thông tin trên video
            h, w, _ = img.shape
            
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                
                if self.exercise:
                    try:
                        angle, count, feedback, stage = self.exercise.process(results.pose_landmarks.landmark)
                        
                        # Cập nhật Reps & Log
                        self.total_reps = count
                        if feedback and "Good" not in feedback and "Tot" not in feedback and "Start" not in feedback:
                            if not self.error_log or self.error_log[-1] != feedback:
                                self.error_log.append(feedback)

                        info_text = f"Rep: {count} | {feedback}"
                        if "Good" in feedback or "Tot" in feedback: 
                            status_color = (0, 255, 0)
                        elif "FIX" in feedback or "Ha" in feedback: 
                            status_color = (0, 0, 255)
                    except:
                        pass
            
            # 3. Vẽ bảng thông báo TO VÀ RÕ
            # Vẽ nền bán trong suốt ở trên cùng
            overlay = img.copy()
            cv2.rectangle(overlay, (0,0), (w, 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
            
            # Vẽ chữ
            cv2.putText(img, info_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2, cv2.LINE_AA)

            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        except Exception as e:
            print(e)
            return frame

# --- 4. GIAO DIỆN CHÍNH ---
def main():
    st.set_page_config(page_title="Virtual Rehab AI", layout="wide")
    
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1rem; padding-bottom: 1rem; }
        video { width: 100% !important; border-radius: 10px; }
        </style>
        """,
        unsafe_allow_html=True
    )

    # --- SIDEBAR ---
    with st.sidebar:
        st.title("🎛️ Bảng Điều Khiển")
        
        CLASS_MAP = {
            "Jumping Jack": JumpingJackExercise,
            "Side Bend": SideBendExercise,
            "Squat": SquatExercise,
            "Push Up": PushUpExercise,
            "Plank": PlankExercise,
            "High Knees": HighKneesExercise
        }
        
        MENU = {
            "Khởi động": ["Jumping Jack", "Side Bend"],
            "Tập luyện": ["Squat", "Push Up"],
            "Thử thách": ["Plank", "High Knees"]
        }

        st.subheader("1. Chọn Bài Tập")
        mode = st.selectbox("Chế độ:", list(MENU.keys()))
        exercise_name = st.selectbox("Bài tập:", MENU[mode])
        current_exercise = CLASS_MAP[exercise_name]

        st.markdown("---")
        st.subheader("📖 Hướng Dẫn")
        st.info(GUIDE_VIETNAMESE.get(exercise_name, ""))

        st.markdown("---")
        st.subheader("📷 Cài đặt Camera")
        flip = st.checkbox("Lật gương", value=True)
        rotate = st.radio("Xoay:", ("Không xoay", "Xoay trái 90°", "Xoay phải 90°"))

    # --- MÀN HÌNH CHÍNH ---
    st.title(f"🏋️ {exercise_name}")
    
    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    ctx = webrtc_streamer(
        key="rehab-cam",
        video_processor_factory=PoseProcessor,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=False 
    )

    if ctx.video_processor:
        ctx.video_processor.set_exercise(current_exercise)
        ctx.video_processor.flip = flip
        ctx.video_processor.rotate_type = rotate

    # --- PHẦN KẾT QUẢ (HIỆN KHI DỪNG CAMERA) ---
    st.markdown("---")
    
    # Tạo một vùng chứa kết quả để người dùng biết nó nằm ở đâu
    result_container = st.container()
    
    with result_container:
        if ctx.state.playing:
            st.info("🔴 Đang tập luyện... Bấm 'STOP' (hoặc tắt Camera) để xem báo cáo chi tiết.")
        
        # Logic hiện báo cáo
        if not ctx.state.playing and ctx.video_processor:
            processor = ctx.video_processor
            
            # Chỉ hiện nếu đã tập ít nhất 1 cái hoặc có lỗi (tránh hiện khi vừa vào web)
            if processor.total_reps > 0 or len(processor.error_log) > 0:
                st.subheader("📊 Kết Quả Buổi Tập Vừa Rồi")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Tổng Reps", processor.total_reps)
                with col2:
                    error_count = len(processor.error_log)
                    score = max(0, 100 - (error_count * 5))
                    
                    if score >= 80: grade, color = "Xuất sắc 🏆", "normal"
                    elif score >= 50: grade, color = "Khá 👍", "off"
                    else: grade, color = "Cần cố gắng ⚠️", "inverse"
                    
                    st.metric("Điểm Kỹ Thuật", f"{score}/100", grade)

                if processor.error_log:
                    st.warning("🧐 Các lỗi cần khắc phục:")
                    from collections import Counter
                    counts = Counter(processor.error_log)
                    for err, c in counts.items():
                        st.write(f"- {err}: {c} lần")
                else:
                    st.success("🎉 Bạn tập rất chuẩn! Không có lỗi nào.")
            
            # Nếu chưa tập gì cả (lần đầu vào web hoặc vừa F5)
            elif processor.total_reps == 0:
                st.caption("👈 Bấm START để bắt đầu tập luyện. Kết quả sẽ hiện ở đây.")

if __name__ == "__main__":
    main()