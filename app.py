import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import av
import time
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

# --- 3. CLASS XỬ LÝ AI (Nâng cấp: Có trí nhớ) ---
class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.exercise = None 
        self.flip = True  
        self.rotate_type = "Không xoay"
        
        # --- BIẾN LƯU TRỮ LỊCH SỬ TẬP ---
        self.total_reps = 0
        self.error_log = [] # Lưu danh sách lỗi (VD: ["Lưng cong", "Chưa xuống sâu"])

    def set_exercise(self, exercise_class):
        if exercise_class:
            self.exercise = exercise_class()
            self.exercise.reset()
            # Reset lịch sử khi đổi bài
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

            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                
                if self.exercise:
                    try:
                        angle, count, feedback, stage = self.exercise.process(results.pose_landmarks.landmark)
                        
                        # Cập nhật Reps
                        self.total_reps = count
                        
                        # Ghi nhớ lỗi (Nếu feedback không phải "Good" và chưa có trong log gần nhất)
                        if feedback and "Good" not in feedback and "Tot" not in feedback and "Start" not in feedback:
                             # Chỉ lưu lỗi nếu nó không bị trùng lặp liên tục (tránh spam)
                            if not self.error_log or self.error_log[-1] != feedback:
                                self.error_log.append(feedback)

                        info_text = f"Rep: {count} | {feedback}"
                        if "Good" in feedback or "Tot" in feedback: 
                            status_color = (0, 255, 0)
                        elif "FIX" in feedback or "Ha" in feedback: 
                            status_color = (0, 0, 255)
                    except:
                        pass
            
            # 3. Vẽ thông báo
            cv2.rectangle(img, (0,0), (img.shape[1], 60), (50, 50, 50), -1)
            cv2.putText(img, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)

            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        except Exception as e:
            print(e)
            return frame

# --- 4. GIAO DIỆN CHÍNH ---
def main():
    st.set_page_config(page_title="Virtual Rehab AI", layout="wide")
    
    # CSS Full màn hình
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1rem; padding-bottom: 1rem; }
        video { width: 100% !important; height: auto !important; border-radius: 10px; }
        div[class*="stWebrtc"] { width: 100% !important; }
        div[class*="stWebrtc"] > div { width: 100% !important; }
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

    # Cấu hình để lưu video
    # media_stream_recorder=True giúp hiện nút "Record" trên video
    ctx = webrtc_streamer(
        key="rehab-cam",
        video_processor_factory=PoseProcessor,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=False
    )

    # Xử lý thông số gửi vào AI
    if ctx.video_processor:
        ctx.video_processor.set_exercise(current_exercise)
        ctx.video_processor.flip = flip
        ctx.video_processor.rotate_type = rotate

    # --- PHẦN BÁO CÁO KẾT QUẢ (REPORT) ---
    # Khi người dùng tắt camera hoặc dừng tập, hiển thị kết quả
    if not ctx.state.playing and ctx.video_processor:
        processor = ctx.video_processor
        if processor.total_reps > 0 or len(processor.error_log) > 0:
            st.divider()
            st.subheader("📊 Báo Cáo Buổi Tập")
            
            col_rep, col_score = st.columns(2)
            
            # 1. Số Reps
            with col_rep:
                st.metric(label="Tổng số lần tập (Reps)", value=processor.total_reps)
            
            # 2. Chấm điểm (Giả lập: Càng ít lỗi điểm càng cao)
            with col_score:
                error_count = len(processor.error_log)
                score = max(0, 100 - (error_count * 5)) # Mỗi lỗi trừ 5 điểm
                
                if score >= 80:
                    grade = "Xuất sắc 🏆"
                    color = "green"
                elif score >= 50:
                    grade = "Khá 👍"
                    color = "orange"
                else:
                    grade = "Cần cố gắng ⚠️"
                    color = "red"
                    
                st.metric(label="Điểm Tư Thế", value=f"{score}/100", delta=grade)

            # 3. Phân tích lỗi
            if processor.error_log:
                st.warning("🧐 Các vấn đề cần cải thiện:")
                # Đếm số lần xuất hiện của từng lỗi
                from collections import Counter
                error_counts = Counter(processor.error_log)
                
                for err, count in error_counts.items():
                    st.write(f"- **{err}**: Lặp lại {count} lần")
            else:
                st.success("🎉 Tuyệt vời! Bạn không mắc lỗi nào.")

if __name__ == "__main__":
    main()