# 🏋️ Virtual Rehab - AI Personal Trainer

**Ứng dụng hỗ trợ tập luyện thể dục tại nhà sử dụng công nghệ AI (Computer Vision) để đếm số lần tập (Reps) và chỉnh sửa tư thế (Posture Correction) theo thời gian thực.**

---

## ✨ Tính Năng Nổi Bật (Features)

* **🎯 Đa dạng chế độ tập:**
    * **Warm Up (Khởi động):** Jumping Jack, Side Bend.
    * **Training (Tập luyện):** Squat, Push Up (Hít đất).
    * **Challenge (Thử thách):** Plank (tính giây), High Knees (Nâng cao đùi).
* **🤖 AI Thông Minh:**
    * Tự động đếm số lần tập (Rep counter).
    * **Chống đếm ảo (Anti-Ghost Rep):** Chỉ đếm khi thực hiện đúng biên độ (xuống sâu/lên thẳng).
    * Cảnh báo sai tư thế bằng giọng nói/văn bản (VD: "Hạ thấp hông xuống", "Đừng võng lưng").
* **🇻🇳 Giao diện thân thiện:**
    * Hướng dẫn chi tiết từng bài tập bằng **Tiếng Việt**.
    * Tự động hiển thị hướng dẫn khi chọn bài.
* **📷 Tùy chỉnh Camera:**
    * Hỗ trợ **Lật gương (Mirror)**.
    * Hỗ trợ **Xoay 90°/180°** (Dành cho ai dùng điện thoại làm Webcam).

---

## 🛠 Hướng Dẫn Cài Đặt (Installation)

⚠ **Lưu ý quan trọng:** Dự án này hoạt động tốt nhất trên **Python 3.11**. Các phiên bản mới hơn (3.12, 3.13) có thể gây lỗi thư viện MediaPipe.

### 1. Clone dự án về máy
Mở Terminal (hoặc CMD/PowerShell) và chạy lệnh:
```bash
git clone https://github.com/TranHai-Dang/Motion_Tracking.git
cd motion_tracking
```

### 2. Cài đặt thư viện cho Python 3.11
Để đảm bảo thư viện được cài đúng vào Python 3.11 (tránh cài nhầm vào bản khác), hãy dùng lệnh sau:

* **Đối với Windows:**
```bash
py -3.11 -m pip install -r requirements.txt
```

* **Đối với Mac/Linux:**
```bash
python3.11 -m pip install -r requirements.txt
```

### 3. Chạy ứng dụng
Khởi động ứng dụng bằng lệnh:
```bash
py -3.11 -m streamlit run app.py
```

---

## 📂 Cấu Trúc Thư Mục
```text
Motion_Tracking/
├── app.py                # File chính chạy ứng dụng (Giao diện & Logic)
├── requirements.txt      # Danh sách thư viện Python
├── packages.txt          # Danh sách thư viện Linux (Fix lỗi libGL trên Cloud)
├── .python-version       # Ép buộc Streamlit Cloud dùng Python 3.11
├── .gitignore            # Loại bỏ file rác
├── utils.py              # Hàm phụ trợ (Tính góc)
├── WarmUp/               # Chứa class bài tập khởi động
├── Exercise/             # Chứa class bài tập chính
└── Challenge/            # Chứa class bài thử thách
```

---

## ☁️ Hướng Dẫn Deploy (Streamlit Cloud)

Để đưa ứng dụng lên mạng, đảm bảo bạn có đủ 2 file quan trọng này trên GitHub để tránh lỗi:

1.  **`.python-version`**:
    ```text
    3.11
    ```
2.  **`packages.txt`** 
    ```text
    libgl1
    libgl1-mesa-glx
    libglib2.0-0
    libsm6
    libxrender1
    libxext6
    ```

---
**Developed by [Đăng]** 🚀