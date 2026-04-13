import sys
import os
import cv2
import numpy as np

# Thêm đường dẫn hệ thống
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r'C:\tritue\project-Emojify')

from FaceProcessor import FaceProcessor
from Utils.Drawing import draw_face_boundary

# Khởi tạo biến toàn cục cho FPS
prev_time = cv2.getTickCount()
frame_count = 0

def get_feature_color(name):
    """Trả về màu sắc BGR đặc trưng cho từng bộ phận"""
    colors = {
        'mui': (0, 255, 0),        # Xanh lá
        'mieng': (0, 165, 255),    # Cam
        'mat': (255, 0, 0),        # Đỏ
        'default': (0, 0, 255)     # Đỏ mặc định
    }
    name_low = name.lower()
    for key, color in colors.items():
        if key in name_low:
            return color
    return colors['default']

def draw_fps(frame):
    """Tính toán và hiển thị FPS lên màn hình"""
    global prev_time, frame_count
    frame_count += 1
    
    if frame_count % 15 == 0:
        current_time = cv2.getTickCount()
        freq = cv2.getTickFrequency()
        fps = freq / (current_time - prev_time)
        prev_time = current_time
        # Vẽ FPS (Chữ xanh lá viền đen cho dễ nhìn)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame

def run_camera():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Lỗi: Không thể mở camera")
        return
    
    processor = FaceProcessor()
    # Tải bộ lọc nhận diện khuôn mặt
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if face_cascade.empty():
        print("Loi: Khong load duoc file nhan dien")
        return

    print("Đang chạy... Nhấn 'q' để thoát, 's' để chụp ảnh")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Vẽ khung khuôn mặt
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        frame = draw_face_boundary(frame, faces)

        # 2. Xử lý chi tiết các điểm mốc
        coords = processor.get_features(frame)
        if coords:
            for name, pos in coords.items():
                x, y = pos
                color = get_feature_color(name)
                
                # Vẽ điểm mốc (Point)
                cv2.circle(frame, (x, y), 4, color, -1)
                cv2.circle(frame, (x, y), 5, (0, 0, 0), 1)
                
                # Tạo nội dung: "Ten (X, Y)"
                label = f"{name.replace('_', ' ').title()} ({x}, {y})"
                
                # Vẽ Text với hiệu ứng Shadow để nổi bật trên mọi màu nền
                # Lớp nền đen (offset 1px)
                cv2.putText(frame, label, (x + 10, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                # Lớp chữ trắng chính
                cv2.putText(frame, label, (x + 10, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 3. Hiển thị FPS và Window
        frame = draw_fps(frame)
        cv2.imshow("Emojify Tracker", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"screenshot_{int(cv2.getTickCount())}.jpg"
            if cv2.imwrite(filename, frame):
                print(f"Đã lưu tại: {filename}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera()