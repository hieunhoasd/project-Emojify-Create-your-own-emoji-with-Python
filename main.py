import sys
import os
import cv2
import json
import numpy as np
from collections import deque, Counter

# Đảm bảo import được các module từ thư mục hiện tại
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from Core.FaceProcessor import FaceProcessor
from Utils.Drawing import draw_face_boundary
from Core.EmotionAndFeatureScanner import EmotionAndFeatureScanner

prev_time = cv2.getTickCount()
frame_count = 0
emotion_history = deque(maxlen=10)

EVALUATIONS = {
    'Happy': {
        'Vibe': 'Năng lượng rực rỡ, tích cực',
        'Style': 'Thân thiện, tỏa sáng',
        'Nhan_xet': 'Khuôn mặt toát lên vẻ phú quý khi cười, duy trì phát huy!'
    },
    'Sad': {
        'Vibe': 'Trầm tư, nội tâm',
        'Style': 'Deep, lạnh lùng',
        'Nhan_xet': 'Đang có tâm sự đúng không? Cười lên cái xem nào.'
    },
    'Angry': {
        'Vibe': 'Quyền lực, sát khí',
        'Style': 'Khó gần, nghiêm nghị',
        'Nhan_xet': 'Đang quạu à bro? Hạ hỏa đi cho đời thêm tươi!'
    },
    'Neutral': {
        'Vibe': 'Điềm tĩnh, thanh lịch',
        'Style': 'Tối giản, nghiêm túc',
        'Nhan_xet': 'Bình thản tựa mặt hồ mùa thu. Rất chuẩn form đi phỏng vấn!'
    }
}

def draw_fps(frame):
    global prev_time, frame_count
    frame_count += 1
    if frame_count % 15 == 0:
        current_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame

def run_camera():
    global frame_count
    
    cap = cv2.VideoCapture(0)
    processor = FaceProcessor()
    
    # Load model haarcascade an toàn
    haarcascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haarcascade_path)
    
    # FIX ĐƯỜNG DẪN: Lấy file .h5 cùng thư mục với file main.py
    model_path = os.path.join(BASE_DIR, 'Emojify', 'model', 'emotion_model_final.h5')
    
    print(f"Đang tải AI từ: {model_path}")
    scanner = EmotionAndFeatureScanner(model_path)
    
    print("Hệ thống AI đã khởi động... Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
        frame = draw_face_boundary(frame, faces)

        coords = processor.get_features(frame)
        
        if len(faces) > 0 and coords:
            x, y, w, h = faces[0]
            face_roi = frame[y:y+h, x:x+w]
        
            if face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
                ngu_quan_info = scanner.analyze_ngu_quan(coords, w, h)
                raw_emotion = scanner.predict_emotion(face_roi)
                
                emotion_history.append(raw_emotion)
                smoothed_emotion = Counter(emotion_history).most_common(1)[0][0]
                
                if frame_count % 30 == 0:
                    danh_gia = EVALUATIONS.get(smoothed_emotion, EVALUATIONS['Neutral'])
                    
                    output_data = {
                        "cam_xuc": smoothed_emotion,
                        "ngu_quan": ngu_quan_info,
                        "danh_gia": {
                            "Vibe": danh_gia['Vibe'],
                            "Style": danh_gia['Style'],
                            "Nhan_xet": danh_gia['Nhan_xet']
                        }
                    }
                    
                    json_result = json.dumps(output_data, ensure_ascii=False, indent=4)
                    print("\n" + "="*30)
                    print(json_result)
                    print("="*30)

        # Hiển thị
        frame = draw_fps(frame)
        cv2.imshow("Nhan Tuong Hoc AI", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera()