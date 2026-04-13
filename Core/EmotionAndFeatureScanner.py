import cv2
import numpy as np
from tensorflow.keras.models import load_model

class EmotionAndFeatureScanner:
    def __init__(self, model_path):
       
        self.model = load_model(model_path)
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def preprocess_face(self, face_roi):
        
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 48, 48, 1))
        return reshaped

    def predict_emotion(self, face_roi):
        try:
            # 1. Chuyển sang ảnh xám (Grayscale)
            if len(face_roi.shape) == 3:
                gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_roi
                
            # 2. Resize đúng về 48x48 (kích thước chuẩn của model FER)
            resized = cv2.resize(gray, (48, 48))
            
            # 3. CHUẨN HÓA DỮ LIỆU (Thủ phạm gây kẹt "Happy" thường là do thiếu dòng này)
            normalized = resized / 255.0
            
            # 4. Reshape về đúng form Keras (1 batch, 48 height, 48 width, 1 channel)
            reshaped = np.reshape(normalized, (1, 48, 48, 1))
            
            # 5. Dự đoán
            result = self.model.predict(reshaped, verbose=0)
            
            # 6. Map kết quả đúng chuẩn bộ label (Tùy biến nếu model của bro train thứ tự khác)
            labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            max_index = int(np.argmax(result))
            
            return labels[max_index]
        
        except Exception as e:
            # Nếu khung hình lỗi đột xuất, trả về Neutral cho an toàn
            return 'Neutral'

    def analyze_ngu_quan(self, coords, face_w, face_h):
        """BƯỚC 4: Tính toán đặc trưng từ Landmark"""
        
        y_mat = (coords['mat_trai'][1] + coords['mat_phai'][1]) / 2
        khoang_cach_mui_mat = coords['mui'][1] - y_mat
        

        ty_le_mui = khoang_cach_mui_mat / face_h
        mui_txt = "mũi cao" if ty_le_mui < 0.15 else "mũi thấp"
        
       
        khoang_cach_mat = abs(coords['mat_phai'][0] - coords['mat_trai'][0])
        mat_txt = "mắt cân đối" if 0.3 < (khoang_cach_mat / face_w) < 0.5 else "khoảng cách mắt rộng"
        
        return f"{mat_txt}... {mui_txt}"