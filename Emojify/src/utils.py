import cv2
import numpy as np

IMG_SIZE = 48

def apply_histogram_equalization(image):
    yuv_img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv_img[:, :, 0] = cv2.equalizeHist(yuv_img[:, :, 0])
    return cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)

def preprocess_face(face_image, is_training=False):
    try:
        if not is_training:
            face_image = apply_histogram_equalization(face_image)
            
        gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        resized_face = cv2.resize(gray_face, (IMG_SIZE, IMG_SIZE))
        normalized_face = resized_face.astype('float32') / 255.0
        
        final_face = np.expand_dims(normalized_face, axis=0)
        final_face = np.expand_dims(final_face, axis=-1)
        
        return final_face
    except:
        return None