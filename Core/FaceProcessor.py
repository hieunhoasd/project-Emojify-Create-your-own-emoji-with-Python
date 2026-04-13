import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request

class FaceProcessor:
    def __init__(self):
        # Đường dẫn model (.task)
        model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

        # [BỔ SUNG AN TOÀN] - Tự động tải file nếu máy chưa có
        if not os.path.exists(model_path):
            print("Đang tải file AI Face Landmarker (vài giây)...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("Tải xong! Hệ thống sẵn sàng.")

        # Base options
        base_options = python.BaseOptions(model_asset_path=model_path)

        # FaceLandmarker options
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )

        # Create landmarker
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def get_features(self, frame):
        # Convert sang RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert sang mediapipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect
        result = self.detector.detect(mp_image)

        ih, iw, _ = frame.shape
        data = {}

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            def get_xy(idx):
                pt = landmarks[idx]
                return (int(pt.x * iw), int(pt.y * ih))

            data['mui'] = get_xy(1)
            data['mieng'] = get_xy(13)
            data['mat_trai'] = get_xy(33)
            data['mat_phai'] = get_xy(263)

        return data