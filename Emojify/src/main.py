import cv2
import numpy as np
import os
from utils import apply_histogram_equalization

# ====================== ĐƯỜNG DẪN======================
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(base_dir, "model", "res10_300x300_ssd.caffemodel")
PROTO_PATH = os.path.join(base_dir, "model", "deploy.prototxt")

if not os.path.exists(MODEL_PATH):
    print(f"Khang tim thay model: {MODEL_PATH}")
    exit()

if not os.path.exists(PROTO_PATH):
    print(f"Khong tim thay prototxt: {PROTO_PATH}")
    exit()

print("Da load model Face Detection thanh cong!")

# Load model
net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)

# Mở camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Khong mo duoc camera!")
    exit()

print("Camera dang chay - Nhan ESC hoac X de thoat...")

window_name = "Face Detection - Emojify Project"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1200, 800)

# Callback function để xử lý khi đóng cửa sổ
def on_window_close():
    print("Da dong bang nut X -> Thoat chuong trinh")
    cap.release()
    cv2.destroyAllWindows()
    os._exit(0)

# Gán callback cho sự kiện đóng cửa sổ (Windows)
cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)  # Đảm bảo window luôn ở trên cùng

while True:
    ret, frame = cap.read()
    if not ret:
        print("Khong nhan duoc frame.")
        break

    (h, w) = frame.shape[:2]

    try:
        enhanced_frame = apply_histogram_equalization(frame)
    except:
        enhanced_frame = frame.copy()

    blob = cv2.dnn.blobFromImage(
        cv2.resize(enhanced_frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w, endX), min(h, endY)

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            text = f"{confidence * 100:.1f}%"
            cv2.putText(frame, text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    frame = cv2.resize(frame, (1200, 800))

    # Kiểm tra cửa sổ có còn tồn tại không
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        print("Da dong bang nut X -> Thoat chuong trinh")
        break  # Thoát vòng lặp thay vì os._exit()

    # Hiển thị
    try:
        cv2.imshow(window_name, frame)
    except cv2.error:
        print("Window da bi dong.")
        break

    # ESC để thoát
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

# Cleanup an toàn
cap.release()
cv2.destroyAllWindows()
print("Chuong trinh da ket thuc.")