import cv2

def draw_face_boundary(image, faces):
    """
    Vẽ khung xanh quanh khuôn mặt
    Args:
        image: frame BGR từ OpenCV
        faces: list [(x,y,w,h)] từ detectMultiScale
    Returns:
        image đã vẽ khung
    """
    for (x, y, w, h) in faces:
        # Khung xanh dày 3px, bo góc
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Text "KHUÔN MẶT" góc trên (bonus)
        cv2.putText(image, "KHUON MAT", (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Đường chéo góc (đẹp hơn)
        "cv2.line(image, (x, y), (x + w, y + h), (0, 255, 0), 2)"
        "cv2.line(image, (x + w, y), (x, y + h), (0, 255, 0), 2)"
    
    return image