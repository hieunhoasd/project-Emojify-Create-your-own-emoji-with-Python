# Dự án Emojify - Nhận diện khuôn mặt và Cảm xúc

## Giới thiệu
Chương trình sử dụng Python và OpenCV để nhận diện khuôn mặt thời gian thực, sau đó sử dụng mô hình Deep Learning để phân loại cảm xúc.

## Cấu trúc dự án
- `asset/`: Chứa các biểu tượng cảm xúc (happy, neutral, sad...).
- `data/`: Bộ dữ liệu hình ảnh dùng để huấn luyện AI (gồm thư mục `test` và `train`).
- `haarcascades/`: Chứa các file cấu hình nhận diện vật thể.
- `model/`: Lưu trữ các mô hình đã huấn luyện (.prototxt, .caffemodel, .h5).
- `src/`: Mã nguồn chính của dự án:
    - `main.py`: Chạy chương trình chính.
    - `train.py`: Huấn luyện mô hình cảm xúc.
    - `utils.py`: Các hàm bổ trợ xử lý hình ảnh.

## Hướng dẫn cài đặt
Cài đặt các thư viện cần thiết bằng lệnh:
```bash
pip install -r requirements.txt