import os
from ultralytics import YOLO

# Lấy đường dẫn tuyệt đối của thư mục chứa file train.py
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'data.yaml')

# Load model pre-trained (YOLOv8 nano - rất nhẹ và nhanh)
model = YOLO('yolov8n.pt')

# Bắt đầu train
results = model.train(
   data=data_path,
   epochs=100,       # Số vòng lặp qua toàn bộ dataset
   imgsz=640,        # Kích thước ảnh đầu vào
   batch=32,         # Đã giảm xuống 32 vì batch 64 gây tràn VRAM tại bước TaskAlignedAssigner
   workers=8,        # Giảm số luồng load data lại cho phù hợp với batch 32
   device=0,         # Sử dụng GPU thứ nhất
   cache=True,       # Load toàn bộ dataset vào RAM để loại bỏ hoàn toàn độ trễ đọc ổ cứng
   amp=True          # Bật Automatic Mixed Precision (tính toán FP16 trên GPU giúp train nhanh gấp đôi)
)