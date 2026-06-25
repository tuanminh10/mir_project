#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Train Model Nhận Diện Đồ Uống (Coca & Lavie) cho Robot MiR
==================================================================
- Model: YOLOv11s (đồng bộ với yolo11s-pose.pt và yolo11s-seg.pt trên robot)
- Dataset: ~914 ảnh train + augmented từ Roboflow
- GPU: NVIDIA RTX 5060
- Output: best.pt → Copy lên robot để chạy verify_tray()

Cách chạy:
    python trainsp.py
"""

import os
from ultralytics import YOLO

# ============================================================
# CẤU HÌNH
# ============================================================

# Đường dẫn tuyệt đối đến file data.yaml
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(BASE_DIR, 'data.yaml')

# Thư mục lưu kết quả train (nằm trong project cho dễ tìm)
PROJECT_DIR = os.path.join(BASE_DIR, 'runs')
RUN_NAME = 'drink_v11s_smooth'

# ============================================================
# KHỞI TẠO MODEL
# ============================================================

# YOLOv11s — 9.4M params, cân bằng hoàn hảo cho ~1000 ảnh
# Đồng bộ engine với yolo11s-pose.pt và yolo11s-seg.pt trên robot
# Sửa đường dẫn để dùng file yolo11s.pt có sẵn ở thư mục ngoài
model = YOLO(os.path.abspath(os.path.join(BASE_DIR, '../../yolo11s.pt')))

# ============================================================
# BẮT ĐẦU TRAIN
# ============================================================

results = model.train(
    data=DATA_YAML,
    project=PROJECT_DIR,
    name=RUN_NAME,

    # === Số epoch & Dừng sớm ===
    epochs=150,           # 914 ảnh → 150 epochs là đủ
    patience=30,          # Dừng nếu 30 epoch liên tiếp val không cải thiện

    # === Kích thước ảnh & Batch ===
    imgsz=640,            # Camera RealSense stream 640x480 → resize 640 là chuẩn
    batch=16,             # RTX 5060 dư sức batch 16 với model Small
    workers=4,            # 4 luồng đọc ảnh song song

    # === GPU & Tối ưu ===
    device=0,             # GPU 0
    cache=True,           # Cache ảnh vào RAM → tăng tốc 2-3x
    amp=True,             # FP16 Mixed Precision — khớp với half=True trên robot

    # === Optimizer ===
    optimizer='AdamW',    # AdamW ổn định hơn SGD cho dataset < 2000 ảnh
    lr0=0.001,            # Learning rate thấp hơn default (0.01) vì dataset nhỏ
    lrf=0.01,             # Learning rate cuối = lr0 * lrf = 0.00001
    cos_lr=True,          # Cosine Annealing — giảm LR mượt mà

    # === Augmentation (Giảm nhẹ để Val curve mượt hơn) ===
    mosaic=0.5,           # Giảm mosaic xuống 50% để tránh làm bài toán quá khó
    mixup=0.0,            # Tắt mixup (Đồ uống thật trên khay không đè lồng lên nhau)
    close_mosaic=15,      # Tắt sớm hơn (15 epoch cuối) để hội tụ mượt
    degrees=5.0,          # Xoay ±5° (camera gắn cố định, không cần xoay nhiều)
    translate=0.1,        # Dịch ảnh ±10%
    scale=0.1,            # Giảm thu phóng xuống 10% (Chai nước có kích thước khá cố định)
    flipud=0.0,           # KHÔNG lật dọc (chai nước không bao giờ lộn ngược)
    fliplr=0.5,           # Lật ngang 50% (đã có trên Roboflow nhưng thêm cũng tốt)
    hsv_h=0.015,          # Biến đổi Hue nhẹ
    hsv_s=0.3,            # Giảm biến đổi Saturation xuống
    hsv_v=0.2,            # Giảm biến đổi Brightness xuống

    # === Chống Overfitting (Tăng hình phạt) ===
    dropout=0.2,          # Tăng Dropout lên 20% bắt model không được học thuộc lòng
    weight_decay=0.001,   # Tăng gấp đôi hình phạt L2 regularization

    # === Validation ===
    val=True,             # Chạy validation sau mỗi epoch
)

# ============================================================
# SAU KHI TRAIN XONG
# ============================================================

print("\n" + "=" * 60)
print("🎉 TRAIN HOÀN TẤT!")
print("=" * 60)

best_path = os.path.join(PROJECT_DIR, RUN_NAME, 'weights', 'best.pt')
print(f"\n📦 File model tốt nhất: {best_path}")
print(f"\n📋 Bước tiếp theo:")
print(f"   1. Copy file 'best.pt' lên robot MiR")
print(f"   2. Đặt vào: /home/tuanminh/mir_project/src/mir_robot/tm/best/best.pt")
print(f"   3. Khởi động lại mainv4.py")
print(f"\n💡 Để xem kết quả chi tiết, mở thư mục:")
print(f"   {os.path.join(PROJECT_DIR, RUN_NAME)}")
