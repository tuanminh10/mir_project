import os

# 1. THIẾT LẬP MÔI TRƯỜNG AI
os.environ['YOLO_OFFLINE'] = 'True'
YOLO_MODEL_PATH = 'yolo11n-pose.pt'

# 2. CẤU HÌNH ROBOT MIR
MIR_IP = "192.168.0.177"
MIR_API_URL = f"http://{MIR_IP}/api/v2.0.0"
MIR_AUTH = "Basic YWRtaW46OGM2OTc2ZTViNTQxMDQxNWJkZTkwOGJkNGRlZTE1ZGZiMTY3YTljODczZmM0YmI4YTgxZjZmMmFiNDQ4YTkxOA=="

ROBOT_WIDTH_M = 0.88
ROBOT_LENGTH_M = 0.55

# 3. KÍCH THƯỚC VÀ KHOẢNG CÁCH (MAP & NAVIGATION)
CAMERA_OFFSET_X_M = 0.475
# Khoảng cách tối thiểu phải dừng lại trước người để tránh vi phạm Costmap Inflation
STOP_DISTANCE_M = 1.2
