# config.py
# Cấu hình chung cho Robot MiR

# IP của Robot MiR để gọi REST API
MIR_IP = "192.168.0.177"
MIR_API_URL = f"http://{MIR_IP}/api/v2.0.0"

# Cấu hình Camera RealSense
CAMERA_HEIGHT_M = 1.8
CAMERA_PITCH_DEG = 20.0

# Cấu hình Navigation
MAX_NAV_RETRIES = 6
