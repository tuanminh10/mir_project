import cv2
import os
import sys
from ultralytics import YOLO

# ==============================================================================
# HACK HỖ TRỢ GPU RTX 5060 TRÊN ROS NOETIC (PYTHON 3.8)
# Tự động nhảy sang môi trường ảo có PyTorch xịn hơn để không bị lỗi "sm_120"
# ==============================================================================
if os.path.exists('/opt/ai_venv/bin/python') and sys.executable != '/opt/ai_venv/bin/python':
    print("🚀 Auto-switched to Python 3.9 venv to unlock NVIDIA GPU RTX 5060...")
    sys.stdout.flush()
    os.execv('/opt/ai_venv/bin/python', ['/opt/ai_venv/bin/python'] + sys.argv)

def main():
    # 1. Đường dẫn tới file weights của model
    # Xử lý đường dẫn tương đối để có thể chạy cả trên Host và trong Docker (start.sh)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "data/detect_laviecoca.v1-laviecoca.yolov11/runs/drink_v11s/weights/best.pt")
    
    if not os.path.exists(model_path):
        print(f"[LỖI] Không tìm thấy model tại: {model_path}")
        print("Vui lòng sửa lại biến model_path trong code trỏ đúng tới file .pt của trò.")
        return

    print(f"[INFO] Đang tải mô hình YOLO từ: {model_path}")
    model = YOLO(model_path)
    
    # 2. Khởi động Camera Laptop (Thường là ID 0)
    print("[INFO] Đang bật Camera Laptop...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[LỖI] Không thể kết nối với Camera Laptop (ID=0).")
        return
    
    # Thiết lập độ phân giải 640x480 để test cho giống với RealSense trên Robot
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("===========================================")
    print("  [BẮT ĐẦU TEST MODEL BẰNG CAMERA LAPTOP]  ")
    print("      Nhấn phím 'q' để thoát chương trình  ")
    print("===========================================")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[LỖI] Không thể đọc khung hình từ Camera.")
            break
            
        # 3. Chạy inference bằng YOLO
        # conf=0.45: Chỉ hiển thị những dự đoán có độ tự tin > 45%
        results = model.predict(source=frame, conf=0.45, verbose=False)
        
        # 4. Vẽ Bounding Box và Nhãn lên ảnh
        annotated_frame = results[0].plot()
        
        # Hiển thị FPS và chữ hướng dẫn
        cv2.putText(annotated_frame, "Press 'q' to quit", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
        # 5. Hiển thị ảnh
        cv2.imshow("Test YOLO Model - Laptop Camera", annotated_frame)
        
        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Đã tắt Camera và thoát chương trình.")

if __name__ == "__main__":
    main()
