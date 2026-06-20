import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

# Import thư viện động cơ (như trong mainv42.py)
try:
    import dongco
    HAS_SERVO = True
except ImportError:
    HAS_SERVO = False
    print("[WARNING] Không tìm thấy module dongco.py. Bỏ qua điều khiển servo.")

def main():
    # 1. Khởi động Servo và gập xuống 155 độ tự động
    if HAS_SERVO:
        try:
            print("[INFO] Đang khởi động Servo và bẻ góc camera xuống 155 độ...")
            servo = dongco.ServoController(pin=18, min_angle=0, max_angle=180)
            servo.set_angle(155)
            time.sleep(2.0)  # Đợi mô tơ gập hẳn xuống
            print("[INFO] Đã gập xong 155 độ!")
        except Exception as e:
            print(f"[ERROR] Không thể điều khiển servo: {e}")

    # 2. Thư mục lưu ảnh
    save_dir = "dataset_lavie_coca"
    os.makedirs(save_dir, exist_ok=True)
    
    # 3. Cấu hình RealSense (Đồng bộ 100% với mainv42.py)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    print("[INFO] Đang bật camera RealSense giống hệt thông số trên Robot...")
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    
    # --- CẤU HÌNH CẢM BIẾN COLOR (Giảm lóa, chống nhòe) ---
    color_sensor = profile.get_device().first_color_sensor()
    if color_sensor:
        color_sensor.set_option(rs.option.enable_auto_exposure, 1)
        if color_sensor.supports(rs.option.auto_exposure_priority):
            color_sensor.set_option(rs.option.auto_exposure_priority, 0)
        if color_sensor.supports(rs.option.sharpness):
            color_sensor.set_option(rs.option.sharpness, 100)
            
    # --- CẤU HÌNH CẢM BIẾN DEPTH (Tăng độ chính xác) ---
    depth_sensor = profile.get_device().first_depth_sensor()
    if depth_sensor:
        if depth_sensor.supports(rs.option.laser_power):
            depth_sensor.set_option(rs.option.laser_power, 360)
        if depth_sensor.supports(rs.option.visual_preset):
            try:
                depth_sensor.set_option(rs.option.visual_preset, 3) # High Accuracy
            except: pass
            
    time.sleep(2.0) # Đợi camera ổn định ánh sáng
    
    # --- TỰ ĐỘNG NỐI TIẾP ẢNH CŨ ---
    img_count = 0
    if os.path.exists(save_dir):
        existing_files = [f for f in os.listdir(save_dir) if f.startswith('frame_') and f.endswith('.jpg')]
        if existing_files:
            # Lọc ra các số thứ tự và lấy số lớn nhất
            indices = [int(f.replace('frame_', '').replace('.jpg', '')) for f in existing_files]
            img_count = max(indices) + 1
            print(f"[INFO] Tìm thấy {len(existing_files)} ảnh cũ. Sẽ chụp tiếp bắt đầu từ: frame_{img_count:04d}.jpg")
    
    images_to_capture = 100
    max_images = img_count + images_to_capture  # Sẽ dừng sau khi chụp THÊM 100 ảnh nữa
    
    mode = "MANUAL" # Bắt đầu bằng chế độ chụp bằng tay
    last_auto_capture_time = time.time()
    auto_interval = 1.0
    
    print("=========================================")
    print("[HƯỚNG DẪN SỬ DỤNG]")
    print("- Nhấn phím 'c' hoặc 'Space': Để chụp THỦ CÔNG (MANUAL) từng tấm một.")
    print("- Nhấn phím 'a': Để chuyển sang chụp TỰ ĐỘNG (AUTO) 1 giây/tấm.")
    print("- Nhấn phím 'm': Để quay về chụp THỦ CÔNG (MANUAL).")
    print("- Nhấn phím 'q': Để THOÁT.")
    print("=========================================")
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame: continue
                
            color_image = np.asanyarray(color_frame.get_data())
            display_img = color_image.copy()
            
            # Hiển thị trạng thái (Mode và số lượng ảnh)
            color_mode = (0, 0, 255) if mode == "MANUAL" else (0, 255, 0)
            cv2.putText(display_img, f"MODE: {mode}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color_mode, 2)
            cv2.putText(display_img, f"Captured: {img_count}/{max_images}", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            cv2.imshow('RealSense Data Collection', display_img)
            
            key = cv2.waitKey(1) & 0xFF
            capture_now = False
            
            # Xử lý phím bấm
            if key == ord('q'):
                print(f"\n[INFO] Thoát! Tổng cộng: {img_count} ảnh.")
                break
            elif key == ord('a') and mode != "AUTO":
                mode = "AUTO"
                print("\n[CHẾ ĐỘ] Đã chuyển sang AUTO (Tự động chụp sau mỗi giây).")
                last_auto_capture_time = time.time()
            elif key == ord('m') and mode != "MANUAL":
                mode = "MANUAL"
                print("\n[CHẾ ĐỘ] Đã chuyển sang MANUAL (Nhấn 'c' để chụp).")
            elif (key == ord('c') or key == 32) and mode == "MANUAL": # Phím C hoặc Space
                capture_now = True
                
            # Xử lý chụp tự động
            if mode == "AUTO":
                current_time = time.time()
                if current_time - last_auto_capture_time >= auto_interval:
                    capture_now = True
                    last_auto_capture_time = current_time
                    
            # Lưu ảnh
            if capture_now and img_count < max_images:
                filename = os.path.join(save_dir, f"frame_{img_count:04d}.jpg")
                cv2.imwrite(filename, color_image)
                print(f"[{mode}] Đã lưu: {filename}")
                img_count += 1
                
                # Hiệu ứng nháy nháy khi chụp thành công
                flash_img = np.ones_like(color_image) * 255
                cv2.imshow('RealSense Data Collection', flash_img)
                cv2.waitKey(50)
                
            if img_count >= max_images:
                print(f"\n[INFO] ĐÃ XONG! Đủ {max_images} ảnh. Tự động thoát.")
                break
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
