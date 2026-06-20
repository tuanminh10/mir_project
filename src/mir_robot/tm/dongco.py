#!/usr/bin/env python3
import time
import math

import serial
import serial.tools.list_ports

import glob

class ServoController:
    def __init__(self, pin=18, min_angle=0, max_angle=180, min_pulse=2.5, max_pulse=12.5):
        """
        Khởi tạo điều khiển Servo thông qua Arduino (Serial/USB).
        Sẽ tự động quét tìm cổng ttyUSB hoặc ttyACM.
        """
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.serial_port = None
        
        # Quét trực tiếp thư mục /dev/ bằng glob thay vì list_ports (tránh lỗi udev trong Docker)
        # Ưu tiên ttyACM trước (thường là Arduino xịn), sau đó mới tới ttyUSB (tránh nhận nhầm Lidar)
        ports_acm = glob.glob('/dev/ttyACM*')
        ports_usb = glob.glob('/dev/ttyUSB*')
        ports = ports_acm + ports_usb
        
        if ports:
            port = ports[0]
            try:
                self.serial_port = serial.Serial(port, 115200, timeout=1)
                time.sleep(2) # Chờ Arduino reset
                print(f"[ServoController] Đã kết nối Arduino tự động tại: {port}")
            except Exception as e:
                print(f"[Cảnh báo] Lỗi kết nối Arduino tại {port}: {e}")
                self.serial_port = None
        else:
            print("[Cảnh báo] Không tìm thấy Arduino nào (ttyACM/ttyUSB). Chế độ mô phỏng (Simulation) được bật.")

    def set_angle(self, angle):
        """
        Quay servo tới một góc cụ thể qua Serial.
        """
        if angle < self.min_angle:
            angle = self.min_angle
        elif angle > self.max_angle:
            angle = self.max_angle
            
        if self.serial_port and self.serial_port.is_open:
            try:
                val = int(float(angle))
                self.serial_port.write(f"{val}\n".encode('utf-8'))
                # Chờ một khoảng nhỏ để servo có thể quay tới đích (nếu cần)
                time.sleep(0.3)
            except Exception as e:
                print(f"[ServoController] Lỗi gửi lệnh: {e}")
        else:
            print(f"[Simulation] Servo quay tới {angle} độ")
            
    def scan(self, start_angle, end_angle, step=1, delay=0.05):
        """
        Chế độ quét (scan) dùng gắn camera xoay qua lại.
        Giúp quét tìm người và tránh vật cản hoặc thân MIR.
        """
        print(f"Bắt đầu quét từ {start_angle} đến {end_angle} độ...")
        # Lượt đi
        for angle in range(start_angle, end_angle + 1, step):
            self.set_angle(angle)
            time.sleep(delay)
            
        # Lượt về
        for angle in range(end_angle, start_angle - 1, -step):
            self.set_angle(angle)
            time.sleep(delay)

    def cleanup(self):
        """
        Đóng cổng Serial khi kết thúc chương trình.
        """
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            print("[ServoController] Đã đóng cổng Serial.")

if __name__ == "__main__":
    # Ví dụ sử dụng:
    # Điều chỉnh chân pin (18) phù hợp với sơ đồ đấu nối của bạn
    # Servo 35kg thường là loại 180 độ hoặc 270 độ
    servo = ServoController(pin=18, min_angle=0, max_angle=180)
    
    try:
        # Căn chỉnh camera nhìn thẳng (90 độ)
        servo.set_angle(90)
        time.sleep(1)
        
        print("Bắt đầu quét liên tục để nhận diện...")
        while True:
            # Quét camera từ 30 độ đến 150 độ, mỗi bước 15 độ
            servo.scan(start_angle=30, end_angle=150, step=15, delay=0.2)
            
    except KeyboardInterrupt:
        print("Đã dừng thủ công.")
    finally:
        servo.cleanup()
