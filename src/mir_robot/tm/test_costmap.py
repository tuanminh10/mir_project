import sys
import time
import math
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
import numpy as np
import cv2

from mir_driver.rosbridge import RosbridgeSetup

MIR_IP = "192.168.0.177"
MIR_PORT = 9090

class TestCostmapApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MiR Global Costmap Viewer")
        self.resize(800, 800)
        
        self.label = QLabel("Đang chờ dữ liệu Costmap từ MiR...")
        self.label.setStyleSheet("background-color: black; color: white;")
        self.setCentralWidget(self.label)
        
        self.robot = RosbridgeSetup(MIR_IP, MIR_PORT)
        time.sleep(1)
        if not self.robot.is_connected():
            print("❌ Không thể kết nối MiR ROSBridge!")
            sys.exit(1)
            
        print("✅ Đã kết nối ROSBridge. Đang subscribe vào các topic costmap...")
        
        # Subscribe thẳng vào lớp mây lơ của MiR
        self.robot.subscribe("/move_base_node/global_costmap/inflated_obstacles", self.costmap_cb)
        
        self.last_msg = None
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(100) # 10 FPS
        
    def costmap_cb(self, msg):
        self.last_msg = msg
        
    def update_gui(self):
        if not self.last_msg:
            return
            
        msg = self.last_msg
        self.last_msg = None 
        
        # GridCells có cấu trúc: cell_width, cell_height, cells [{x, y, z}, ...]
        cells = msg.get('cells', [])
        
        # Tạo ảnh đen
        w, h = 800, 800
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Offset để đưa gốc tọa độ bản đồ vào giữa màn hình
        offset_x = 400
        offset_y = 400
        scale = 20 # 1 mét = 20 pixels
        
        for p in cells:
            x_m = p.get('x', 0)
            y_m = p.get('y', 0)
            
            # Map tọa độ thực (mét) sang pixel
            px = int(x_m * scale + offset_x)
            py = int(h - (y_m * scale + offset_y))
            
            if 0 <= px < w and 0 <= py < h:
                # Vẽ đám mây màu xanh lơ
                cv2.circle(img, (px, py), 1, (255, 200, 100), -1)
            
        print(f"☁️ Đã nhận và vẽ {len(cells)} điểm mây xanh lơ!")
        
        qImg = QImage(img.data, w, h, w * 3, QImage.Format_BGR888).copy()
        self.label.setPixmap(QPixmap.fromImage(qImg))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = TestCostmapApp()
    w.show()
    sys.exit(app.exec_())
