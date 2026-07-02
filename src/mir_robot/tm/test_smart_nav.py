#!/usr/bin/env python3
import os
import sys

# ==============================================================================
# HACK HỖ TRỢ GPU RTX 5060 TRÊN ROS NOETIC (PYTHON 3.8)
# ==============================================================================
if os.path.exists('/opt/ai_venv/bin/python') and sys.executable != '/opt/ai_venv/bin/python':
    print("🚀 Auto-switched to Python 3.9 venv to unlock NVIDIA GPU...")
    sys.stdout.flush()
    os.execv('/opt/ai_venv/bin/python', ['/opt/ai_venv/bin/python'] + sys.argv)

os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
os.environ['QT_API'] = 'pyqt5'
os.environ['YOLO_OFFLINE'] = 'True'

_ros_sys_path = '/opt/ros/noetic/lib/python3/dist-packages'
if os.path.isdir(_ros_sys_path) and _ros_sys_path not in sys.path:
    sys.path.insert(1, _ros_sys_path)
    for mod_name in list(sys.modules.keys()):
        if any(mod_name.startswith(p) for p in ['geometry_msgs', 'nav_msgs', 'sensor_msgs',
                                                  'std_msgs', 'actionlib_msgs', 'tf2_msgs', 'move_base_msgs']):
            del sys.modules[mod_name]

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush
from PyQt5.QtCore import Qt, QTimer, pyqtSignal

# ĐẶC BIỆT: Khởi tạo QApplication NGAY LẬP TỨC để khóa plugin PyQt5 an toàn
# Việc này ngăn chặn thư viện cv2 (OpenCV) tải nhầm plugin Qt của nó và gây crash (core dumped).
app = QApplication(sys.argv)

import math
import numpy as np
import cv2
import rospy
from nav_msgs.msg import OccupancyGrid
import tf.transformations

import navigationcacdiem as nav

class MapSimulatorLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.map_data = None
        self.map_info = None
        self.virtual_obstacles = [] # List of (px, py, radius)
        
        self.target_px = None
        self.target_py = None
        
        # Results to draw
        self.order_px = None
        self.order_yaw = None
        self.deliver_px = None
        self.deliver_yaw = None
        
        self.ray_pixels = []
        self.deliver_ray_pixels = []
        self.footprint_pixels = []
        
        self.current_robot_x = 10.0 # Tọa độ robot giả lập (để test phá băng)
        self.current_robot_y = 15.0

    def set_map(self, msg):
        print(f"[GUI] Đã nhận bản đồ thành công từ ROS! Kích thước: {msg.info.width}x{msg.info.height}")
        self.map_info = msg.info
        data = np.array(msg.data, dtype=np.int8)
        data = data.reshape((msg.info.height, msg.info.width))
        
        self.map_data = np.zeros_like(data, dtype=np.uint8)
        self.map_data[data == 100] = 100
        self.map_data[data == -1] = 127
        self.update_display()

    def add_virtual_obstacle(self, cx, cy, radius=10):
        self.virtual_obstacles.append((cx, cy, radius))
        self.update_display()

    def get_merged_map(self):
        if self.map_data is None: return None
        merged = self.map_data.copy()
        for cx, cy, r in self.virtual_obstacles:
            cv2.circle(merged, (cx, cy), r, 100, -1)
        return merged

    def mousePressEvent(self, event):
        if self.map_info is None: return
        
        # Tính toán pixel được click
        w = self.map_info.width
        h = self.map_info.height
        
        # Scaled click
        if self.pixmap() is None: return
        scaled_w = self.pixmap().width()
        scaled_h = self.pixmap().height()
        
        offset_x = (self.width() - scaled_w) / 2.0
        offset_y = (self.height() - scaled_h) / 2.0
        
        mx = event.pos().x() - offset_x
        my = event.pos().y() - offset_y
        
        if mx < 0 or mx >= scaled_w or my < 0 or my >= scaled_h:
            return
            
        px = int(mx * (w / scaled_w))
        py_qt = int(my * (h / scaled_h))
        py = h - 1 - py_qt # Đảo ngược trục Y của ROS
        
        if event.button() == Qt.LeftButton:
            self.target_px = px
            self.target_py = py
            self.run_algorithm()
        elif event.button() == Qt.RightButton:
            self.add_virtual_obstacle(px, py, radius=15)
            if self.target_px is not None:
                self.run_algorithm()

    def run_algorithm(self):
        if self.target_px is None or self.map_info is None: return
        merged_map = self.get_merged_map()
        
        res = self.map_info.resolution
        w = self.map_info.width
        h = self.map_info.height
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        
        px_t = self.target_px
        py_t = self.target_py
        
        # Xóa dữ liệu vẽ cũ
        self.ray_pixels.clear()
        self.deliver_ray_pixels.clear()
        self.footprint_pixels.clear()
        
        # --- THUẬT TOÁN TỪ MAINV5 ---
        safe_radius_m = 0.55
        inflate_m = 0.25
        min_dock_dist_m = 0.75
        max_dock_dist_m = 2.0
        
        safe_r_px = int(safe_radius_m / res)
        inflate_px = int(inflate_m / res)
        
        # Tạo costmap nội bộ
        obs_mask = (merged_map == 100).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*inflate_px+1, 2*inflate_px+1))
        inflated_obs = cv2.dilate(obs_mask, kernel)
        
        best_rays = []
        for deg in range(0, 360, 5):
            rad = math.radians(deg)
            free_count = 0
            for step in range(1, int(4.0 / res)):
                cx = int(px_t + step * math.cos(rad))
                cy = int(py_t + step * math.sin(rad))
                if not (0 <= cx < w and 0 <= cy < h): break
                if inflated_obs[cy, cx] > 0: break
                free_count += 1
            best_rays.append((rad, free_count))
            
        max_free = max(c for _, c in best_rays) if best_rays else 0
        open_rays = [r for r, c in best_rays if c >= max_free * 0.9]
        if not open_rays: return
        
        sx = sum(math.cos(r) for r in open_rays)
        sy = sum(math.sin(r) for r in open_rays)
        theta_open = math.atan2(sy, sx)
        
        fp_m = [(0.506, -0.32), (0.506, 0.32), (-0.454, 0.32), (-0.454, -0.32)]
        
        def test_path(theta_dock_test, yaw_test, test_min_dist_m=0.70):
            test_min_step = int(test_min_dist_m / res)
            max_test_step = int(max_dock_dist_m / res)
            for step in range(test_min_step, max_test_step):
                cx = int(px_t + step * math.cos(theta_dock_test))
                cy = int(py_t + step * math.sin(theta_dock_test))
                if not (0 <= cx < w and 0 <= cy < h): break
                
                pts = []
                for dx, dy in fp_m:
                    rx = (dx * math.cos(yaw_test) - dy * math.sin(yaw_test)) / res
                    ry = (dx * math.sin(yaw_test) + dy * math.cos(yaw_test)) / res
                    pts.append([int(cx + rx), int(cy + ry)])
                    
                pts = np.array(pts, np.int32).reshape((-1, 1, 2))
                x_min, y_min = np.min(pts, axis=0)[0]
                x_max, y_max = np.max(pts, axis=0)[0]
                x_min = max(0, x_min); y_min = max(0, y_min)
                x_max = min(w-1, x_max); y_max = min(h-1, y_max)
                if x_min >= x_max or y_min >= y_max: continue
                
                roi = inflated_obs[y_min:y_max+1, x_min:x_max+1]
                if np.any(roi > 0):
                    # Check polygon
                    mask = np.zeros((y_max-y_min+1, x_max-x_min+1), dtype=np.uint8)
                    local_pts = pts - np.array([x_min, y_min])
                    cv2.fillPoly(mask, [local_pts], 255)
                    overlap = cv2.bitwise_and(roi, mask)
                    if np.any(overlap > 0):
                        continue
                return step, pts
            return None, None

        # ==================== ORDER ====================
        theta_raw_left = theta_open + math.radians(45)
        theta_raw_right = theta_open - math.radians(45)
        global_angles = [45, 135, -45, -135]
        
        def get_snapped_angle(raw_rad):
            deg = math.degrees(raw_rad)
            best = min(global_angles, key=lambda a: abs((a - deg + 180) % 360 - 180))
            return math.radians(best)
            
        theta_left = get_snapped_angle(theta_raw_left)
        theta_right = get_snapped_angle(theta_raw_right)
        
        yaw_left = (theta_left - math.pi + math.pi) % (2 * math.pi) - math.pi
        yaw_right = (theta_right - math.pi + math.pi) % (2 * math.pi) - math.pi
        
        step_left, fp_left = test_path(theta_left, yaw_left, min_dock_dist_m)
        step_right, fp_right = test_path(theta_right, yaw_right, min_dock_dist_m)
        
        theta_dock = theta_left
        target_step = None
        
        if step_left is not None and step_right is not None:
            if step_left <= step_right:
                theta_dock = theta_left; target_step = step_left
            else:
                theta_dock = theta_right; target_step = step_right
        elif step_left is not None: 
            theta_dock = theta_left; target_step = step_left
        elif step_right is not None: 
            theta_dock = theta_right; target_step = step_right
            
        if target_step is not None:
            self.order_px = (int(px_t + target_step * math.cos(theta_dock)), h - 1 - int(py_t + target_step * math.sin(theta_dock)))
            self.order_yaw = (theta_dock - math.pi + math.pi) % (2 * math.pi) - math.pi
            
            for step in range(int(min_dock_dist_m/res), target_step + 1):
                cx = int(px_t + step * math.cos(theta_dock))
                cy = int(py_t + step * math.sin(theta_dock))
                if 0 <= cx < w and 0 <= cy < h:
                    self.ray_pixels.append((cx, h - 1 - cy))

        # ==================== DELIVER ====================
        theta_raw_d_left = theta_open + math.radians(90)
        theta_raw_d_right = theta_open - math.radians(90)
        deliver_global_angles = [0, 90, 180, -90]
        
        def get_snapped_deliver_angle(raw_rad):
            deg = math.degrees(raw_rad)
            best = min(deliver_global_angles, key=lambda a: abs((a - deg + 180) % 360 - 180))
            return math.radians(best)

        theta_d_left = get_snapped_deliver_angle(theta_raw_d_left)
        theta_d_right = get_snapped_deliver_angle(theta_raw_d_right)

        def get_deliver_yaw(dock_rad):
            perp_deg = math.degrees(dock_rad) + 90
            best_yaw = min([0, 180], key=lambda a: abs((a - perp_deg + 180) % 360 - 180))
            return math.radians(best_yaw)
            
        yaw_d_left = get_deliver_yaw(theta_d_left)
        step_d_left, fp_dl = test_path(theta_d_left, yaw_d_left, min_dock_dist_m)
        
        yaw_d_right = get_deliver_yaw(theta_d_right)
        step_d_right, fp_dr = test_path(theta_d_right, yaw_d_right, min_dock_dist_m)
        
        theta_dock_d = theta_d_left
        target_step_d = None
        yaw_d = yaw_d_left
        
        if step_d_left is not None and step_d_right is not None:
            px_x_dl = px_t + step_d_left * math.cos(theta_d_left)
            px_y_dl = py_t + step_d_left * math.sin(theta_d_left)
            dist_dl = math.hypot((ox + px_x_dl * res) - self.current_robot_x, (oy + px_y_dl * res) - self.current_robot_y)
            
            px_x_dr = px_t + step_d_right * math.cos(theta_d_right)
            px_y_dr = py_t + step_d_right * math.sin(theta_d_right)
            dist_dr = math.hypot((ox + px_x_dr * res) - self.current_robot_x, (oy + px_y_dr * res) - self.current_robot_y)
            
            if abs(step_d_left - step_d_right) * res < 0.15:
                if dist_dl <= dist_dr:
                    theta_dock_d = theta_d_left; target_step_d = step_d_left; yaw_d = yaw_d_left; fp_sel = fp_dl
                else:
                    theta_dock_d = theta_d_right; target_step_d = step_d_right; yaw_d = yaw_d_right; fp_sel = fp_dr
            else:
                if step_d_left <= step_d_right: 
                    theta_dock_d = theta_d_left; target_step_d = step_d_left; yaw_d = yaw_d_left; fp_sel = fp_dl
                else: 
                    theta_dock_d = theta_d_right; target_step_d = step_d_right; yaw_d = yaw_d_right; fp_sel = fp_dr
        elif step_d_left is not None: 
            theta_dock_d = theta_d_left; target_step_d = step_d_left; yaw_d = yaw_d_left; fp_sel = fp_dl
        elif step_d_right is not None: 
            theta_dock_d = theta_d_right; target_step_d = step_d_right; yaw_d = yaw_d_right; fp_sel = fp_dr
            
        if target_step_d is not None:
            # Lưu vết tia
            for step in range(int(min_dock_dist_m/res), target_step_d + 1):
                cx = int(px_t + step * math.cos(theta_dock_d))
                cy = int(py_t + step * math.sin(theta_dock_d))
                if 0 <= cx < w and 0 <= cy < h:
                    self.deliver_ray_pixels.append((cx, h - 1 - cy))
                    
            # Tính toán tịnh tiến khay
            final_step_d = (target_step_d * res + 0.02) / res
            px_x_d = int(px_t + final_step_d * math.cos(theta_dock_d))
            px_y_d = int(py_t + final_step_d * math.sin(theta_dock_d))
            
            shift_m = -0.30
            px_x_d += int((shift_m * math.cos(yaw_d)) / res)
            px_y_d += int((shift_m * math.sin(yaw_d)) / res)
            
            self.deliver_px = (px_x_d, h - 1 - px_y_d)
            self.deliver_yaw = yaw_d
            
            # Tính footprint cho deliver
            for dx, dy in fp_m:
                rx = (dx * math.cos(yaw_d) - dy * math.sin(yaw_d)) / res
                ry = (dx * math.sin(yaw_d) + dy * math.cos(yaw_d)) / res
                self.footprint_pixels.append((int(px_x_d + rx), h - 1 - int(px_y_d + ry)))

        self.update_display()

    def update_display(self):
        if self.map_data is None: return
        merged = self.get_merged_map()
        
        # Colorize
        rgb = np.zeros((merged.shape[0], merged.shape[1], 3), dtype=np.uint8)
        rgb[merged == 0] = [255, 255, 255] # Free
        rgb[merged == 100] = [0, 0, 0]     # Obs
        rgb[merged == 127] = [200, 200, 200] # Unknown
        
        # Lật ảnh Y
        rgb = cv2.flip(rgb, 0)
        
        h, w, c = rgb.shape
        bytes_per_line = 3 * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        
        painter = QPainter(pix)
        
        # Vẽ tia ORDER
        if self.ray_pixels:
            painter.setPen(QPen(QColor("blue"), 2))
            for px, py in self.ray_pixels:
                painter.drawPoint(px, py)
                
        # Vẽ tia DELIVER
        if self.deliver_ray_pixels:
            painter.setPen(QPen(QColor("purple"), 2))
            for px, py in self.deliver_ray_pixels:
                painter.drawPoint(px, py)
                
        # Vẽ target
        if self.target_px is not None:
            painter.setPen(QPen(QColor("red"), 5))
            qt_py = self.map_info.height - 1 - self.target_py
            painter.drawEllipse(self.target_px-3, qt_py-3, 6, 6)
            
        # Vẽ ORDER goal
        if self.order_px is not None:
            painter.setPen(QPen(QColor("blue"), 4))
            painter.drawEllipse(self.order_px[0]-3, self.order_px[1]-3, 6, 6)
            
        # Vẽ DELIVER goal
        if self.deliver_px is not None:
            painter.setPen(QPen(QColor("purple"), 4))
            painter.drawEllipse(self.deliver_px[0]-3, self.deliver_px[1]-3, 6, 6)
            
        # Vẽ Footprint của DELIVER
        if len(self.footprint_pixels) == 4:
            painter.setPen(QPen(QColor("purple"), 2, Qt.DashLine))
            pts = self.footprint_pixels
            for i in range(4):
                p1 = pts[i]
                p2 = pts[(i+1)%4]
                painter.drawLine(p1[0], p1[1], p2[0], p2[1])
                
        painter.end()
        self.setPixmap(pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        self.update_display()

class MainWindow(QMainWindow):
    map_signal = pyqtSignal(OccupancyGrid)
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MiR SMART NAV - HỆ THỐNG GIẢ LẬP ĐỖ NGANG VÀ LẤY ĐỒ")
        self.map_label = MapSimulatorLabel()
        self.map_label.setAlignment(Qt.AlignCenter)
        self.map_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCentralWidget(self.map_label)
        
        # Kết nối tới MiR để kích hoạt ROS bridge đẩy map xuống
        print("[INFO] Đang kết nối tới MiR Robot...")
        self.robot = nav.ws_connect()
        self.mir_headers = nav.api_login()
        print("[INFO] Kết nối thành công! Đang chờ tải bản đồ...")
        
        self.map_signal.connect(self.map_label.set_map)
        
        rospy.init_node('test_nav_gui', anonymous=True)
        rospy.Subscriber('/map', OccupancyGrid, self.map_cb)

    def map_cb(self, msg):
        self.map_signal.emit(msg)

if __name__ == "__main__":
    window = MainWindow()
    window.resize(1000, 800)
    window.show()
    
    # Let Qt process ROS callbacks
    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(100)
    
    sys.exit(app.exec())
