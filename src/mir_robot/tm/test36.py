#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
import math
import rospy
import tf.transformations
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import pyqtSignal, Qt
from nav_msgs.msg import OccupancyGrid

class MapLabel(QLabel):
    clicked_signal = pyqtSignal(float, float, object)
    right_clicked_signal = pyqtSignal(float, float)

    def __init__(self):
        super().__init__()
        self.setText("Đang chờ dữ liệu từ ROS topic /map ...")
        self.setStyleSheet("background-color: #333; color: white; font-size: 16px;")
        
        self.map_img = None
        self.map_info = None
        self.map_data = None
        
        self.target_px = None
        self.goal_px = None
        self.goal_yaw = None
        self.deliver_px = None
        self.deliver_yaw = None
        self.ray_pixels = []
        self.simulated_obstacles = []

    def set_map(self, occ_grid):
        self.map_info = occ_grid.info
        w, h = self.map_info.width, self.map_info.height
        data = np.array(occ_grid.data, dtype=np.int8).reshape((h, w))
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[data == -1] = [220, 220, 220] 
        img[data == 0] = [255, 255, 255]  
        img[data > 0] = [0, 0, 0]         
        self.map_img = cv2.flip(img, 0)
        self.map_data = data
        
        # Áp dụng chướng ngại vật mô phỏng vào map_data và map_img
        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        for wx, wy in self.simulated_obstacles:
            px = int((wx - ox) / res)
            py = int((wy - oy) / res)
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        self.map_data[ny, nx] = 100
                        # Tọa độ trên ảnh đã flip:
                        img_y = h - ny - 1
                        self.map_img[img_y, nx] = [0, 0, 255] # Màu đỏ cho vật cản giả
                        
        self.update_view()

    def update_view(self):
        if self.map_img is None: return
        display_img = self.map_img.copy()
        h, w, ch = display_img.shape

        # VẼ KHÁCH HÀNG
        if self.target_px:
            cv2.circle(display_img, self.target_px, 6, (0, 0, 255), -1) 
            cv2.putText(display_img, "CUSTOMER", (self.target_px[0]+10, self.target_px[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # VẼ TIA RAYCAST
        if hasattr(self, 'ray_pixels') and self.ray_pixels:
            for pt in self.ray_pixels:
                cv2.circle(display_img, pt, 1, (0, 255, 255), -1) 

        res = self.map_info.resolution if self.map_info else 0.05
        fp_m = [(0.506, -0.32), (0.506, 0.32), (-0.454, 0.32), (-0.454, -0.32)]

        # VẼ SMART GOAL (XANH LÁ)
        if self.goal_px and self.goal_yaw is not None:
            gx, gy = self.goal_px
            gui_yaw = -self.goal_yaw 
            
            goal_pts = []
            for dx, dy in fp_m:
                rx = (dx * math.cos(gui_yaw) - dy * math.sin(gui_yaw)) / res
                ry = (dx * math.sin(gui_yaw) + dy * math.cos(gui_yaw)) / res
                goal_pts.append([int(gx + rx), int(gy + ry)]) 
            goal_pts = np.array(goal_pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(display_img, [goal_pts], True, (0, 255, 0), 2)
            cv2.putText(display_img, "ORDER (45 DEG)", (gx - 50, gy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Mũi tên hướng đỗ Order
            ar_len = 35
            end_x = int(gx + ar_len * math.cos(gui_yaw))
            end_y = int(gy + ar_len * math.sin(gui_yaw))
            cv2.arrowedLine(display_img, (gx, gy), (end_x, end_y), (0, 255, 0), 3, tipLength=0.3)

        # VẼ DELIVER GOAL (CAM)
        if self.deliver_px and self.deliver_yaw is not None:
            d_px, d_py = self.deliver_px
            d_yaw = -self.deliver_yaw
            d_pts = []
            for dx, dy in fp_m:
                rx = (dx * math.cos(d_yaw) - dy * math.sin(d_yaw)) / res
                ry = (dx * math.sin(d_yaw) + dy * math.cos(d_yaw)) / res
                d_pts.append([int(d_px + rx), int(d_py + ry)])
            d_pts = np.array(d_pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(display_img, [d_pts], True, (0, 165, 255), 2)
            cv2.putText(display_img, "DELIVER GOAL", (d_px - 60, d_py - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

            # Mũi tên hướng đỗ Delivery
            ar_len = 35
            end_x = int(d_px + ar_len * math.cos(d_yaw))
            end_y = int(d_py + ar_len * math.sin(d_yaw))
            cv2.arrowedLine(display_img, (d_px, d_py), (end_x, end_y), (0, 165, 255), 3, tipLength=0.3)

        qImg = QImage(display_img.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qImg)
        if self.width() > 0 and self.height() > 0:
            self.setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.setPixmap(pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_view()

    def mouseReleaseEvent(self, event):
        if self.map_info is None: return
        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        h = self.map_info.height
        
        # Bù trừ pixel cho QPixmap bị scale
        scaled_w = self.pixmap().width()
        scaled_h = self.pixmap().height()
        
        x_offset = (self.width() - scaled_w) // 2
        y_offset = (self.height() - scaled_h) // 2
        
        mx = event.x() - x_offset
        my = event.y() - y_offset
        
        if mx < 0 or mx >= scaled_w or my < 0 or my >= scaled_h: return
        
        px = int(mx * (self.map_info.width / scaled_w))
        py = int(my * (self.map_info.height / scaled_h))
        
        wx = ox + px * res
        wy = oy + (h - py - 1) * res
        
        if event.button() == Qt.LeftButton:
            self.clicked_signal.emit(wx, wy, None)
        elif event.button() == Qt.RightButton:
            self.right_clicked_signal.emit(wx, wy)


class TestApp(QMainWindow):
    map_signal = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chẩn Đoán Thuật Toán HRI - Giáo Sư Antigravity")
        self.resize(1200, 900)
        
        self.central_widget = QWidget()
        self.layout = QVBoxLayout(self.central_widget)
        
        self.map_label = MapLabel()
        self.map_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.map_label)
        self.setCentralWidget(self.central_widget)

        rospy.init_node('test_hri_goals', anonymous=True)
        self.map_signal.connect(self.map_label.set_map)
        self.map_label.clicked_signal.connect(self.calculate_hybrid_safe_goal)
        self.map_label.right_clicked_signal.connect(self.add_simulated_obstacle)

        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        self.last_target_x = None
        self.last_target_y = None

    def add_simulated_obstacle(self, wx, wy):
        self.map_label.simulated_obstacles.append((wx, wy))
        print(f"Đã thêm chướng ngại vật mô phỏng tại: {wx:.2f}, {wy:.2f}")
        # Re-apply map
        if self.map_label.map_info and self.map_label.map_data is not None:
            # We need the original OccupancyGrid to re-apply. But we only have map_data.
            # It's better to just redraw the small area.
            res = self.map_label.map_info.resolution
            ox = self.map_label.map_info.origin.position.x
            oy = self.map_label.map_info.origin.position.y
            w = self.map_label.map_info.width
            h = self.map_label.map_info.height
            px = int((wx - ox) / res)
            py = int((wy - oy) / res)
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        self.map_label.map_data[ny, nx] = 100
                        self.map_label.map_img[h - ny - 1, nx] = [0, 0, 255]
            self.map_label.update_view()
            
            # Tính toán lại đích đến nếu đã click khách hàng
            if self.last_target_x is not None:
                self.calculate_hybrid_safe_goal(self.last_target_x, self.last_target_y, None)

    def map_callback(self, msg):
        self.map_signal.emit(msg)

    def calculate_hybrid_safe_goal(self, target_x, target_y, obs_pt_map=None, min_dist_m=0.50):
        self.last_target_x = target_x
        self.last_target_y = target_y
        
        if not self.map_label.map_info:
            return
            
        res = self.map_label.map_info.resolution
        ox = self.map_label.map_info.origin.position.x
        oy = self.map_label.map_info.origin.position.y
        w = self.map_label.map_info.width
        h = self.map_label.map_info.height
        
        px_t = int((target_x - ox) / res)
        py_t = int((target_y - oy) / res)
        
        if not (0 <= px_t < w and 0 <= py_t < h): return

        self.map_label.target_px = (int(px_t), h - int(py_t) - 1)

        # 1. TẠO MASK VẬT CẢN (Inflate 0.15m)
        obs_mask = np.where((self.map_label.map_data != 0), 255, 0).astype(np.uint8)
        combined_obs = obs_mask.copy()
        
        inflate_m = 0.15
        inflate_px = max(1, int(inflate_m / res))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*inflate_px+1, 2*inflate_px+1))
        inflated_obs = cv2.dilate(combined_obs, kernel, iterations=1)
        
        self.map_label.ray_pixels = []
        
        # 2. TÌM THETA_OPEN (Phóng tia dò đường)
        best_rays = []
        max_ray_len = int(5.0 / res)
        for deg in range(0, 360, 5):
            rad = math.radians(deg)
            free_count = 0
            for step in range(1, max_ray_len):
                cx = int(px_t + step * math.cos(rad))
                cy = int(py_t + step * math.sin(rad))
                if not (0 <= cx < w and 0 <= cy < h): break
                if self.map_label.map_data[cy, cx] == 0: free_count += 1
                elif self.map_label.map_data[cy, cx] == 100: break
            best_rays.append((rad, free_count))
            
        max_free = max(c for _, c in best_rays)
        open_rays = [r for r, c in best_rays if c >= max_free * 0.9]
        sx = sum(math.cos(r) for r in open_rays)
        sy = sum(math.sin(r) for r in open_rays)
        theta_open = math.atan2(sy, sx)
        
        # ==========================================
        # MODULE 1: SMART GOAL (Order Mode) - Góc chéo 45 độ
        # ==========================================
        theta_raw_left = theta_open + math.radians(45)
        theta_raw_right = theta_open - math.radians(45)
        global_angles = [45, 135, -45, -135]
        def get_snapped_angle(raw_rad):
            deg = math.degrees(raw_rad)
            best = min(global_angles, key=lambda a: abs((a - deg + 180) % 360 - 180))
            return math.radians(best), best
            
        theta_left, deg_left = get_snapped_angle(theta_raw_left)
        theta_right, deg_right = get_snapped_angle(theta_raw_right)
        
        fp_m = [(0.506, -0.32), (0.506, 0.32), (-0.454, 0.32), (-0.454, -0.32)]
        min_step = int(max(0.50, min_dist_m) / res)
        
        def test_path(theta_dock_test, yaw_test):
            for step in range(min_step, max_ray_len):
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
                local_pts = pts - np.array([x_min, y_min])
                mask = np.zeros_like(roi)
                cv2.fillPoly(mask, [local_pts], 255)
                if not np.any((roi > 0) & (mask > 0)): return step, pts
            return None, None

        yaw_o_left = (theta_left - math.pi + math.pi) % (2 * math.pi) - math.pi
        step_left, pts_left = test_path(theta_left, yaw_o_left)
        yaw_o_right = (theta_right - math.pi + math.pi) % (2 * math.pi) - math.pi
        step_right, pts_right = test_path(theta_right, yaw_o_right)
        
        theta_dock = theta_left
        target_step = None
        if step_left is not None and step_right is not None:
            if step_left <= step_right: theta_dock = theta_left; target_step = step_left
            else: theta_dock = theta_right; target_step = step_right
        elif step_left is not None: theta_dock = theta_left; target_step = step_left
        elif step_right is not None: theta_dock = theta_right; target_step = step_right
        else: theta_dock = theta_left; target_step = min_step
            
        target_dist_m = target_step * res
        target_dist_m += 0.10 # Lùi thêm an toàn 0.10m
        final_step = target_dist_m / res
        px_x = int(px_t + final_step * math.cos(theta_dock))
        px_y = int(py_t + final_step * math.sin(theta_dock))
        yaw = theta_dock - math.pi 
        yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
        
        self.map_label.goal_yaw = yaw
        self.map_label.goal_px = (px_x, h - px_y - 1)
        
        for step in range(min_step, target_step + 1):
            cx = int(px_t + step * math.cos(theta_dock))
            cy = int(py_t + step * math.sin(theta_dock))
            if not (0 <= cx < w and 0 <= cy < h): break
            self.map_label.ray_pixels.append((cx, h - cy - 1))

        # ==========================================
        # MODULE 2: DELIVER GOAL (Delivery Mode)
        # Hướng tiếp cận trực tiếp từ sau lưng (theta_open)
        # Ép ngang thân 0 hoặc 180 độ
        # Tịnh tiến trái phải 20cm
        # ==========================================
        theta_d_left = theta_open 
        
        def get_deliver_yaw(dock_rad):
            perp_deg = math.degrees(dock_rad) + 90
            best_yaw = min([0, 180], key=lambda a: abs((a - perp_deg + 180) % 360 - 180))
            return math.radians(best_yaw)
            
        yaw_d_left = get_deliver_yaw(theta_d_left)
        step_d_left, _ = test_path(theta_d_left, yaw_d_left)
        
        if step_d_left is None:
            step_d_left = min_step
            
        final_step_d = (step_d_left * res + 0.02) / res # Ôm sát, chỉ lùi 0.02m
        px_x_d = int(px_t + final_step_d * math.cos(theta_d_left))
        px_y_d = int(py_t + final_step_d * math.sin(theta_d_left))
        
        # Tịnh tiến xe sang trái/phải khách
        shift_m = -0.20 # Lùi mạnh 0.20m dọc theo trục ngang thân
        px_x_d += int((shift_m * math.cos(yaw_d_left)) / res)
        px_y_d += int((shift_m * math.sin(yaw_d_left)) / res)
        
        self.map_label.deliver_px = (px_x_d, h - px_y_d - 1)
        self.map_label.deliver_yaw = yaw_d_left
        
        for step in range(min_step, step_d_left + 1):
            cx = int(px_t + step * math.cos(theta_d_left))
            cy = int(py_t + step * math.sin(theta_d_left))
            if not (0 <= cx < w and 0 <= cy < h): break
            self.map_label.ray_pixels.append((cx, h - cy - 1))

        self.map_label.update_view()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TestApp()
    ex.show()
    sys.exit(app.exec_())
