#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import pyqtSignal, Qt
import cv2
import numpy as np
import math
import config

class MapLabel(QLabel):
    clicked_signal = pyqtSignal(float, float, object)

    def __init__(self):
        super().__init__()
        self.setText("Đang chờ dữ liệu từ ROS topic /map ...")
        self.setStyleSheet("background-color: #333; color: white; font-size: 16px;")
        
        self.map_img = None
        self.map_info = None
        self.robot_px = None
        self.robot_yaw = 0.0
        self.map_data = None
        
        self.target_px = None
        self.goal_px = None
        self.ray_pixels = []
        self.obstacle_px = None
        self.obs3d_px = None
        self.table_box_px = []
        self.scan_msg = None
        self.last_rejected_pts = None

    def set_robot_pose(self, wx, wy, yaw=0.0):
        if not self.map_info: return
        res, ox, oy, h = self.map_info.resolution, self.map_info.origin.position.x, self.map_info.origin.position.y, self.map_info.height
        px = int((wx - ox) / res)
        py = h - int((wy - oy) / res) - 1
        if 0 <= px < self.map_info.width and 0 <= py < h:
            self.robot_px = (px, py)
            self.robot_yaw = yaw
            self.update_view()

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
        self.update_view()

    def update_view(self):
        if self.map_img is None: return
        display_img = self.map_img.copy()
        h, w, ch = display_img.shape

        # Vẽ Target (Khách hàng)
        if self.target_px:
            cv2.circle(display_img, self.target_px, 6, (0, 0, 255), -1) # Chấm xanh dương (RGB format)
            cv2.putText(display_img, "CUSTOMER", (self.target_px[0]+10, self.target_px[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Vẽ tia dò đường (Raycasting)
        # Đã ẩn theo yêu cầu của user để GUI nhìn rõ hơn
        # if hasattr(self, 'cone_pixels') and self.cone_pixels:
        #     for pt in self.cone_pixels:
        #         cv2.circle(display_img, pt, 1, (0, 255, 255), -1) 
        # if hasattr(self, 'end_pixels') and self.end_pixels:
        #     for pt in self.end_pixels:
        #         cv2.circle(display_img, pt, 1, (0, 0, 255), -1)
        
        # VẼ ĐƯỜNG ĐI DỰ ĐOÁN (MÀU VÀNG) 
        if hasattr(self, 'ray_pixels') and self.ray_pixels:
            for pt in self.ray_pixels:
                cv2.circle(display_img, pt, 1, (0, 255, 255), -1) # Màu Vàng (BGR)

        # Vẽ vật cản 3D từ Camera
        if self.obs3d_px:
            cv2.circle(display_img, self.obs3d_px, 5, (255, 0, 255), -1) # Tím
            cv2.putText(display_img, "3D OBS", (self.obs3d_px[0]+10, self.obs3d_px[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        # VẼ FOOTPRINT ĐANG TEST / BỊ TỪ CHỐI (MÀU ĐỎ) ĐỂ ANIMATION
        if hasattr(self, 'last_rejected_pts') and self.last_rejected_pts is not None:
            rej_pts = self.last_rejected_pts
            draw_pts = []
            for pt in rej_pts:
                draw_pts.append([pt[0][0], self.map_info.height - pt[0][1] - 1])
            draw_pts = np.array([draw_pts], np.int32)
            cv2.polylines(display_img, draw_pts, True, (0, 0, 255), 2) # Đỏ
            cv2.putText(display_img, "TESTING COLLISION...", (draw_pts[0][0][0], draw_pts[0][0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Vẽ Goal an toàn
        if self.goal_px:
            # VẼ CHÍNH XÁC FOOTPRINT HÌNH CHỮ NHẬT CỦA MIR ĐỂ USER KIỂM CHỨNG
            if self.map_info and hasattr(self, 'goal_yaw'):
                res = self.map_info.resolution
                fp_m = [(0.42, -0.28), (0.42, 0.28), (-0.42, 0.28), (-0.42, -0.28)]
                pts = []
                gui_yaw = -self.goal_yaw # Giao diện OpenCV có trục Y hướng xuống
                px, py = self.goal_px
                for dx, dy in fp_m:
                    rx = (dx * math.cos(gui_yaw) - dy * math.sin(gui_yaw)) / res
                    ry = (dx * math.sin(gui_yaw) + dy * math.cos(gui_yaw)) / res
                    pts.append([int(px + rx), int(py + ry)])
                
                pts = np.array(pts, np.int32).reshape((-1, 1, 2))
                cv2.polylines(display_img, [pts], True, (0, 255, 255), 2) # Hình chữ nhật Vàng
                cv2.putText(display_img, "MiR Footprint", (px+10, py+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
            # Vẽ đường đỗ từ Robot ra Smart Goal (Màu Vàng RGB: 255, 255, 0)
            if self.robot_px:
                cv2.line(display_img, self.robot_px, self.goal_px, (255, 255, 0), 2, cv2.LINE_AA)
            
            # Vẽ mũi tên hướng đỗ (Yaw) của Goal
            if hasattr(self, 'goal_yaw'):
                gui_yaw = -self.goal_yaw
                ar_len = 35
                gx, gy = self.goal_px
                end_x = int(gx + ar_len * math.cos(gui_yaw))
                end_y = int(gy + ar_len * math.sin(gui_yaw))
                cv2.arrowedLine(display_img, (gx, gy), (end_x, end_y), (0, 255, 0), 3, tipLength=0.3)
                
                # Bỏ vẽ all_rejected_pts theo yêu cầu User
                        
                # VẼ FOOTPRINT CỦA XE TẠI ĐIỂM ĐỖ CHỐT ĐƯỢC (MÀU XANH LÁ)
                # Lấy đúng kích thước hình chữ nhật gốc
                fp_m = [(0.506, -0.32), (0.506, 0.32), (-0.454, 0.32), (-0.454, -0.32)]
                res = self.map_info.resolution
                goal_pts = []
                for dx, dy in fp_m:
                    gyaw = self.goal_yaw
                    rx = (dx * math.cos(gyaw) - dy * math.sin(gyaw)) / res
                    ry = (dx * math.sin(gyaw) + dy * math.cos(gyaw)) / res
                    goal_pts.append([int(gx + rx), int(gy - ry)]) # Trừ ry vì Y lật
                goal_pts = np.array(goal_pts, np.int32).reshape((-1, 1, 2))
                cv2.polylines(display_img, [goal_pts], True, (0, 255, 0), 2) # Vẽ hình xe màu Xanh Lá
                cv2.putText(display_img, "ROBOT (ACCEPTED)", (gx - 50, gy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # VẼ FOOTPRINT ĐIỂM GIAO HÀNG ĐỖ NGANG (MÀU CAM)
            if hasattr(self, 'deliver_px') and hasattr(self, 'deliver_yaw') and self.deliver_px:
                d_px, d_py = self.deliver_px
                d_yaw = -self.deliver_yaw # Lật Y
                d_pts = []
                for dx, dy in fp_m:
                    rx = (dx * math.cos(d_yaw) - dy * math.sin(d_yaw)) / res
                    ry = (dx * math.sin(d_yaw) + dy * math.cos(d_yaw)) / res
                    d_pts.append([int(d_px + rx), int(d_py + ry)])
                d_pts = np.array(d_pts, np.int32).reshape((-1, 1, 2))
                cv2.polylines(display_img, [d_pts], True, (0, 165, 255), 2) # BGR Màu Cam
                cv2.putText(display_img, "DELIVER (PARALLEL)", (d_px - 60, d_py - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        # Vẽ Robot (cập nhật theo Footprint thực tế)
        if self.robot_px and self.map_info:
            res = self.map_info.resolution
            fp_m = [(0.506, -0.32), (0.506, 0.32), (-0.454, 0.32), (-0.454, -0.32)]
            pts = []
            for dx, dy in fp_m:
                rx = (dx * math.cos(-self.robot_yaw) - dy * math.sin(-self.robot_yaw)) / res
                ry = (dx * math.sin(-self.robot_yaw) + dy * math.cos(-self.robot_yaw)) / res
                pts.append([int(self.robot_px[0] + rx), int(self.robot_px[1] + ry)])
            pts = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(display_img, [pts], (0, 165, 255))
            cv2.polylines(display_img, [pts], True, (0, 0, 0), 2)

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
        pixmap = self.pixmap()
        if not pixmap: return
        
        lbl_w, lbl_h = self.width(), self.height()
        pix_w, pix_h = pixmap.width(), pixmap.height()
        
        offset_x = (lbl_w - pix_w) / 2.0
        offset_y = (lbl_h - pix_h) / 2.0
        
        px_click = event.x() - offset_x
        py_click = event.y() - offset_y
        
        if px_click < 0 or px_click >= pix_w or py_click < 0 or py_click >= pix_h:
            return
            
        orig_w = self.map_info.width
        orig_h = self.map_info.height
        
        px = int(px_click * orig_w / pix_w)
        py = int(py_click * orig_h / pix_h)
        
        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        
        wx = ox + px * res
        wy = oy + (orig_h - py - 1) * res
        
        self.clicked_signal.emit(wx, wy, None)


