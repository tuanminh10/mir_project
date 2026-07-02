#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
import math
import time
import threading
import csv
from datetime import datetime

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
os.environ['YOLO_UPDATE_CHECK'] = 'False' # Tắt kiểm tra version YOLO trên Github để không bị lỗi 403 Rate Limit

# ==============================================================================
# HACK CHẶN TRIỆT ĐỂ WARNING RÁC CỦA TF2 (TF_REPEATED_DATA, TF_OLD_DATA) TỪ LÕI C++
# ==============================================================================
import threading
def filter_std(fileno):
    r, w = os.pipe()
    backup_fd = os.dup(fileno)
    os.dup2(w, fileno)
    os.close(w)
    f = os.fdopen(r, 'r')
    real_f = os.fdopen(backup_fd, 'w')
    def process():
        while True:
            try:
                line = f.readline()
                if not line: break
                if "TF_REPEATED_DATA" not in line and "TF_OLD_DATA" not in line and "buffer_core.cpp" not in line:
                    real_f.write(line)
                    real_f.flush()
            except: break
    threading.Thread(target=process, daemon=True).start()

filter_std(sys.stderr.fileno())
filter_std(sys.stdout.fileno())

from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QPushButton, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import pyrealsense2 as rs

def get_vn_name(target):
    if not target: return ""
    if target == "bep": return "bếp"
    if target == "sac": return "trạm sạc"
    if target.startswith("ban "): return target.replace("ban ", "bàn ")
    return target

_ros_sys_path = '/opt/ros/noetic/lib/python3/dist-packages'
if os.path.isdir(_ros_sys_path) and _ros_sys_path not in sys.path:
    sys.path.insert(1, _ros_sys_path)
    for mod_name in list(sys.modules.keys()):
        if any(mod_name.startswith(p) for p in ['geometry_msgs', 'nav_msgs', 'sensor_msgs',
                                                  'std_msgs', 'actionlib_msgs', 'tf2_msgs', 'move_base_msgs']):
            del sys.modules[mod_name]

import rospy
import tf
import tf.transformations
from nav_msgs.msg import OccupancyGrid, GridCells
from geometry_msgs.msg import PointStamped, Pose

# Thay đổi bằng file import nav của bạn
import navigationcacdiem as nav
from ultralytics import YOLO

import dongco
import mir_tts
import requests
import queue
import json
import threading
from std_msgs.msg import String

# ================= Utils =================

def get_depth_distance_m(depth_frame, box, frame_w, frame_h, center_pt=None):
    x1, y1, x2, y2 = map(int, box)
    width = x2 - x1
    height = y2 - y1
    
    # 1. Bắt chước kien.py: Cắt ROI chỉ lấy ngực/đầu (né bàn ở dưới và tường ở 2 bên)
    roi_x1 = max(0, int(x1 + width * 0.20))
    roi_x2 = min(frame_w, int(x2 - width * 0.20))
    roi_y1 = max(0, int(y1 + height * 0.05))
    roi_y2 = min(frame_h, int(y1 + height * 0.45))
    
    if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
        return -1.0
        
    distances = []
    step_x = max(1, (roi_x2 - roi_x1) // 20)
    step_y = max(1, (roi_y2 - roi_y1) // 20)
    
    for px in range(roi_x1, roi_x2, step_x):
        for py in range(roi_y1, roi_y2, step_y):
            orig_px = frame_w - 1 - px # Lật trục X vì camera lật
            if 0 <= orig_px < frame_w and 0 <= py < frame_h:
                d = depth_frame.get_distance(orig_px, py)
                # Bỏ qua nhiễu quá gần (<0.3m) để tránh dính ngón tay chĩa sát camera
                if 0.3 < d < 6.0: distances.append(d)
    
    if not distances:
        return -1.0
        
    # 2. Giải thuật lọc Foreground của kien.py (Tinh chỉnh lại)
    d_arr = np.array(distances)
    
    # KHI NGỒI XA: Người chiếm diện tích rất nhỏ trong ROI.
    # Giải pháp: Hạ xuống 5% để đảm bảo bắt trúng thân người, không bao giờ bắt nhầm tường (nền sau).
    p5_Z = np.percentile(d_arr, 5)
    
    # Lấy các điểm dao động xung quanh ngực (cho sai số lùi 0.2m để bao trọn nếp áo)
    mask = (d_arr >= p5_Z - 0.2) & (d_arr <= p5_Z + 0.5)
    person_pts = d_arr[mask]
    
    if len(person_pts) < 5:
        person_pts = d_arr # Fallback an toàn nếu mask quá gắt
        
    return float(np.median(person_pts))

# Tích hợp Segmentation & 2D Radius Filter (Morphological Erosion)
def get_depth_distance_m_seg(depth_frame, binary_mask, frame_w, frame_h):
    # 1. Cắt gọt hình thái học (Morphological Erosion) - Tương đương Radius Outlier Removal 3D
    kernel = np.ones((7, 7), np.uint8) # Gọt mạnh viền (7x7) để tránh hoàn toàn nhiễu từ nền phía sau
    eroded_mask = cv2.erode(binary_mask, kernel, iterations=1)
    
    ys, xs = np.where(eroded_mask == 1)
    if len(ys) == 0:
        return -1.0
        
    distances = []
    # Lấy ngẫu nhiên tối đa 400 điểm để tối ưu tốc độ tính toán (Đạt > 60 FPS)
    step = max(1, len(ys) // 400)
    
    for i in range(0, len(ys), step):
        py = int(ys[i])
        px = int(xs[i])
        orig_px = frame_w - 1 - px # Lật trục X vì RGB đã bị cv2.flip(1)
        
        if 0 <= orig_px < frame_w and 0 <= py < frame_h:
            d = depth_frame.get_distance(orig_px, py)
            if 0.3 < d < 6.0: distances.append(d)
            
    if not distances: return -1.0
        
    d_arr = np.array(distances)
    p5_Z = np.percentile(d_arr, 5)
    
    mask = (d_arr >= p5_Z - 0.2) & (d_arr <= p5_Z + 0.5)
    person_pts = d_arr[mask]
    if len(person_pts) < 5: person_pts = d_arr
        
    return float(np.median(person_pts))
def get_person_relative_position_m(depth_frame, center_pt, frame_w, frame_h, depth_intrinsics, distance_m):
    import math
    import pyrealsense2 as rs
    if len(center_pt) == 4:
        x1, y1, x2, y2 = map(int, center_pt)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
    else:
        center_x, center_y = map(int, center_pt)
        
    # LẬT LẠI TRỤC X: Do camera gắn ngược hoặc ảnh hiển thị bị lật,
    # CẦN PHẢI GIỮ nguyên phép lật này thì tọa độ Trái/Phải mới đúng với base_link!
    orig_px = float(frame_w - 1 - center_x)
    orig_py = float(center_y)
    
    if distance_m <= 0: return None
    
    if depth_intrinsics is not None:
        point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [orig_px, orig_py], distance_m)
        x_opt, y_opt, z_opt = point_3d[0], point_3d[1], point_3d[2]
    else:
        hfov_rad = math.radians(69.0)
        vfov_rad = math.radians(42.0)
        angle_x = ((orig_px - frame_w / 2.0) / frame_w) * hfov_rad
        angle_y = ((orig_py - frame_h / 2.0) / frame_h) * vfov_rad
        x_opt = distance_m * math.tan(angle_x)
        y_opt = distance_m * math.tan(angle_y)
        z_opt = distance_m

    pitch_rad = math.radians(20.0)
    # CÔNG THỨC CHUẨN ĐÃ ĐƯỢC GIÁO SƯ CHỨNG MINH:
    forward_m_camera = z_opt * math.cos(pitch_rad) - y_opt * math.sin(pitch_rad)
    
    # BÙ TRỪ TRỤC QUANG HỌC VÀ VỊ TRÍ LẮP ĐẶT (Camera Calibration):
    lateral_offset_m = 0.0 # Giáo sư yêu cầu bỏ bù trái/phải
    forward_offset_m = 0.05 # Bù khung motor nhô ra trước 10cm (Từ đuôi lên đầu)
    
    forward_m = forward_m_camera + forward_offset_m
    left_m = -x_opt + lateral_offset_m
    
    # CÔNG THỨC CHUẨN Z:
    down_m = z_opt * math.sin(pitch_rad) + y_opt * math.cos(pitch_rad)
    camera_height_m = 1.8
    z_m = camera_height_m - down_m
    
    return forward_m, left_m, z_m


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
            # (Đã ẩn đoạn footprint nhỏ theo yêu cầu để tránh vẽ chồng 2 hình chữ nhật lên nhau)

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
        
        # BÙ TRỪ SCALE VÀ OFFSET CỦA GIAO DIỆN (Tránh lỗi bấm lệch)
        scaled_w = self.pixmap().width()
        scaled_h = self.pixmap().height()
        x_offset = (self.width() - scaled_w) // 2
        y_offset = (self.height() - scaled_h) // 2
        
        mx = event.x() - x_offset
        my = event.y() - y_offset
        
        if not (0 <= mx < scaled_w and 0 <= my < scaled_h): return

        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        h = self.map_info.height
        
        px = int(mx * (self.map_info.width / scaled_w))
        py = int(my * (self.map_info.height / scaled_h))
        
        wx = ox + px * res
        wy = oy + (h - py - 1) * res
        
        self.clicked_signal.emit(wx, wy, None)


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    target_locked_signal = pyqtSignal(float, float, object)
    status_update_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.pause_emit = False
        self.device = 0 if os.path.exists('/opt/ai_venv/bin/python') else 'cpu'
        self.model_pose = YOLO('yolo11s-pose.pt')
        self.model_seg = YOLO('yolo11s-seg.pt')
        if self.device == 0:
            self.model_pose.to('cuda')
            self.model_seg.to('cuda')
            
        self.locked_target_id = None
        self.robot_state = "IDLE"
        self.locked_bbox = None
        
        self.hand_raise_start = {}
        self.open5_confirm_count = {}
        self.target_history = {} # Lưu đệm tọa độ trong 2s
        
        self.fist_confirm_count = 0
        self.fist_hold_start = None

        self.latest_rgb_frame = None
        self.frame_lock = threading.Lock()

    def run(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        try:
            profile = self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)
            self.pc = rs.pointcloud()
            
            # --- CẤU HÌNH CẢM BIẾN COLOR ĐỂ GIẢM LÓA & MỜ ---
            color_sensor = profile.get_device().first_color_sensor()
            if color_sensor:
                # Trả lại Auto-Exposure để không bị lóa khi phòng có đèn sáng chói
                color_sensor.set_option(rs.option.enable_auto_exposure, 1)
                
                # BÍ QUYẾT: Tắt tính năng "Auto-Exposure Priority" (Ưu tiên phơi sáng).
                # Khi tắt, camera BẮT BUỘC phải giữ framerate 30fps bằng mọi giá, 
                # nên nó KHÔNG THỂ tăng thời gian phơi sáng quá dài -> Cắt đứt hoàn toàn hiện tượng Motion Blur (nhòe ảnh)
                if color_sensor.supports(rs.option.auto_exposure_priority):
                    color_sensor.set_option(rs.option.auto_exposure_priority, 0)
                    
                # Vẫn giữ Sharpness ở mức cao để nét hình
                if color_sensor.supports(rs.option.sharpness):
                    color_sensor.set_option(rs.option.sharpness, 100)
            # ------------------------------------------------------------------
            
            # --- CẤU HÌNH CẢM BIẾN DEPTH ĐỂ TĂNG ĐỘ CHÍNH XÁC KHI ĐỨNG XA ---
            depth_sensor = profile.get_device().first_depth_sensor()
            if depth_sensor:
                # Tăng công suất phát tia hồng ngoại (Laser Power) lên tối đa (Thường là 360mW)
                # Tia hồng ngoại mạnh hơn sẽ dội lại rõ hơn khi khách hàng ngồi xa > 5m
                if depth_sensor.supports(rs.option.laser_power):
                    depth_sensor.set_option(rs.option.laser_power, 360)
                
                # Bật chế độ "High Accuracy" (Preset 3) trực tiếp trên chip xử lý của camera
                # Giúp camera tự động gọt bỏ các pixel bị nhiễu viền ở khoảng cách xa
                if depth_sensor.supports(rs.option.visual_preset):
                    try:
                        depth_sensor.set_option(rs.option.visual_preset, 3) # 3 là High Accuracy
                    except Exception as e:
                        print(f"[CẢNH BÁO] Không thể set High Accuracy Preset: {e}")
            # ------------------------------------------------------------------

            # QUAN TRỌNG: Lấy intrinsics của COLOR stream vì depth đã được align sang color
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            self.depth_intrinsics = color_profile.get_intrinsics()
            
            print("[INFO] Đã KẾT NỐI RealSense (Đã đồng bộ Intrinsics Color) và Cấu hình Sensor!")
        except Exception as e:
            print(f"[ERROR] RealSense: {e}")
            return

        self.tf_listener = tf.TransformListener()

        while self._run_flag:
            try: frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            except: continue
                
            aligned = self.align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame: continue

            frame = np.asanyarray(color_frame.get_data())
            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            
            with self.frame_lock:
                self.latest_rgb_frame = frame.copy()

            results_pose = []
            results_seg = []
            
            # TỐI ƯU HÓA HIỆU NĂNG: Chỉ chạy YOLO Pose và Seg khi đang ở trạng thái TÌM KHÁCH
            # Tránh việc Camera và GPU chạy 100% công suất 24/7 gây lag/quá nhiệt Jetson
            if getattr(self, 'is_scanning_for_hand', False):
                if self.robot_state in ["IDLE", "STANDBY (SAVING GPU)"]: 
                    self.robot_state = "SCANNING"
                results_pose = self.model_pose.track(frame, conf=0.45, persist=True, tracker="bytetrack.yaml", verbose=False, half=(self.device==0), device=self.device)
                results_seg = self.model_seg.predict(frame, conf=0.45, verbose=False, half=(self.device==0), device=self.device)
            else:
                self.robot_state = "STANDBY (SAVING GPU)"
                self.locked_target_id = None
            
            curr_time = time.time()
            annotated_frame = frame.copy()

            for result_pose in results_pose:
                boxes = result_pose.boxes
                if boxes is None or boxes.id is None: continue
                
                keypoints = result_pose.keypoints.data if result_pose.keypoints is not None else None
                seg_result = results_seg[0] if len(results_seg) > 0 else None
                masks = seg_result.masks.xy if (seg_result and seg_result.masks is not None) else None

                # LỌC NGƯỜI GIƠ TAY BẰNG GIẢI PHẪU (ANATOMICAL LOGIC) QUA YOLO POSE
                raising_hands_ids = set()
                if keypoints is not None:
                    kpts_xy = result_pose.keypoints.xy.cpu().numpy()
                    kpts_conf = result_pose.keypoints.conf.cpu().numpy() if result_pose.keypoints.conf is not None else None
                    for j, box_j in enumerate(boxes.xyxy.cpu().numpy()):
                        if kpts_conf is None: continue
                        kp = kpts_xy[j]
                        cf = kpts_conf[j]
                        box_h = box_j[3] - box_j[1]
                        t_id = int(boxes.id[j].item())
                        
                        def get_valid_arm(kp_wrist, kp_elbow, side):
                            if cf[kp_wrist] > 0.25 and cf[kp_elbow] > 0.25:
                                wx, wy = kp[kp_wrist]
                                ex, ey = kp[kp_elbow]
                                forearm_len = math.hypot(wx - ex, wy - ey)
                                
                                is_pointing_up = (ey - wy) > (forearm_len * 0.85)
                                is_long_enough = forearm_len > max(30, box_h * 0.1)
                                
                                shoulder_idx = 5 if side == 'L' else 6
                                if cf[shoulder_idx] > 0.3:
                                    is_elbow_raised = ey < kp[shoulder_idx][1]
                                elif cf[0] > 0.3: # NOSE
                                    is_elbow_raised = ey < (kp[0][1] + forearm_len * 0.5)
                                else:
                                    is_elbow_raised = ey < (box_j[1] + box_h * 0.3)
                                    
                                if cf[0] > 0.3:
                                    is_high_enough = wy < (kp[0][1] - forearm_len * 0.4)
                                else:
                                    is_high_enough = wy < box_j[1]
                                    
                                if is_pointing_up and is_long_enough and is_elbow_raised and is_high_enough:
                                    return True
                            return False
                            
                        # KP_L_WRIST=9, KP_L_ELBOW=7, KP_R_WRIST=10, KP_R_ELBOW=8
                        if get_valid_arm(9, 7, 'L') or get_valid_arm(10, 8, 'R'):
                            raising_hands_ids.add(t_id)

                for i, box in enumerate(boxes):
                    track_id = int(box.id[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    binary_mask = None
                    if seg_result and seg_result.masks is not None:
                        # THUẬT TOÁN CENTER DISTANCE: Tìm đúng cái mask thuộc về người này (Tránh lấy nhầm người ngồi cạnh)
                        best_dist = float('inf')
                        best_mask_idx = -1
                        
                        target_cx = (x1 + x2) / 2
                        target_cy = (y1 + y2) / 2
                        
                        for m_idx, seg_box in enumerate(seg_result.boxes.xyxy.cpu().numpy()):
                            if int(seg_result.boxes.cls[m_idx].item()) != 0: continue # Chỉ lấy class 0 (Person)
                            
                            sx1, sy1, sx2, sy2 = seg_box
                            mcx = (sx1 + sx2) / 2
                            mcy = (sy1 + sy2) / 2
                            
                            dist = math.hypot(target_cx - mcx, target_cy - mcy)
                            if dist < best_dist and dist < 100: # Lệch tâm tối đa 100px
                                best_dist = dist
                                best_mask_idx = m_idx
                                
                        if best_mask_idx != -1:
                            mask_raw = seg_result.masks.data[best_mask_idx].cpu().numpy()
                            binary_mask = cv2.resize(mask_raw, (frame_w, frame_h))
                            binary_mask = (binary_mask > 0.5).astype(np.uint8)
                        
                        # --- TRỰC QUAN HÓA THUẬT TOÁN GỌT VIỀN LÊN GIAO DIỆN CHỨNG MINH CHO ĐỒ ÁN ---
                        try:
                            # 1. Vẽ đường viền gốc (Chưa gọt, còn nhiễu) -> Màu Đỏ nét mỏng
                            contours_raw, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(annotated_frame, contours_raw, -1, (0, 0, 255), 1)
                            
                            # 2. Gọt viền và vẽ đường viền ĐÃ LỌC (Sạch tuyệt đối) -> Màu Xanh Lá nét dày
                            kernel_vis = np.ones((7, 7), np.uint8)
                            eroded_mask_vis = cv2.erode(binary_mask, kernel_vis, iterations=1)
                            contours_eroded, _ = cv2.findContours(eroded_mask_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(annotated_frame, contours_eroded, -1, (0, 255, 0), 2)
                        except Exception as e:
                            pass # Bỏ qua nếu lỗi thư viện đồ họa để tránh sập luồng chính
                    
                    person_center_x, person_center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    
                    # SỬA LỖI TỌA ĐỘ CUSTOMER BỊ LỆCH (X tăng, Y giảm):
                    # Khi người dùng giơ tay lên, bounding box bị phình to về phía cánh tay.
                    # Điều này làm tâm bounding box bị lệch sang một bên, khiến tọa độ Y trên map bị tụt và X bị tăng!
                    # Giải pháp: Dùng điểm giữa 2 vai (keypoint 5 và 6) làm tâm thực sự của cơ thể.
                    if keypoints is not None and i < len(keypoints):
                        kp = keypoints[i]
                        if len(kp) >= 7:
                            sh_l = kp[5]
                            sh_r = kp[6]
                            nose = kp[0]
                            # BẮT BUỘC kiểm tra độ tin cậy (Confidence > 0.4) để tránh AI nhận diện bậy bạ
                            def v_kpt(k): return len(k) >= 3 and k[2].item() > 0.4
                            
                            if v_kpt(sh_l) and v_kpt(sh_r):
                                person_center_x = (sh_l[0].item() + sh_r[0].item()) / 2
                                person_center_y = (sh_l[1].item() + sh_r[1].item()) / 2
                            elif v_kpt(nose):
                                person_center_x = nose[0].item()
                                person_center_y = nose[1].item()
                
                    # MULTI-LAYER LIDAR FALLBACK (3 lớp bảo vệ chống xuyên tường)
                    body_distances = []
                    depth_layer_used = "none"
                    if keypoints is not None and i < len(keypoints):
                        kp = keypoints[i]
                        
                        # LỚP 1: Quét KHUÔN MẶT (Keypoints 0-4: Mũi, Mắt, Tai)
                        # Khuôn mặt phản xạ IR tốt nhất, nằm trọn trong cơ thể.
                        for k_idx in range(5): 
                            if len(kp) > k_idx and len(kp[k_idx]) >= 3 and kp[k_idx][2].item() > 0.4:
                                kx, ky = int(kp[k_idx][0].item()), int(kp[k_idx][1].item())
                                for dx in range(-7, 8, 3):
                                    for dy in range(-7, 8, 3):
                                        px, py = kx + dx, ky + dy
                                        if 0 <= px < frame_w and 0 <= py < frame_h:
                                            orig_px = frame_w - 1 - px
                                            d = depth_frame.get_distance(orig_px, py)
                                            if 0.2 < d < 6.0: body_distances.append(d)
                        
                        if body_distances:
                            depth_layer_used = "face"
                        else:
                            # LỚP 2: Quét VAI + HÔNG (Keypoints 5,6,11,12)
                            # Khi mặt bị khuất nhưng thân vẫn thấy rõ.
                            for k_idx in [5, 6, 11, 12]:
                                if len(kp) > k_idx and len(kp[k_idx]) >= 3 and kp[k_idx][2].item() > 0.4:
                                    kx, ky = int(kp[k_idx][0].item()), int(kp[k_idx][1].item())
                                    for dx in range(-10, 11, 4):
                                        for dy in range(-10, 11, 4):
                                            px, py = kx + dx, ky + dy
                                            if 0 <= px < frame_w and 0 <= py < frame_h:
                                                orig_px = frame_w - 1 - px
                                                d = depth_frame.get_distance(orig_px, py)
                                                if 0.2 < d < 6.0: body_distances.append(d)
                            if body_distances:
                                depth_layer_used = "body"
                                        
                    if body_distances:
                        body_distances.sort()
                        if depth_layer_used == "face":
                            d_m = float(body_distances[int(len(body_distances) * 0.3)])
                        else:
                            # Vai/Hông có thể lem ra ngoài -> dùng 15% an toàn hơn
                            d_m = float(body_distances[int(len(body_distances) * 0.15)])
                    else:
                        # LỚP 3: CẮT GỌT THEO SEGMENTATION VÀ 2D RADIUS FILTER
                        d_m = -1.0
                        if binary_mask is not None:
                            d_m = get_depth_distance_m_seg(depth_frame, binary_mask, frame_w, frame_h)
                        
                        # LỚP 4 (Fallback cuối cùng): Quét BBox cũ nếu Segmentation thất bại
                        if binary_mask is None or d_m <= 0:
                            d_m = get_depth_distance_m(depth_frame, (x1, y1, x2, y2), frame_w, frame_h, (person_center_x, person_center_y))
                    d_ngang_m = d_m
                
                    is_raising = False
                    if getattr(self, "is_scanning_for_hand", False) and track_id in raising_hands_ids:
                        is_raising = True

                    if track_id != -1 and self.locked_target_id is None:
                        # Tăng khoảng cách nhận diện từ 5.0m lên 8.0m vì YOLO nhận diện được rất xa
                        if is_raising and d_ngang_m <= 6.5:
                            self.open5_confirm_count[track_id] = self.open5_confirm_count.get(track_id, 0) + 1
                            if track_id not in self.hand_raise_start:
                                self.hand_raise_start[track_id] = curr_time
                                self.target_history[track_id] = []
                        else:
                            count = self.open5_confirm_count.get(track_id, 0)
                            if count > 0: self.open5_confirm_count[track_id] = count - 1
                            else: 
                                self.hand_raise_start.pop(track_id, None)
                                self.target_history.pop(track_id, None)
                    
                        if self.open5_confirm_count.get(track_id, 0) >= 2:
                            if track_id in self.hand_raise_start:
                                # Tích lũy (buffer) tọa độ tâm và khoảng cách nếu d_ngang_m hợp lệ
                                if d_ngang_m > 0 and d_ngang_m < 6.0:
                                    self.target_history[track_id].append((person_center_x, person_center_y, d_ngang_m))

                                hold_time = curr_time - self.hand_raise_start[track_id]
                                cv2.putText(annotated_frame, f"DANG KHOA TARGET: {hold_time:.1f}s/2s", (int(x1), int(y1)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                if hold_time >= 2.0:
                                    self.hand_raise_start.pop(track_id, None)
                                    self.open5_confirm_count[track_id] = 0
                                
                                    # 3. CHỐT TỌA ĐỘ BẰNG CÁCH LẤY TRUNG BÌNH (MEDIAN) CỦA POINTCLOUD TRONG VÙNG NGỰC
                                    hist = self.target_history.pop(track_id, [])
                                    
                                    # TÍNH HỆ QUY CHIẾU (ANCHOR) TỪ 60 KHUNG HÌNH (2 GIÂY VỪA QUA) ĐỂ KIỂM TRA CHÉO
                                    avg_cx, avg_cy, avg_dm = person_center_x, person_center_y, d_m
                                    if len(hist) > 0:
                                        avg_cx = float(np.median([h[0] for h in hist]))
                                        avg_cy = float(np.median([h[1] for h in hist]))
                                        avg_dm = float(np.median([h[2] for h in hist]))
                                        
                                    # Tạo PointCloud 3D
                                    self.pc.map_to(color_frame)
                                    points = self.pc.calculate(depth_frame)
                                    vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(frame_h, frame_w, 3)
                                    
                                    # Tích hợp Segmentation Mask vào PointCloud để độ chính xác tuyệt đối
                                    pc_success = False
                                    if binary_mask is not None:
                                        # binary_mask là ảnh đã flip. Cần unflip để khớp với vertices (ảnh gốc từ Lidar)
                                        unflipped_mask = cv2.flip(binary_mask, 1)
                                        
                                        # Lấy điểm PointCloud nằm trong Mask (Và có cự ly hợp lệ)
                                        valid_mask = (unflipped_mask > 0.7) & (vertices[:, :, 2] > 0.3) & (vertices[:, :, 2] < 6.0)
                                        valid_pts = vertices[valid_mask]
                                        
                                        if len(valid_pts) > 10:
                                            Z_values = valid_pts[:, 2]
                                            if avg_dm > 0:
                                                # Giáo sư Antigravity: Dùng mỏ neo Pose (avg_dm) làm trung tâm. 
                                                # Khoanh vùng +- 0.3m để BẮT CHUẨN cái Thân Người (Ngực/Bụng).
                                                # Tự động loại bỏ cánh tay vươn ra (quá gần) và mảng tường (quá xa).
                                                person_mask = (Z_values >= avg_dm - 0.3) & (Z_values <= avg_dm + 0.3)
                                            else:
                                                p15_Z = np.percentile(Z_values, 15)
                                                person_mask = (Z_values >= p15_Z - 0.1) & (Z_values <= p15_Z + 0.5)
                                                
                                            person_pts = valid_pts[person_mask]
                                            if len(person_pts) < 5: person_pts = valid_pts
                                            
                                            median_pt = np.median(person_pts, axis=0)
                                            x_opt, y_opt, z_opt = float(median_pt[0]), float(median_pt[1]), float(median_pt[2])
                                            
                                            # KIỂM TRA CHÉO (CROSS-VALIDATION) VỚI LỊCH SỬ 60 KHUNG HÌNH (ANCHOR)
                                            if avg_dm > 0 and abs(z_opt - avg_dm) > 0.4:
                                                print(f"[SMART NAV] ❌ CẢNH BÁO: PointCloud bị nhiễu (Z={z_opt:.2f}m khác xa Anchor={avg_dm:.2f}m). Từ chối PointCloud!")
                                                pc_success = False
                                            else:
                                                print(f"[SMART NAV] Ổn định tọa độ bằng Segmentation + PointCloud (Z_raw={z_opt:.3f}m, X={-x_opt:.3f}m)")
                                                
                                                pitch_rad = math.radians(20.0)
                                                forward_m_camera = z_opt * math.cos(pitch_rad) - y_opt * math.sin(pitch_rad)
                                                forward_m = forward_m_camera + 0.00 - 0.26 
                                                left_m = -x_opt
                                                
                                                # HIỆU CHUẨN THỰC NGHIỆM: Hàm bù trừ Bậc 2 (Polynomial Fit)
                                                # Fit từ 40 mẫu TN3 calibration (3m/3.5m/4m/4.5m), RMSE=0.062m
                                                if forward_m > 2.5:
                                                    fwd_raw = forward_m
                                                    forward_m = -0.042954 * fwd_raw**2 + 1.057498 * fwd_raw + 0.111814
                                                    # Hiệu chỉnh tỷ lệ luôn cho trục ngang X (left_m)
                                                    left_m = left_m * (forward_m / fwd_raw)
                                                    print(f"[CALIBRATION] Bù trừ thực nghiệm: Fwd {fwd_raw:.3f}→{forward_m:.3f}m | Left scale: x{forward_m/fwd_raw:.2f}")
                                                
                                                pc_success = True
                                            
                                    if not pc_success:
                                        # Nếu không có Segmentation, Fallback về cắt BBox (Hình chữ nhật)
                                        roi_w = x2 - x1
                                        roi_h = y2 - y1
                                        roi_x1 = max(0, int(x1 + roi_w * 0.20))
                                        roi_x2 = min(frame_w, int(x2 - roi_w * 0.20))
                                        roi_y1 = max(0, int(y1 + roi_h * 0.05))
                                        roi_y2 = min(frame_h, int(y1 + roi_h * 0.45))
                                        
                                        if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                                            unflip_x1 = frame_w - roi_x2
                                            unflip_x2 = frame_w - roi_x1
                                            roi_pts = vertices[roi_y1:roi_y2, unflip_x1:unflip_x2]
                                            valid_mask = (roi_pts[:, :, 2] > 0.3) & (roi_pts[:, :, 2] < 6.0)
                                            valid_pts = roi_pts[valid_mask]
                                            
                                            if len(valid_pts) > 10:
                                                Z_values = valid_pts[:, 2]
                                                if avg_dm > 0:
                                                    person_mask = (Z_values >= avg_dm - 0.3) & (Z_values <= avg_dm + 0.3)
                                                else:
                                                    p15_Z = np.percentile(Z_values, 15)
                                                    person_mask = (Z_values >= p15_Z - 0.1) & (Z_values <= p15_Z + 0.5)
                                                    
                                                person_pts = valid_pts[person_mask]
                                                if len(person_pts) < 5: person_pts = valid_pts
                                                
                                                median_pt = np.median(person_pts, axis=0)
                                                x_opt, y_opt, z_opt = float(median_pt[0]), float(median_pt[1]), float(median_pt[2])
                                                
                                                # KIỂM TRA CHÉO (CROSS-VALIDATION) VỚI LỊCH SỬ 60 KHUNG HÌNH (ANCHOR)
                                                if avg_dm > 0 and abs(z_opt - avg_dm) > 0.4:
                                                    print(f"[SMART NAV] ❌ CẢNH BÁO: PointCloud bị nhiễu (Z={z_opt:.2f}m khác xa Anchor={avg_dm:.2f}m). Từ chối PointCloud!")
                                                    pc_success = False
                                                else:
                                                    print(f"[SMART NAV] Ổn định tọa độ bằng BBox + PointCloud (Z_raw={z_opt:.3f}m, X={-x_opt:.3f}m)")
                                                    
                                                    pitch_rad = math.radians(20.0)
                                                    forward_m_camera = z_opt * math.cos(pitch_rad) - y_opt * math.sin(pitch_rad)
                                                    forward_m = forward_m_camera + 0.00 - 0.26
                                                    left_m = -x_opt
                                                    
                                                    # HIỆU CHUẨN THỰC NGHIỆM: Hàm bù trừ Bậc 2 (Polynomial Fit)
                                                    if forward_m > 2.5:
                                                        fwd_raw = forward_m
                                                        forward_m = -0.042954 * fwd_raw**2 + 1.057498 * fwd_raw + 0.111814
                                                        # Hiệu chỉnh tỷ lệ luôn cho trục ngang X (left_m)
                                                        left_m = left_m * (forward_m / fwd_raw)
                                                        print(f"[CALIBRATION] Bù trừ thực nghiệm: Fwd {fwd_raw:.3f}→{forward_m:.3f}m | Left scale: x{forward_m/fwd_raw:.2f}")
                                                    
                                                    pc_success = True
                                                
                                    if not pc_success:
                                        # Fallback cuối cùng nếu PointCloud hoàn toàn mù hoặc bị từ chối
                                        print(f"[SMART NAV] Ổn định tọa độ bằng Lidar Fallback 4 Lớp (Z={avg_dm:.3f}m)")
                                        rel = get_person_relative_position_m(depth_frame, (avg_cx, avg_cy), frame_w, frame_h, self.depth_intrinsics, avg_dm)
                                        if rel is not None:
                                            forward_m = rel[0] - 0.26; left_m = rel[1]
                                        else:
                                            forward_m = left_m = 0
                                        
                                    self.locked_target_id = track_id
                                    self.locked_bbox = (x1, y1, x2, y2)
                                    self.robot_state = "COLLECTING"
                                    self.status_update_signal.emit(f"ĐÃ KHÓA ! Đang tải dữ liệu không gian...")
                                    # ============ LOG TN3 ĐẦY ĐỦ ============
                                    # Tính song song tất cả 4 lớp (không dùng fallback)
                                    d_L1 = d_L2 = d_L3 = d_L4 = -1.0
                                    if keypoints is not None and i < len(keypoints):
                                        kp_tn3 = keypoints[i]
                                        # Lớp 1: Mặt
                                        face_d = []
                                        for k_idx in range(5):
                                            if len(kp_tn3) > k_idx and len(kp_tn3[k_idx]) >= 3 and kp_tn3[k_idx][2].item() > 0.4:
                                                kx, ky = int(kp_tn3[k_idx][0].item()), int(kp_tn3[k_idx][1].item())
                                                for ddx in range(-7, 8, 3):
                                                    for ddy in range(-7, 8, 3):
                                                        px2, py2 = kx+ddx, ky+ddy
                                                        if 0 <= px2 < frame_w and 0 <= py2 < frame_h:
                                                            orig_px_tn3 = frame_w - 1 - px2
                                                            dd = depth_frame.get_distance(orig_px_tn3, py2)
                                                            if 0.2 < dd < 6.0: face_d.append(dd)
                                        if face_d:
                                            face_d.sort()
                                            d_L1 = float(face_d[int(len(face_d)*0.3)])
                                        # Lớp 2: Vai/Hông
                                        body_d = []
                                        for k_idx in [5,6,11,12]:
                                            if len(kp_tn3) > k_idx and len(kp_tn3[k_idx]) >= 3 and kp_tn3[k_idx][2].item() > 0.4:
                                                kx, ky = int(kp_tn3[k_idx][0].item()), int(kp_tn3[k_idx][1].item())
                                                for ddx in range(-10, 11, 4):
                                                    for ddy in range(-10, 11, 4):
                                                        px2, py2 = kx+ddx, ky+ddy
                                                        if 0 <= px2 < frame_w and 0 <= py2 < frame_h:
                                                            orig_px_tn3 = frame_w - 1 - px2
                                                            dd = depth_frame.get_distance(orig_px_tn3, py2)
                                                            if 0.2 < dd < 6.0: body_d.append(dd)
                                        if body_d:
                                            body_d.sort()
                                            d_L2 = float(body_d[int(len(body_d)*0.15)])
                                        # Lớp 3: Seg
                                        if binary_mask is not None:
                                            d_L3 = get_depth_distance_m_seg(depth_frame, binary_mask, frame_w, frame_h)
                                        # Lớp 4: BBox
                                        d_L4 = get_depth_distance_m(depth_frame, (x1,y1,x2,y2), frame_w, frame_h, (person_center_x, person_center_y))

                                    print(f"\n[TN3] ===== BÁO CÁO ĐẦY ĐỦ =====")
                                    print(f"[TN3] Tư thế: Ngồi bàn / Đứng thẳng")
                                    print(f"[TN3] 4 Lớp Depth: L1={d_L1:.3f}m | L2={d_L2:.3f}m | L3={d_L3:.3f}m | L4={d_L4:.3f}m")
                                    print(f"[TN3] Anchor (avg_dm): {avg_dm:.3f}m | Lớp đã dùng: {depth_layer_used}")
                                    print(f"[TN3] PointCloud: z_opt={'N/A' if not pc_success else f'{z_opt:.3f}m'} | Phương pháp chốt: {'PointCloud+Seg' if pc_success else 'rs2_deproject (Fallback)'}")
                                    print(f"[TN3] Kết quả cuối: forward_m={forward_m:.3f}m | left_m={left_m:.3f}m")
                                    print(f"[TN3] ================================")
                                    
                                    # LƯU VÀO FILE CSV ĐỂ NGHIÊN CỨU HIỆU CHUẨN
                                    csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tn3chuan.csv")
                                    file_exists = os.path.isfile(csv_file)
                                    try:
                                        with open(csv_file, mode='a', newline='') as f:
                                            writer = csv.writer(f)
                                            if not file_exists:
                                                writer.writerow(["Time", "Z_raw", "Forward_m_raw", "Forward_m_Calibrated", "L1", "L2", "L3", "L4", "Avg_DM"])
                                            
                                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                            z_raw_val = z_opt if pc_success else avg_dm
                                            fw_cam = z_raw_val * math.cos(math.radians(20.0)) - (y_opt if pc_success else avg_cy) * math.sin(math.radians(20.0))
                                            fw_raw = fw_cam - 0.26
                                            writer.writerow([timestamp, f"{z_raw_val:.3f}", f"{fw_raw:.3f}", f"{forward_m:.3f}", f"{d_L1:.3f}", f"{d_L2:.3f}", f"{d_L3:.3f}", f"{d_L4:.3f}", f"{avg_dm:.3f}"])
                                        print(f"✅ Đã tự động lưu dữ liệu đã HIỆU CHUẨN vào: {csv_file}")
                                    except Exception as e:
                                        print(f"❌ Lỗi khi lưu CSV: {e}")
                                        
                                    # ============ HẾT LOG TN3 ============
                                
                                    print(f"\n{'='*60}")
                                    print(f"[DEBUG VISION] PC base_link = fwd:{forward_m:.3f}, left:{left_m:.3f}")
                                    print(f"{'='*60}")
                                    
                                    msg = PointStamped()
                                    msg.header.stamp = rospy.Time(0)
                                    msg.header.frame_id = "base_link"
                                    msg.point.x = forward_m
                                    msg.point.y = left_m
                                    msg.point.z = 0.0
                                
                                    try:
                                        self.tf_listener.waitForTransform("/map", "base_link", rospy.Time(0), rospy.Duration(2.0))
                                        pt = self.tf_listener.transformPoint("/map", msg)
                                        # [TN3] VÔ HIỆU HÓA DI CHUYỂN ĐỂ CHỈ ĐO ĐẠC!
                                        # self.target_locked_signal.emit(pt.point.x, pt.point.y, None) 
                                        print("[TN3] Đã khóa đo đạc xong! (LOCKED)")
                                        print("[TN3] => BẤM PHÍM 'SPACE' TRÊN BÀN PHÍM ĐỂ ĐO LẦN TIẾP THEO!")
                                        self.robot_state = "LOCKED"
                                        # Vẫn giữ nguyên self.locked_target_id để nó không tự động quét lại
                                    except Exception as e:
                                        print(f"Lỗi TF: {e}")
                                        self.robot_state = "LOCKED"

                    # Đã loại bỏ logic has_fist (Nắm tay hủy lệnh) vì file này không dùng Mediapipe nữa
                    is_too_close = (0 < d_ngang_m < 1.0)
                    is_invalid = (d_ngang_m <= 0.0 or d_ngang_m > 5.0)
                
                    if self.locked_target_id is not None:
                        if track_id == self.locked_target_id:
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 3)
                            cv2.putText(annotated_frame, "LOCKED TARGET", (int(x1), int(y1)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        if not is_invalid:
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            cv2.putText(annotated_frame, f"STATE: {self.robot_state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) if self.robot_state != "IDLE" else (0,255,0), 2)
            if getattr(self, 'pause_emit', False) == False:
                self.change_pixmap_signal.emit(annotated_frame)
            
        self.pipeline.stop()

    def get_latest_frame(self):
        with self.frame_lock:
            if self.latest_rgb_frame is not None:
                return self.latest_rgb_frame.copy()
            return None

    def stop(self):
        self._run_flag = False
        self.wait()

# ================= Main App =================
class MainApp(QMainWindow):
    map_signal = pyqtSignal(object)
    pose_signal = pyqtSignal(float, float, float)
    request_gui_update_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TN3 - ĐO ĐẠC 3D PIPELINE")
        self.resize(1600, 900)
        
        self.central_widget = QWidget()
        self.layout = QHBoxLayout(self.central_widget)
        
        self.left_panel = QVBoxLayout()
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.left_panel.addWidget(self.camera_label, 1)
        
        self.layout.addLayout(self.left_panel, 1)
        
        self.map_label = MapLabel()
        self.map_label.setAlignment(Qt.AlignCenter)
        self.map_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout.addWidget(self.map_label, 1)
        
        self.setCentralWidget(self.central_widget)

        rospy.init_node('tn3_measurement', anonymous=True)
        
        self.map_signal.connect(self.map_label.set_map)
        self.pose_signal.connect(self.map_label.set_robot_pose)
        self.request_gui_update_signal.connect(self.map_label.update_view)

        self.video_thread = VideoThread()
        self.video_thread.change_pixmap_signal.connect(self.update_camera_image)
        
        # [TN3] ÉP LUÔN LUÔN QUÉT TAY (Không cần qua bước ấn IoT hay GUEST_CALL)
        self.video_thread.is_scanning_for_hand = True
        self.video_thread.start()

        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        rospy.Subscriber('/robot_pose', Pose, self.pose_callback)

        rospy.on_shutdown(self.on_shutdown_hook)
        print("\n" + "="*60)
        print("🚀 [TN3] GIAO DIỆN ĐO ĐẠC ĐÃ SẴN SÀNG!")
        print("Mời bạn cầm thước dây, ra mốc 2m/3m/4m/5m, ngồi sau bàn và giơ tay.")
        print("Hệ thống sẽ liên tục tính toán khoảng cách mà không điều khiển xe di chuyển.")
        print("="*60 + "\n")

    def on_shutdown_hook(self):
        self.video_thread.stop()

    def update_camera_image(self, cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qImg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self.camera_label.setPixmap(QPixmap.fromImage(qImg).scaled(self.camera_label.size(), Qt.KeepAspectRatio))

    def map_callback(self, msg): 
        self.map_signal.emit(msg)

    def pose_callback(self, msg):
        q = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        yaw = tf.transformations.euler_from_quaternion(q)[2]
        self.pose_signal.emit(msg.position.x, msg.position.y, yaw)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            if hasattr(self, 'video_thread'):
                self.video_thread.locked_target_id = None
                self.video_thread.robot_state = "IDLE"
                self.video_thread.open5_confirm_count.clear()
                self.video_thread.hand_raise_start.clear()
                self.video_thread.target_history.clear()
                if hasattr(self.video_thread, 'target_lost_time'):
                    del self.video_thread.target_lost_time
                print("\n" + "*"*60)
                print("🔄 [RESET] Đã làm mới! Bạn có thể giơ tay để đo lần tiếp theo...")
                print("*"*60 + "\n")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
