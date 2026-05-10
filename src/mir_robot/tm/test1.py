#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

# ==============================================================================
# HACK HỖ TRỢ GPU RTX 5060 (Blackwell - sm_120) TRÊN ROS NOETIC (PYTHON 3.8)
# ==============================================================================
if os.path.exists('/opt/ai_venv/bin/python') and sys.executable != '/opt/ai_venv/bin/python':
    print("🚀 Auto-switched to Python 3.9 venv to unlock NVIDIA RTX 5060 (sm_120) GPU cho YOLO...")
    sys.stdout.flush()
    os.execv('/opt/ai_venv/bin/python', ['/opt/ai_venv/bin/python'] + sys.argv)

# ==============================================================================
# SỬA XUNG ĐỘT QT, LOẠI BỎ LỖI CV2.IMSHOW HEADLESS BẰNG CÁCH DÙNG PYQT5
# ==============================================================================
os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
os.environ['QT_API'] = 'pyqt5'

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
import time
import math
import pyrealsense2 as rs
import requests

# --- ROS FIX FOR VENV ---
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
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PointStamped, PoseStamped, Pose, PoseWithCovarianceStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

MIR_IP = "192.168.0.177"
MIR_API_URL = f"http://{MIR_IP}/api/v2.0.0"
MIR_AUTH = "Basic YWRtaW46OGM2OTc2ZTViNTQxMDQxNWJkZTkwOGJkNGRlZTE1ZGZiMTY3YTljODczZmM0YmI4YTgxZjZmMmFiNDQ4YTkxOA=="

# ==============================================================================
# HÀM XỬ LÝ TOÁN HỌC VÀ CAMERA DEPTH
# ==============================================================================
def get_depth_distance_m(depth_frame, box, frame_w, frame_h):
    x1, y1, x2, y2 = map(int, box)
    width = x2 - x1
    height = y2 - y1
    roi_x1 = int(x1 + width * 0.15)
    roi_x2 = int(x2 - width * 0.15)
    roi_y1 = int(y1 + height * 0.10)
    roi_y2 = int(y1 + height * 0.60)
    
    distances = []
    step_x = max(1, (roi_x2 - roi_x1) // 10)
    step_y = max(1, (roi_y2 - roi_y1) // 10)
    
    for px in range(roi_x1, roi_x2 + 1, step_x):
        for py in range(roi_y1, roi_y2 + 1, step_y):
            if 0 <= px < frame_w and 0 <= py < frame_h:
                d = depth_frame.get_distance(px, py)
                if 0.3 < d < 6.0: 
                    distances.append(d)
                
    if not distances:
        return -1.0
    return float(np.percentile(distances, 30))

def get_person_relative_position_m(box, frame_w, depth_intrinsics, distance_m):
    x1, y1, x2, y2 = map(int, box)
    center_x = (x1 + x2) // 2
    if distance_m <= 0: return None
    if depth_intrinsics is None:
        hfov_rad = math.radians(69.0)
        angle = ((center_x - frame_w / 2.0) / frame_w) * hfov_rad
        x_cam = distance_m * math.tan(angle)
    else:
        x_cam = (center_x - depth_intrinsics.ppx) / depth_intrinsics.fx * distance_m
    return (distance_m, -x_cam)

try:
    from ultralytics import YOLO
except ImportError:
    print("Vui lòng cài đặt ultralytics: pip install ultralytics")
    sys.exit()

try:
    import mediapipe as mp
except ImportError:
    print("Vui lòng cài đặt mediapipe: pip install mediapipe")
    sys.exit()

# ==============================================================================
# CLASS 1: Bản đồ tương tác (Interactive Map Label)
# ==============================================================================
class MapLabel(QLabel):
    clicked_signal = pyqtSignal(float, float, float)

    def __init__(self):
        super().__init__()
        self.map_img = None
        self.map_info = None
        
        self.goal_px = None
        self.goal_yaw = 0.0
        self.auto_target_px = None # Điểm đích tự động từ Camera
        self.table_box_px = None # Khung viền đỏ bao quanh bàn
        
        self.path_px = []
        self.robot_px = None
        self.robot_yaw = 0.0
        
        self.drag_start_px = None
        self.drag_current_px = None
        self.is_dragging = False

    def set_robot_pose(self, wx, wy, yaw=0.0):
        if not self.map_info:
            return
        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        h = self.map_info.height
        
        px = int((wx - ox) / res)
        py = h - int((wy - oy) / res) - 1
        
        if 0 <= px < self.map_info.width and 0 <= py < h:
            self.robot_px = (px, py)
            self.robot_yaw = yaw
            self.update_view()

    def set_auto_target(self, wx, wy):
        if not self.map_info:
            return
        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        h = self.map_info.height
        
        px = int((wx - ox) / res)
        py = h - int((wy - oy) / res) - 1
        
        if 0 <= px < self.map_info.width and 0 <= py < h:
            self.auto_target_px = (px, py)
            self.update_view()

    def set_map(self, occ_grid):
        self.map_info = occ_grid.info
        w = self.map_info.width
        h = self.map_info.height
        data = np.array(occ_grid.data, dtype=np.int8).reshape((h, w))
        
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[data == -1] = [127, 127, 127]
        img[data == 0] = [255, 255, 255]
        img[data == 100] = [0, 0, 0]
        
        self.map_img = cv2.flip(img, 0)
        self.map_data = data
        self.update_view()

    def set_path(self, path_msg):
        if not self.map_info:
            return
        self.path_px = []
        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        h = self.map_info.height
        
        for pose in path_msg.poses:
            wx = pose.pose.position.x
            wy = pose.pose.position.y
            px = int((wx - ox) / res)
            py = h - int((wy - oy) / res) - 1
            if 0 <= px < self.map_info.width and 0 <= py < h:
                self.path_px.append((px, py))
        self.update_view()

    def update_view(self):
        if self.map_img is None:
            return
        display_img = self.map_img.copy()
        
        # Vẽ đường (Đỏ RGB)
        if len(self.path_px) > 1:
            for i in range(len(self.path_px)-1):
                cv2.line(display_img, self.path_px[i], self.path_px[i+1], (255, 0, 0), 2)
                
        # Vẽ đích Click thủ công (Xanh lá)
        if self.goal_px:
            cv2.circle(display_img, self.goal_px, 6, (0, 255, 0), -1)
            gx, gy = self.goal_px
            ar_len = 30
            end_x = int(gx + ar_len * math.cos(-self.goal_yaw))
            end_y = int(gy + ar_len * math.sin(-self.goal_yaw))
            cv2.arrowedLine(display_img, (gx, gy), (end_x, end_y), (0, 255, 0), 2, tipLength=0.3)
            
        # Vẽ viền đỏ bao quanh bàn (Geometry Detection)
        if getattr(self, 'table_box_px', None):
            pts = np.array(self.table_box_px, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(display_img, [pts], True, (0, 0, 255), 2)
            # Không vẽ target gốc màu đỏ nữa vì đã có viền đỏ


        # Vẽ đích Tự động từ YOLO (Cam chói)
        if self.auto_target_px:
            cv2.drawMarker(display_img, self.auto_target_px, (0, 165, 255), markerType=cv2.MARKER_STAR, markerSize=20, thickness=2)
            # Thêm viền đỏ báo hiệu đang nhận diện
            cv2.circle(display_img, self.auto_target_px, 15, (0, 0, 255), 2)

        if self.is_dragging and self.drag_start_px and self.drag_current_px:
            dist = math.hypot(self.drag_current_px[0] - self.drag_start_px[0], self.drag_current_px[1] - self.drag_start_px[1])
            if dist > 5:
                cv2.arrowedLine(display_img, self.drag_start_px, self.drag_current_px, (255, 0, 255), 2, tipLength=0.3)

        if self.robot_px and self.map_info:
            res = self.map_info.resolution
            rob_len_px = (0.89 / res) / 2
            rob_wid_px = (0.58 / res) / 2
            pts = []
            for dx, dy in [(-rob_len_px, -rob_wid_px), (rob_len_px, -rob_wid_px), 
                           (rob_len_px, rob_wid_px), (-rob_len_px, rob_wid_px)]:
                rx = dx * math.cos(-self.robot_yaw) - dy * math.sin(-self.robot_yaw)
                ry = dx * math.sin(-self.robot_yaw) + dy * math.cos(-self.robot_yaw)
                pts.append([int(self.robot_px[0] + rx), int(self.robot_px[1] + ry)])

            pts = np.array(pts, np.int32).reshape((-1, 1, 2))
            overlay = display_img.copy()
            cv2.fillPoly(overlay, [pts], (0, 165, 255))
            cv2.addWeighted(overlay, 0.6, display_img, 0.4, 0, display_img)
            cv2.polylines(display_img, [pts], True, (0, 0, 0), 2)
            
            end_x = int(self.robot_px[0] + rob_len_px * 1.5 * math.cos(-self.robot_yaw))
            end_y = int(self.robot_px[1] + rob_len_px * 1.5 * math.sin(-self.robot_yaw))
            cv2.arrowedLine(display_img, self.robot_px, (end_x, end_y), (0, 0, 255), 2, tipLength=0.3)

        self.display_img = display_img
        h, w, ch = self.display_img.shape
        bytesPerLine = ch * w
        qImg = QImage(self.display_img.data, w, h, bytesPerLine, QImage.Format_RGB888).copy()
        self.setPixmap(QPixmap.fromImage(qImg))

    def mousePressEvent(self, event):
        if self.map_info is None: return
        self.drag_start_px = (event.x(), event.y())
        self.drag_current_px = self.drag_start_px
        self.is_dragging = True
        self.update_view()

    def mouseMoveEvent(self, event):
        if not self.is_dragging: return
        self.drag_current_px = (event.x(), event.y())
        self.update_view()

    def mouseReleaseEvent(self, event):
        if not self.is_dragging or self.map_info is None: return
        self.is_dragging = False
        self.drag_current_px = (event.x(), event.y())
        
        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        h = self.map_info.height
        
        spx, spy = self.drag_start_px
        wx = ox + spx * res
        wy = oy + (h - spy - 1) * res
        
        epx, epy = self.drag_current_px
        ewx = ox + epx * res
        ewy = oy + (h - epy - 1) * res
        
        yaw = math.atan2(ewy - wy, ewx - wx)
        if math.hypot(ewx - wx, ewy - wy) < 0.1:
            yaw = 0.0
            
        self.goal_px = self.drag_start_px
        self.goal_yaw = yaw
        self.auto_target_px = None # Reset auto target visual
        
        self.clicked_signal.emit(wx, wy, yaw)
        self.update_view()

# ==============================================================================
# CLASS 2: Camera YOLO Tracking Thread (Luồng phụ xử lý AI)
# ==============================================================================
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    # Signal: map_x, map_y, track_id
    target_locked_signal = pyqtSignal(float, float, int)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        
        print("[INFO] Đang tải mô hình cảnh báo người YOLO11n (Siêu Tốc Độ)...")
        self.model = YOLO("/home/tuanminh/mir_project/yolo11n.pt") 

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.7, min_tracking_confidence=0.5
        )

        self.locked_track_id = None
        self.lost_target_start_time = 0 
        self.dist_history = []
        self.target_candidate_id = None
        self.open_hand_start_time = 0
        self.fist_start_time = 0
        self.lock_target_time = 0
        self.has_picked_coord = False

    def is_hand_open(self, hand_landmarks):
        open_fingers = 0
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                open_fingers += 1
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
            open_fingers += 1
        return open_fingers >= 4

    def is_hand_fist(self, hand_landmarks):
        closed_fingers = 0
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y:
                closed_fingers += 1
        return closed_fingers >= 4

    def run(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        try:
            self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)
            profile = self.pipeline.get_active_profile()
            depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            self.depth_intrinsics = depth_profile.get_intrinsics()
            print("[INFO] Đã KẾT NỐI THÀNH CÔNG camera RealSense 3D!")
        except Exception as e:
            print(f"[ERROR] Không thể khởi tạo RealSense: {e}")
            return

        self.tf_listener = tf.TransformListener()

        while self._run_flag:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            except Exception:
                continue
                
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame: continue

            frame = np.asanyarray(color_frame.get_data())

            results = self.model.track(
                frame, classes=[0], conf=0.45, iou=0.6, 
                persist=True, tracker="bytetrack.yaml", verbose=False
            )

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = self.hands.process(rgb_frame)

            annotated_frame = frame.copy()
            current_time = time.time()

            people = []
            if results[0].boxes and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    people.append({"id": track_id, "box": (x1, y1, x2, y2)})

            hand_owner_id = None
            hand_state = None

            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    wrist = hand_landmarks.landmark[0]
                    h, w, _ = frame.shape
                    hx, hy = int(wrist.x * w), int(wrist.y * h)

                    if self.is_hand_open(hand_landmarks):
                        hand_state = "open"
                    elif self.is_hand_fist(hand_landmarks):
                        hand_state = "fist"

                    for person in people:
                        x1, y1, x2, y2 = person["box"]
                        if (x1 - 30) <= hx <= (x2 + 30) and hy < y1 + (y2 - y1) * 0.25:
                            hand_owner_id = person["id"]
                            break
                    if hand_owner_id is not None: break

            # Khóa mục tiêu bằng tay xoè
            if hand_owner_id is not None:
                if self.locked_track_id is None:
                    if hand_state == "open":
                        if self.target_candidate_id != hand_owner_id:
                            self.target_candidate_id = hand_owner_id
                            self.open_hand_start_time = current_time
                        else:
                            if current_time - self.open_hand_start_time >= 3.0:
                                self.locked_track_id = hand_owner_id
                                self.dist_history.clear()
                                print(f"[TARGET] ĐÃ KHÓA MỤC TIÊU ID {self.locked_track_id}")
                                self.target_candidate_id = None
                                self.lock_target_time = current_time
                                self.has_picked_coord = False
                    else:
                        self.target_candidate_id = None
                        self.open_hand_start_time = current_time

                elif self.locked_track_id == hand_owner_id:
                    if hand_state == "fist":
                        if self.fist_start_time == 0:
                            self.fist_start_time = current_time
                        else:
                            if current_time - self.fist_start_time >= 3.0:
                                print(f"[TARGET] ĐÃ HỦY KHÓA MỤC TIÊU ID {self.locked_track_id}")
                                self.locked_track_id = None
                                self.dist_history.clear()
                                self.fist_start_time = 0
                    else:
                        self.fist_start_time = 0
            else:
                self.target_candidate_id = None
                self.fist_start_time = 0
            
            if self.locked_track_id is not None and (hand_owner_id != self.locked_track_id or hand_state != "fist"):
                self.fist_start_time = 0

            is_target_in_frame = False
            if self.locked_track_id is not None:
                for p in people:
                    if p["id"] == self.locked_track_id:
                        is_target_in_frame = True
                        self.locked_bbox = p["box"]
                        break
                
                if not is_target_in_frame and hasattr(self, 'locked_bbox') and self.locked_bbox is not None and len(people) > 0:
                    lx1, ly1, lx2, ly2 = self.locked_bbox
                    lcx, lcy = (lx1 + lx2) / 2, (ly1 + ly2) / 2
                    best_match = None
                    min_dist = float('inf')
                    for p in people:
                        px1, py1, px2, py2 = p["box"]
                        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
                        dist = ((pcx - lcx)**2 + (pcy - lcy)**2)**0.5
                        if dist < 150 and dist < min_dist:
                            min_dist = dist
                            best_match = p
                    if best_match is not None:
                        self.locked_track_id = best_match["id"]
                        self.locked_bbox = best_match["box"]
                        is_target_in_frame = True
                        self.lost_target_start_time = 0

            if self.locked_track_id is not None:
                if is_target_in_frame:
                    self.lost_target_start_time = 0 
                else:
                    if self.lost_target_start_time == 0:
                        self.lost_target_start_time = current_time
                    else:
                        if current_time - self.lost_target_start_time > 1.5:
                            print(f"[TARGET] Tự động huỷ khoá mục tiêu do mất dấu! ID: {self.locked_track_id}")
                            self.locked_track_id = None
                            self.dist_history.clear()
                            self.lost_target_start_time = 0

            # Render hình ảnh và phát toạ độ
            for person in people:
                track_id = person["id"]
                x1, y1, x2, y2 = person["box"]
                
                if self.locked_track_id is not None:
                    if track_id != self.locked_track_id: continue
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                    cv2.putText(annotated_frame, f"LOCKED TARGET #{track_id}", (x1, max(20, y1-10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    d_m_raw = get_depth_distance_m(depth_frame, (x1, y1, x2, y2), 640, 480)
                    if d_m_raw > 0:
                        self.dist_history.append(d_m_raw)
                        if len(self.dist_history) > 7: self.dist_history.pop(0)
                        d_m = float(np.median(self.dist_history))
                        delta_h = 1.2
                        d_ngang_m = math.sqrt(d_m**2 - delta_h**2) if d_m > delta_h else d_m
                        
                        dist_text = f"Dist: {d_ngang_m:.2f}m"
                        cv2.putText(annotated_frame, dist_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        rel = get_person_relative_position_m((x1, y1, x2, y2), 640, self.depth_intrinsics, d_ngang_m)
                        if rel is not None:
                            forward_m, left_m = rel
                            forward_m -= 0.475 # Offset camera
                            msg = PointStamped()
                            msg.header.stamp = rospy.Time(0)
                            msg.header.frame_id = "base_link"
                            msg.point.x = forward_m
                            msg.point.y = left_m
                            msg.point.z = 0.0
                            try:
                                self.tf_listener.waitForTransform("/map", "base_link", rospy.Time(0), rospy.Duration(0.05))
                                pt = self.tf_listener.transformPoint("/map", msg)
                                map_x, map_y = pt.point.x, pt.point.y
                                map_text = f"Map: ({map_x:.2f}, {map_y:.2f})"
                                cv2.putText(annotated_frame, map_text, (x1, y2 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                
                                if not self.has_picked_coord:
                                    progress = current_time - self.lock_target_time
                                    if progress >= 3.0:
                                        self.target_locked_signal.emit(map_x, map_y, track_id)
                                        self.has_picked_coord = True
                                        print(f"[TARGET] ĐÃ CHỐT TỌA ĐỘ MỤC TIÊU TẠI ({map_x:.2f}, {map_y:.2f})!")
                                    else:
                                        text = f"PICKING COORD: {3.0 - progress:.1f}s"
                                        cv2.putText(annotated_frame, text, (x1, int(y1 + (y2-y1)/3)), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 165, 0), 2)
                                else:
                                    cv2.putText(annotated_frame, "COORD LOCKED!", (x1, int(y1 + (y2-y1)/3)), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
                                
                            except Exception as e:
                                cv2.putText(annotated_frame, "Map: N/A (TF Error)", (x1, y2 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    if self.fist_start_time > 0 and hand_owner_id == track_id and hand_state == "fist":
                        progress = current_time - self.fist_start_time
                        text = f"UNLOCKING: {progress:.1f}s"
                        cv2.putText(annotated_frame, text, (x1, int(y1 + (y2-y1)/2)), 
                                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)
                else:
                    color = (0, 0, 255)
                    thickness = 2
                    label = f"Person #{track_id}"
                    if self.target_candidate_id == track_id and hand_state == "open" and hand_owner_id == track_id:
                        progress = current_time - self.open_hand_start_time
                        color = (0, 200, 255)
                        thickness = 3
                        text = f"LOCKING: {progress:.1f}s"
                        cv2.putText(annotated_frame, text, (x1, int(y1 + (y2-y1)/2)), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 3)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                    cv2.putText(annotated_frame, label, (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if self.locked_track_id:
                if self.lost_target_start_time > 0:
                    lost_prog = current_time - self.lost_target_start_time
                    mode_text = f"TARGET LOST! AUTO UNLOCK IN: {1.5 - lost_prog:.1f}s"
                    mode_color = (0, 165, 255)
                else:
                    mode_text = f"TARGET LOCKED: ID {self.locked_track_id}"
                    mode_color = (0, 255, 0)
            else:
                mode_text = "MODE: AUTO DETECT"
                mode_color = (0, 0, 255)

            cv2.putText(annotated_frame, mode_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2)
            self.change_pixmap_signal.emit(annotated_frame)
            
        self.pipeline.stop()

    def stop(self):
        self._run_flag = False
        self.wait()

# ==============================================================================
# CLASS 3: Giao diện Kết hợp (Combined GUI App)
# ==============================================================================
class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MiR Auto Navigation & Tracking Dashboard")
        self.resize(1280, 600)
        
        # UI Layout: Trái là Camera, Phải là Bản đồ
        self.central_widget = QWidget()
        self.layout = QHBoxLayout(self.central_widget)
        
        # Trái: Camera
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.layout.addWidget(self.camera_label, 1)
        
        # Phải: Bản đồ
        self.map_label = MapLabel()
        self.map_label.setAlignment(Qt.AlignCenter)
        self.map_label.setMinimumSize(400, 400)
        self.layout.addWidget(self.map_label, 1)
        
        self.setCentralWidget(self.central_widget)

        # --- Khởi tạo ROS Map Subscriptions ---
        rospy.init_node('interactive_map_gui', anonymous=True, disable_signals=True)

        # --- Variables Cho Auto-Nav ---
        self.last_goal_time = 0
        self.last_goal_x = 0
        self.last_goal_y = 0

        # --- Khởi tạo Threads ---
        self.video_thread = VideoThread()
        self.video_thread.change_pixmap_signal.connect(self.update_camera_image)
        self.video_thread.target_locked_signal.connect(self.handle_target_locked)
        self.video_thread.start()

        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        for topic in ['/move_base_node/GlobalPlanner/plan', '/move_base/GlobalPlanner/plan',
                      '/move_base_node/SBPLLatticePlanner/plan', '/move_base_node/mir_global_planner/plan',
                      '/move_base/NavfnROS/plan', '/mir_planner/global_path']:
            rospy.Subscriber(topic, Path, self.path_callback)
            
        for topic in ['/robot_pose', '/mir_pose_simple']:
            rospy.Subscriber(topic, Pose, self.pose_callback)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.amcl_pose_callback)
        
        self.map_label.clicked_signal.connect(self.manual_send_goal)
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.ros_spin)
        self.timer.start(100)

    # --- Cập nhật Camera ---
    def update_camera_image(self, cv_img):
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qImg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        # Scale để vừa khung label
        pixmap = QPixmap.fromImage(qImg)
        self.camera_label.setPixmap(pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio))

    # --- Nhận tọa độ tự động từ AI ---
    def handle_target_locked(self, map_x, map_y, track_id):
        print(f"[AUTO-NAV] Mục tiêu {track_id} chốt tọa độ tại ({map_x:.2f}, {map_y:.2f})")
        
        rx, ry = (0,0)
        if self.map_label.robot_px and self.map_label.map_info:
            res = self.map_label.map_info.resolution
            ox = self.map_label.map_info.origin.position.x
            oy = self.map_label.map_info.origin.position.y
            h = self.map_label.map_info.height
            rx_px, ry_px = self.map_label.robot_px
            rx = ox + rx_px * res
            ry = oy + (h - ry_px - 1) * res
        yaw = math.atan2(map_y - ry, map_x - rx)
        
        self.manual_send_goal(map_x, map_y, yaw)

    # --- Điều hướng Hình Học (Table Geometry) ---
    def calculate_geometry_safe_goal(self, target_x, target_y, original_yaw=0.0):
        if not self.map_label.map_info or self.map_label.robot_px is None:
            return target_x, target_y, original_yaw
            
        res = self.map_label.map_info.resolution
        ox = self.map_label.map_info.origin.position.x
        oy = self.map_label.map_info.origin.position.y
        w = self.map_label.map_info.width
        h = self.map_label.map_info.height
        
        px_t = int((target_x - ox) / res)
        py_t = int((target_y - oy) / res)
        
        if not (0 <= px_t < w and 0 <= py_t < h):
            return target_x, target_y, original_yaw

        obs_mask = np.where((self.map_label.map_data != 0) & (self.map_label.map_data != -1), 255, 0).astype(np.uint8)
        
        win_m = 6.0
        win_px = int(win_m / res)
        half_win = win_px // 2
        
        x1 = max(0, px_t - half_win)
        x2 = min(w, px_t + half_win)
        y1 = max(0, py_t - half_win)
        y2 = min(h, py_t + half_win)
        
        local_mask = obs_mask[y1:y2, x1:x2].copy()
        
        contours, _ = cv2.findContours(local_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        global_contours = []
        for cnt in contours:
            global_cnt = cnt + np.array([[[x1, y1]]])
            global_contours.append(global_cnt)
            
        best_contour = None
        min_dist = float('inf')
        pt = (px_t, py_t)
        
        for cnt in global_contours:
            if cv2.contourArea(cnt) < 2 and len(cnt) < 5:
                continue
            dist = cv2.pointPolygonTest(cnt, pt, True)
            if dist >= 0:
                best_contour = cnt
                break
            else:
                abs_dist = abs(dist)
                if abs_dist < min_dist:
                    min_dist = abs_dist
                    best_contour = cnt
                
        if best_contour is None:
            self.map_label.table_box_px = None
            return target_x, target_y, original_yaw
            
        rect = cv2.minAreaRect(best_contour)
        box = cv2.boxPoints(rect)
        # Handle np.int0 deprecation by casting to int32 directly
        box = np.array(box, dtype=np.int32)
        
        edges = []
        for i in range(4):
            p1 = box[i]
            p2 = box[(i+1)%4]
            length = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
            center = ((p1[0]+p2[0])/2.0, (p1[1]+p2[1])/2.0)
            edges.append({'p1': p1, 'p2': p2, 'len': length, 'center': center})
            
        edges.sort(key=lambda e: e['len'], reverse=True)
        long_edges = edges[0:2]
        short_edges = edges[2:4]
        
        best_long_edge = None
        min_ed = float('inf')
        for edge in long_edges:
            d = math.hypot(edge['center'][0] - px_t, edge['center'][1] - py_t)
            if d < min_ed:
                min_ed = d
                best_long_edge = edge
                
        p1 = np.array(best_long_edge['p1'], dtype=float)
        p2 = np.array(best_long_edge['p2'], dtype=float)
        
        vec_edge = p2 - p1
        length_edge = best_long_edge['len']
        vec_edge_unit = vec_edge / length_edge if length_edge > 0 else np.array([1, 0])
        
        vec_pt = np.array([px_t - p1[0], py_t - p1[1]], dtype=float)
        proj_length = np.dot(vec_pt, vec_edge_unit)
        
        t = proj_length / length_edge if length_edge > 0 else 0.5
        t = max(0.0, min(1.0, t))
        
        proj_pt = p1 + vec_edge_unit * proj_length
        
        rect_center = np.array(rect[0])
        vec_center_to_edge = np.array(best_long_edge['center']) - rect_center
        normal_long = np.array([-vec_edge_unit[1], vec_edge_unit[0]])
        if np.dot(vec_center_to_edge, normal_long) < 0:
            normal_long = -normal_long
            
        goal_px_x, goal_px_y = None, None
        goal_yaw = 0.0
        
        if t < 0.2 or t > 0.8:
            best_short_edge = None
            min_sd = float('inf')
            for edge in short_edges:
                d = math.hypot(edge['center'][0] - proj_pt[0], edge['center'][1] - proj_pt[1])
                if d < min_sd:
                    min_sd = d
                    best_short_edge = edge
                    
            safe_dist_px = int(0.7 / res)
            
            sp1 = np.array(best_short_edge['p1'], dtype=float)
            sp2 = np.array(best_short_edge['p2'], dtype=float)
            vec_se = sp2 - sp1
            vec_se_unit = vec_se / best_short_edge['len'] if best_short_edge['len'] > 0 else np.array([1,0])
            normal_short = np.array([-vec_se_unit[1], vec_se_unit[0]])
            if np.dot(np.array(best_short_edge['center']) - rect_center, normal_short) < 0:
                normal_short = -normal_short
                
            goal_px_x = best_short_edge['center'][0] + normal_short[0] * safe_dist_px
            goal_px_y = best_short_edge['center'][1] + normal_short[1] * safe_dist_px
            
            dir_yaw = np.array([px_t - goal_px_x, py_t - goal_px_y])
            goal_yaw = math.atan2(dir_yaw[1], dir_yaw[0])
            
        else:
            offset_px = int(0.7 / res)
            lateral_offset_px = int(0.3 / res)
            goal_p2 = proj_pt + normal_long * offset_px + vec_edge_unit * lateral_offset_px
            goal_p1 = proj_pt + normal_long * offset_px - vec_edge_unit * lateral_offset_px
            
            if t < 0.5:
                preferred_goal = goal_p1
                fallback_goal = goal_p2
            else:
                preferred_goal = goal_p2
                fallback_goal = goal_p1
                
            free_space = (self.map_label.map_data == 0).astype(np.uint8)
            safe_radius_px = int(0.4 / res)
            kernel_safe = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (safe_radius_px*2+1, safe_radius_px*2+1))
            safe_mask = cv2.erode(free_space, kernel_safe)
            
            def is_safe(g):
                gx, gy = int(g[0]), int(g[1])
                if 0 <= gx < w and 0 <= gy < h:
                    return safe_mask[gy, gx] == 1
                return False
                
            if is_safe(preferred_goal):
                chosen = preferred_goal
            elif is_safe(fallback_goal):
                chosen = fallback_goal
            else:
                chosen = proj_pt + normal_long * offset_px
                
            goal_px_x = chosen[0]
            goal_px_y = chosen[1]
            dir_yaw = np.array([px_t - goal_px_x, py_t - goal_px_y])
            goal_yaw = math.atan2(dir_yaw[1], dir_yaw[0])

        safe_x = ox + goal_px_x * res
        safe_y = oy + goal_px_y * res
        
        self.map_label.table_box_px = []
        for b in box:
            img_x = b[0]
            img_y = h - b[1] - 1
            self.map_label.table_box_px.append((int(img_x), int(img_y)))
            
        return safe_x, safe_y, goal_yaw

    # --- Điều hướng thủ công / Tự động cuối ---
    def manual_send_goal(self, wx, wy, wyaw):
        print(f"[MANUAL-NAV] Yêu cầu gốc: X={wx:.2f}, Y={wy:.2f}, Yaw={wyaw:.2f}")
        
        # 1. Đi qua bộ lọc Hình học để tìm điểm đỗ 45 độ quanh bàn
        safe_x, safe_y, safe_yaw = self.calculate_geometry_safe_goal(wx, wy, wyaw)
        print(f"🎯 [GEOMETRY NAV] Chốt hạ điểm an toàn: X={safe_x:.2f}, Y={safe_y:.2f}, Yaw={safe_yaw:.2f}")
        
        # 2. Cập nhật Visual trên giao diện
        if self.map_label.map_info:
            res = self.map_label.map_info.resolution
            ox = self.map_label.map_info.origin.position.x
            oy = self.map_label.map_info.origin.position.y
            h = self.map_label.map_info.height
            px = int((safe_x - ox) / res)
            py = h - int((safe_y - oy) / res) - 1
            if 0 <= px < self.map_label.map_info.width and 0 <= py < h:
                self.map_label.goal_px = (px, py)
                self.map_label.goal_yaw = safe_yaw
                self.map_label.update_view()
                
        # Hủy marker tìm kiếm AI (dấu SAO màu cam) nếu có
        self.map_label.auto_target_px = None
        
        # 3. Gửi lệnh thực sự xuống robot
        self.send_goal_to_mir(safe_x, safe_y, safe_yaw)
        
    # --- Hàm Gửi lệnh xuống MiR ---
    def send_goal_to_mir(self, wx, wy, wyaw):
        headers = {"Content-Type": "application/json", "Authorization": MIR_AUTH}
        try:
            requests.delete(f"{MIR_API_URL}/status", headers=headers, timeout=1)
            requests.put(f"{MIR_API_URL}/status", headers=headers, json={"state_id": 3}, timeout=1)
        except Exception:
            pass
            
        try:
            st = requests.get(f"{MIR_API_URL}/status", headers=headers, timeout=2).json()
            map_id = st.get("map_id", "")
            if not map_id: return
        except Exception:
            return

        move_guid = None
        try:
            ms = requests.get(f"{MIR_API_URL}/missions", headers=headers, timeout=2).json()
            for m in ms:
                if m.get("name", "").lower() in ("move", "go to", "di chuyen", "goto"):
                    move_guid = m.get("guid")
                    break
        except Exception: pass
        if not move_guid: return

        pos_name = f"auto_nav_{int(time.time()*1000)}"
        try:
            r = requests.post(f"{MIR_API_URL}/positions", headers=headers, json={
                "name": pos_name, "pos_x": wx, "pos_y": wy,
                "orientation": math.degrees(wyaw), "type_id": 0, "map_id": map_id
            }, timeout=2)
            if r.status_code not in (200, 201): return
            pos_guid = r.json().get("guid", "")
        except Exception: return

        try:
            requests.delete(f"{MIR_API_URL}/mission_queue", headers=headers, timeout=2)
        except: pass

        try:
            r = requests.post(f"{MIR_API_URL}/mission_queue", headers=headers, json={
                "mission_id": move_guid,
                "parameters": [{"input_name": "Position", "value": pos_guid}]
            }, timeout=2)
            if r.status_code in (200, 201):
                print(f"🎯 Lệnh Move tới ({wx:.2f}, {wy:.2f}) đã đẩy thành công!")
        except Exception: pass

        goal_msg = PoseStamped()
        goal_msg.header.frame_id = "map"
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.pose.position.x = wx
        goal_msg.pose.position.y = wy
        q = tf.transformations.quaternion_from_euler(0, 0, wyaw)
        goal_msg.pose.orientation.x = q[0]
        goal_msg.pose.orientation.y = q[1]
        goal_msg.pose.orientation.z = q[2]
        goal_msg.pose.orientation.w = q[3]
        self.goal_pub.publish(goal_msg)

    # --- ROS Callbacks cho Map ---
    def map_callback(self, msg):
        self.map_label.set_map(msg)
    def path_callback(self, msg):
        self.map_label.set_path(msg)
    def pose_callback(self, msg):
        yaw = tf.transformations.euler_from_quaternion([
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])[2]
        self.map_label.set_robot_pose(msg.position.x, msg.position.y, yaw)
    def amcl_pose_callback(self, msg):
        yaw = tf.transformations.euler_from_quaternion([
            msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, 
            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]
        self.map_label.set_robot_pose(msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)
    def ros_spin(self):
        if rospy.is_shutdown(): self.close()
    def closeEvent(self, event):
        self.video_thread.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
