import sys

content = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

# ==============================================================================
# HACK HỖ TRỢ GPU RTX 5060 (Blackwell - sm_120) TRÊN ROS NOETIC (PYTHON 3.8)
# ==============================================================================
if os.path.exists('/opt/ai_venv/bin/python') and sys.executable != '/opt/ai_venv/bin/python':
    print("🚀 Auto-switched to Python 3.9 venv to unlock NVIDIA RTX 5060 (sm_120) GPU...")
    sys.stdout.flush()
    os.execv('/opt/ai_venv/bin/python', ['/opt/ai_venv/bin/python'] + sys.argv)

os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
os.environ['QT_API'] = 'pyqt5'
os.environ['YOLO_OFFLINE'] = 'True'

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QHBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
import time
import math
import pyrealsense2 as rs
import requests
import queue
import threading
import json

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
from std_msgs.msg import String
import actionlib
from actionlib_msgs.msg import GoalStatus

import navigationcacdiem as nav
import mir_tts

try:
    from ultralytics import YOLO
except ImportError:
    print("Vui lòng cài ultralytics")
    sys.exit()

try:
    import mediapipe as mp
except ImportError:
    print("Vui lòng cài mediapipe")
    sys.exit()

MIR_IP = "192.168.0.177"
MIR_API_URL = f"http://{MIR_IP}/api/v2.0.0"
MIR_AUTH = "Basic YWRtaW46OGM2OTc2ZTViNTQxMDQxNWJkZTkwOGJkNGRlZTE1ZGZiMTY3YTljODczZmM0YmI4YTgxZjZmMmFiNDQ4YTkxOA=="

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
                if 0.3 < d < 6.0: distances.append(d)
    if not distances: return -1.0
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

# ==============================================================================
class MapLabel(QLabel):
    clicked_signal = pyqtSignal(float, float, float)
    def __init__(self):
        super().__init__()
        self.map_img = None
        self.map_info = None
        self.goal_px = None
        self.goal_yaw = 0.0
        self.auto_target_px = None
        self.table_box_px = None
        self.path_px = []
        self.robot_px = None
        self.robot_yaw = 0.0
        self.map_data = None

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
        img[data == -1] = [127, 127, 127]
        img[data == 0] = [255, 255, 255]
        img[data == 100] = [0, 0, 0]
        self.map_img = cv2.flip(img, 0)
        self.map_data = data
        self.update_view()

    def set_path(self, path_msg):
        if not self.map_info: return
        self.path_px = []
        res, ox, oy, h = self.map_info.resolution, self.map_info.origin.position.x, self.map_info.origin.position.y, self.map_info.height
        for pose in path_msg.poses:
            wx, wy = pose.pose.position.x, pose.pose.position.y
            px = int((wx - ox) / res)
            py = h - int((wy - oy) / res) - 1
            if 0 <= px < self.map_info.width and 0 <= py < h:
                self.path_px.append((px, py))
        self.update_view()

    def update_view(self):
        if self.map_img is None: return
        display_img = self.map_img.copy()
        if len(self.path_px) > 1:
            for i in range(len(self.path_px)-1):
                cv2.line(display_img, self.path_px[i], self.path_px[i+1], (255, 0, 0), 2)
        if self.goal_px:
            cv2.circle(display_img, self.goal_px, 6, (0, 255, 0), -1)
        if getattr(self, 'table_box_px', None):
            pts = np.array(self.table_box_px, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(display_img, [pts], True, (0, 0, 255), 2)
        if getattr(self, 'auto_target_px', None):
            cv2.drawMarker(display_img, self.auto_target_px, (0, 165, 255), markerType=cv2.MARKER_STAR, markerSize=20, thickness=2)
            cv2.circle(display_img, self.auto_target_px, 15, (0, 0, 255), 2)

        if self.robot_px and self.map_info:
            res = self.map_info.resolution
            rl, rw = (0.89 / res) / 2, (0.58 / res) / 2
            pts = []
            for dx, dy in [(-rl, -rw), (rl, -rw), (rl, rw), (-rl, rw)]:
                rx = dx * math.cos(-self.robot_yaw) - dy * math.sin(-self.robot_yaw)
                ry = dx * math.sin(-self.robot_yaw) + dy * math.cos(-self.robot_yaw)
                pts.append([int(self.robot_px[0] + rx), int(self.robot_px[1] + ry)])
            pts = np.array(pts, np.int32).reshape((-1, 1, 2))
            overlay = display_img.copy()
            cv2.fillPoly(overlay, [pts], (0, 165, 255))
            cv2.addWeighted(overlay, 0.6, display_img, 0.4, 0, display_img)
            cv2.polylines(display_img, [pts], True, (0, 0, 0), 2)

        h, w, ch = display_img.shape
        qImg = QImage(display_img.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self.setPixmap(QPixmap.fromImage(qImg))

# ==============================================================================
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    target_locked_signal = pyqtSignal(float, float, int)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.is_scanning_for_hand = False 
        
        print("[INFO] Đang tải mô hình cảnh báo người YOLO11n...")
        self.model = YOLO("/home/tuanminh/mir_project/yolo11n.pt") 
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

        self.locked_track_id = None
        self.target_candidate_id = None
        self.open_hand_start_time = 0

    def is_hand_open(self, hand_landmarks):
        open_fingers = sum(1 for tip, pip in [(8,6), (12,10), (16,14), (20,18)] if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y)
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x: open_fingers += 1
        return open_fingers >= 4

    def run(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        try:
            self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)
            depth_profile = self.pipeline.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile()
            self.depth_intrinsics = depth_profile.get_intrinsics()
            print("[INFO] Đã KẾT NỐI RealSense!")
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
            results = self.model.track(frame, classes=[0], conf=0.45, iou=0.6, persist=True, tracker="bytetrack.yaml", verbose=False)
            annotated_frame = frame.copy()
            
            # Chỉ xử lý nhận diện tay khi bật cờ
            if self.is_scanning_for_hand:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hand_results = self.hands.process(rgb_frame)
                current_time = time.time()
                
                people = []
                if results[0].boxes and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    for box, tid in zip(boxes, track_ids): people.append({"id": tid, "box": box})

                hand_owner_id = None
                hand_state = None
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        wrist = hand_landmarks.landmark[0]
                        h, w, _ = frame.shape
                        hx, hy = int(wrist.x * w), int(wrist.y * h)
                        if self.is_hand_open(hand_landmarks): hand_state = "open"
                        
                        for person in people:
                            x1, y1, x2, y2 = person["box"]
                            if (x1 - 30) <= hx <= (x2 + 30) and hy < y1 + (y2 - y1) * 0.25:
                                hand_owner_id = person["id"]
                                break
                        if hand_owner_id is not None: break

                if hand_owner_id is not None and self.locked_track_id is None:
                    if hand_state == "open":
                        if self.target_candidate_id != hand_owner_id:
                            self.target_candidate_id = hand_owner_id
                            self.open_hand_start_time = current_time
                        elif current_time - self.open_hand_start_time >= 2.0: # Giảm thời gian khóa xuống 2s cho nhanh
                            self.locked_track_id = hand_owner_id
                            
                for person in people:
                    tid, box = person["id"], person["box"]
                    x1, y1, x2, y2 = map(int, box)
                    if self.locked_track_id == tid:
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                        d_m_raw = get_depth_distance_m(depth_frame, box, 640, 480)
                        if d_m_raw > 0:
                            d_ngang_m = math.sqrt(d_m_raw**2 - 1.2**2) if d_m_raw > 1.2 else d_m_raw
                            rel = get_person_relative_position_m(box, 640, self.depth_intrinsics, d_ngang_m)
                            if rel is not None:
                                msg = PointStamped()
                                msg.header.frame_id = "base_link"
                                msg.point.x, msg.point.y = rel[0] - 0.475, rel[1]
                                try:
                                    self.tf_listener.waitForTransform("/map", "base_link", rospy.Time(0), rospy.Duration(0.05))
                                    pt = self.tf_listener.transformPoint("/map", msg)
                                    self.target_locked_signal.emit(pt.point.x, pt.point.y, tid)
                                    self.locked_track_id = None # Tắt khóa ngay sau khi emit
                                    self.target_candidate_id = None
                                except: pass

            cv2.putText(annotated_frame, "SCANNING" if self.is_scanning_for_hand else "IDLE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            self.change_pixmap_signal.emit(annotated_frame)
            
        self.pipeline.stop()

    def stop(self):
        self._run_flag = False
        self.wait()

# ==============================================================================
class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MiR Auto Navigation - V3 STATE MACHINE")
        self.resize(1280, 600)
        
        self.central_widget = QWidget()
        self.layout = QHBoxLayout(self.central_widget)
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.camera_label, 1)
        self.map_label = MapLabel()
        self.map_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.map_label, 1)
        self.setCentralWidget(self.central_widget)

        rospy.init_node('main_control_v3', anonymous=True, disable_signals=True)
        
        # Load Laptop YOLO Model
        rospy.loginfo("Đang tải YOLO Laptop (Đồ uống)...")
        model_path = '/home/tuanminh/mir_project/src/mir_robot/tm/best/best.pt'
        if os.path.exists(model_path):
            self.laptop_yolo = YOLO(model_path)
        else:
            self.laptop_yolo = None
            rospy.logwarn("❌ Lỗi: Không tìm thấy model laptop.")

        self.task_queue = queue.Queue()
        self.active_orders = {} 
        self.saved_locations = {} # LƯU TỌA ĐỘ VẪY TAY { "ban 1": (x, y) }
        self.current_location = "sac"
        
        self.action_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        
        # Threads & Events
        self.video_thread = VideoThread()
        self.video_thread.change_pixmap_signal.connect(self.update_camera_image)
        self.video_thread.target_locked_signal.connect(self.on_hand_locked)
        self.video_thread.start()

        self.wait_event = threading.Event()
        self.scanning_event = threading.Event()
        self.target_locked_coords = None

        # Publishers / Subscribers
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        rospy.Subscriber('/robot_pose', Pose, self.pose_callback)
        rospy.Subscriber('/table_call_buttons', String, self.on_guest_call)
        rospy.Subscriber('/robot_orders', String, self.on_web_order)
        rospy.Subscriber('/kitchen_commands', String, self.on_kitchen_cmd)
        
        self.pub_arrived = rospy.Publisher('/robot_arrived_table', String, queue_size=10)

        # Mở phanh MiR
        headers = {"Content-Type": "application/json", "Authorization": MIR_AUTH}
        try:
            requests.delete(f"{MIR_API_URL}/mission_queue", headers=headers, timeout=2)
            requests.put(f"{MIR_API_URL}/status", headers=headers, json={"state_id": 3}, timeout=2)
        except: pass

        self.worker_thread = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker_thread.start()

    def update_camera_image(self, cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qImg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self.camera_label.setPixmap(QPixmap.fromImage(qImg).scaled(self.camera_label.size(), Qt.KeepAspectRatio))

    def map_callback(self, msg):
        self.map_label.set_map(msg)

    def pose_callback(self, msg):
        q = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        yaw = tf.transformations.euler_from_quaternion(q)[2]
        self.map_label.set_robot_pose(msg.position.x, msg.position.y, yaw)

    def on_guest_call(self, msg):
        try:
            data = json.loads(msg.data)
            ban = str(data.get("ban")).strip()
            if ban.isdigit(): ban = f"ban {ban}"
            self.task_queue.put({"type": "GUEST_CALL", "target": ban})
        except: pass

    def on_web_order(self, msg):
        try:
            data = json.loads(msg.data)
            ban = str(data.get("ban", "")).strip()
            if ban.isdigit(): ban = f"ban {ban}"
            self.active_orders[ban] = {"coca": int(data.get("coca", 0)), "lavie": int(data.get("lavie", 0))}
            self.wait_event.set() # Ngắt chờ order
        except: pass

    def on_kitchen_cmd(self, msg):
        try:
            data = json.loads(msg.data)
            if data.get("action") == "deliver":
                ban = str(data.get("table", "")).strip()
                if ban.isdigit(): ban = f"ban {ban}"
                self.task_queue.put({"type": "DELIVER", "target": ban})
        except: pass

    def on_hand_locked(self, mx, my, tid):
        self.target_locked_coords = (mx, my)
        self.scanning_event.set()

    # ================= GEOMETRY LOGIC =================
    def calculate_geometry_safe_goal(self, target_x, target_y, original_yaw=0.0):
        if not self.map_label.map_info or self.map_label.robot_px is None:
            return target_x, target_y, original_yaw
        res, ox, oy, w, h = self.map_label.map_info.resolution, self.map_label.map_info.origin.position.x, self.map_label.map_info.origin.position.y, self.map_label.map_info.width, self.map_label.map_info.height
        px_t, py_t = int((target_x - ox)/res), int((target_y - oy)/res)
        
        obs_mask = np.where((self.map_label.map_data != 0) & (self.map_label.map_data != -1), 255, 0).astype(np.uint8)
        win_px = int(6.0 / res)
        x1, x2 = max(0, px_t - win_px//2), min(w, px_t + win_px//2)
        y1, y2 = max(0, py_t - win_px//2), min(h, py_t + win_px//2)
        
        local_mask = obs_mask[y1:y2, x1:x2].copy()
        contours, _ = cv2.findContours(local_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        global_contours = [c + np.array([[[x1, y1]]]) for c in contours]
        
        best_contour, min_dist = None, float('inf')
        for cnt in global_contours:
            if cv2.contourArea(cnt) < 2 and len(cnt) < 5: continue
            dist = cv2.pointPolygonTest(cnt, (px_t, py_t), True)
            if dist >= 0: best_contour = cnt; break
            if abs(dist) < min_dist: min_dist = abs(dist); best_contour = cnt
            
        if best_contour is None: return target_x, target_y, original_yaw
            
        rect = cv2.minAreaRect(best_contour)
        box = np.array(cv2.boxPoints(rect), dtype=np.int32)
        edges = [{'p1': box[i], 'p2': box[(i+1)%4], 'len': math.hypot(box[(i+1)%4][0]-box[i][0], box[(i+1)%4][1]-box[i][1]), 'center': ((box[i][0]+box[(i+1)%4][0])/2, (box[i][1]+box[(i+1)%4][1])/2)} for i in range(4)]
        edges.sort(key=lambda e: e['len'], reverse=True)
        long_edges, short_edges = edges[0:2], edges[2:4]
        
        best_long = min(long_edges, key=lambda e: math.hypot(e['center'][0]-px_t, e['center'][1]-py_t))
        p1, p2 = np.array(best_long['p1'], float), np.array(best_long['p2'], float)
        vec_edge = p2 - p1
        vec_unit = vec_edge / best_long['len'] if best_long['len']>0 else np.array([1,0])
        proj_length = np.dot(np.array([px_t - p1[0], py_t - p1[1]]), vec_unit)
        t = max(0.0, min(1.0, proj_length / best_long['len'] if best_long['len']>0 else 0.5))
        proj_pt = p1 + vec_unit * proj_length
        
        rect_center = np.array(rect[0])
        normal_long = np.array([-vec_unit[1], vec_unit[0]])
        if np.dot(np.array(best_long['center']) - rect_center, normal_long) < 0: normal_long = -normal_long
        
        if t < 0.2 or t > 0.8:
            best_short = min(short_edges, key=lambda e: math.hypot(e['center'][0]-proj_pt[0], e['center'][1]-proj_pt[1]))
            vec_se = np.array(best_short['p2'], float) - np.array(best_short['p1'], float)
            vec_se_unit = vec_se / best_short['len'] if best_short['len']>0 else np.array([1,0])
            normal_short = np.array([-vec_se_unit[1], vec_se_unit[0]])
            if np.dot(np.array(best_short['center']) - rect_center, normal_short) < 0: normal_short = -normal_short
            
            goal_x = best_short['center'][0] + normal_short[0] * int(0.7/res)
            goal_y = best_short['center'][1] + normal_short[1] * int(0.7/res)
            yaw = math.atan2(py_t - goal_y, px_t - goal_x)
        else:
            offset_px, lat_px = int(0.7/res), int(0.3/res)
            g1, g2 = proj_pt + normal_long*offset_px - vec_unit*lat_px, proj_pt + normal_long*offset_px + vec_unit*lat_px
            pref, fall = (g1, g2) if t < 0.5 else (g2, g1)
            
            safe_mask = cv2.erode((self.map_label.map_data == 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.4/res)*2+1, int(0.4/res)*2+1)))
            def is_safe(g):
                gx, gy = int(g[0]), int(g[1])
                return safe_mask[gy, gx] == 1 if 0<=gx<w and 0<=gy<h else False
                
            chosen = pref if is_safe(pref) else (fall if is_safe(fall) else proj_pt + normal_long*offset_px)
            goal_x, goal_y = chosen[0], chosen[1]
            yaw = math.atan2(py_t - goal_y, px_t - goal_x)
            
        self.map_label.table_box_px = [(int(b[0]), h-int(b[1])-1) for b in box]
        self.map_label.update_view()
        return ox + goal_x*res, oy + goal_y*res, yaw

    # ================= WORKER STATE MACHINE =================
    def worker_loop(self):
        while not rospy.is_shutdown():
            try:
                task = self.task_queue.get(timeout=2.0)
                self.execute_task(task)
                self.task_queue.task_done()
            except queue.Empty:
                if self.current_location not in ["sac", "moving_to_sac", "bep"]:
                    rospy.loginfo("Rảnh rỗi -> Về sạc")
                    self.current_location = "moving_to_sac"
                    self.task_queue.put({"type": "RETURN_HOME", "target": "sac"})

    def move_to_pose(self, x, y, yaw):
        self.action_client.wait_for_server()
        g = MoveBaseGoal()
        g.target_pose.header.frame_id = "map"
        g.target_pose.pose.position.x = x
        g.target_pose.pose.position.y = y
        q = tf.transformations.quaternion_from_euler(0, 0, yaw)
        g.target_pose.pose.orientation.z = q[2]
        g.target_pose.pose.orientation.w = q[3]
        self.action_client.send_goal(g)
        self.action_client.wait_for_result()

    def move_to_static_goal(self, target_name):
        if target_name not in nav.DIEM: return
        diem = nav.DIEM[target_name]
        self.action_client.wait_for_server()
        g = MoveBaseGoal()
        g.target_pose.header.frame_id = "map"
        g.target_pose.pose.position.x = diem["x"]
        g.target_pose.pose.position.y = diem["y"]
        g.target_pose.pose.orientation.z = diem["qz"]
        g.target_pose.pose.orientation.w = diem["qw"]
        self.action_client.send_goal(g)
        self.action_client.wait_for_result()
        self.current_location = target_name

    def verify_tray(self, exp_coca, exp_lavie, check_empty=False):
        if not self.laptop_yolo: return True
        cap = cv2.VideoCapture(0)
        start = rospy.Time.now()
        success = 0
        while (rospy.Time.now() - start).to_sec() < 30.0:
            ret, frame = cap.read()
            if not ret: break
            res = self.laptop_yolo.track(frame, persist=True, stream=True, conf=0.40, verbose=False)
            coca, lavie = 0, 0
            for r in res:
                if r.boxes:
                    for b in r.boxes:
                        if int(b.cls[0]) == 0: coca += 1
                        else: lavie += 1
            
            if check_empty:
                if coca == 0 and lavie == 0: success += 1
                else: success = 0
            else:
                ec, el = max(0, exp_coca), max(0, exp_lavie)
                if ec==0 and el==0: el=1
                if coca >= ec and lavie >= el: success += 1
                else: success = 0
                
            if success >= 5:
                cap.release()
                return True
        cap.release()
        return False

    def execute_task(self, task):
        ttype, target = task["type"], task["target"]
        
        if ttype == "GUEST_CALL":
            self.move_to_static_goal(target)
            mir_tts.speak_on_mir("Chào quý khách, khách nào order thì giơ tay lên.")
            
            self.target_locked_coords = None
            self.scanning_event.clear()
            self.video_thread.is_scanning_for_hand = True
            
            if self.scanning_event.wait(timeout=20.0):
                self.video_thread.is_scanning_for_hand = False
                tx, ty = self.target_locked_coords
                self.saved_locations[target] = (tx, ty)
                
                safe_x, safe_y, safe_yaw = self.calculate_geometry_safe_goal(tx, ty, 0.0)
                self.move_to_pose(safe_x, safe_y, safe_yaw)
                self.current_location = "specific_" + target
                
                mir_tts.speak_on_mir("Mời khách order.")
                self.pub_arrived.publish(json.dumps({"action": "popup_menu", "ban": target}))
                
                self.wait_event.clear()
                self.wait_event.wait(timeout=45.0)
                mir_tts.speak_on_mir("Vui lòng đợi món.")
            else:
                self.video_thread.is_scanning_for_hand = False
                mir_tts.speak_on_mir("Chưa thấy khách hàng giơ tay.")
                
        elif ttype == "DELIVER":
            if self.current_location != "bep":
                self.move_to_static_goal("bep")
            mir_tts.speak_on_mir("Đã tới bếp, yêu cầu để đồ ăn lên.")
            
            order = self.active_orders.get(target, {"coca":0, "lavie":0})
            if not self.verify_tray(order["coca"], order["lavie"], check_empty=False):
                mir_tts.speak_on_mir("Đồ ăn chưa đủ, xin thử lại sau.")
                return
                
            mir_tts.speak_on_mir(f"Đang giao món tới {target}.")
            
            if target in self.saved_locations:
                tx, ty = self.saved_locations[target]
                safe_x, safe_y, safe_yaw = self.calculate_geometry_safe_goal(tx, ty, 0.0)
                self.move_to_pose(safe_x, safe_y, safe_yaw)
                self.current_location = "specific_" + target
            else:
                self.move_to_static_goal(target)
                
            mir_tts.speak_on_mir("Đồ ăn của quý khách đã tới, yêu cầu quý khách lấy đồ ăn.")
            
            if self.verify_tray(0, 0, check_empty=True):
                mir_tts.speak_on_mir("Cảm ơn quý khách, chúc quý khách bữa ăn ngon miệng.")
            else:
                mir_tts.speak_on_mir("Khách chưa lấy hết đồ, robot xin phép rời đi.")
                
            if target in self.saved_locations: del self.saved_locations[target]
            if target in self.active_orders: del self.active_orders[target]

        elif ttype == "RETURN_HOME":
            self.move_to_static_goal(target)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
"""

with open("/home/tuanminh/mir_project/src/mir_robot/tm/mainv3.py", "w") as f:
    f.write(content)
