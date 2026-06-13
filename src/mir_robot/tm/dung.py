#!/usr/bin/env python3
import os
import sys

# ==============================================================================
# HACK HỖ TRỢ GPU RTX 5060 (Blackwell - sm_120) TRÊN ROS NOETIC (PYTHON 3.8)
# ==============================================================================
if os.path.exists('/opt/ai_venv/bin/python') and sys.executable != '/opt/ai_venv/bin/python':
    print("🚀 Auto-switched to Python 3.9 venv to unlock NVIDIA RTX 5060 (sm_120) GPU for YOLO...")
    sys.stdout.flush()
    os.execv('/opt/ai_venv/bin/python', ['/opt/ai_venv/bin/python'] + sys.argv)

# ==============================================================================
# SỬA XUNG ĐỘT QT PLUGIN GIỮA OPENCV VÀ PYQT5
# ==============================================================================
os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)  # Xóa đường dẫn giả của OpenCV
os.environ['QT_API'] = 'pyqt5'

import json
import time
import math
import threading
import queue
from collections import deque

# Đảm bảo headless mode cho OpenCV
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

import numpy as np
import hashlib
import base64
import requests

import rospy

# ==============================================================================
# CHÈN ĐƯỜNG DẪN ROS NOETIC HỆ THỐNG ĐỂ LẤY ĐÚNG MESSAGE DEFINITIONS
# rospypi (pip) có MD5 hash khác với ROS Noetic thật → subscriber bị từ chối im lặng
# Cách fix: giữ rospy từ venv (đã cached), nhưng lấy msg types từ hệ thống
# ==============================================================================
_ros_sys_path = '/opt/ros/noetic/lib/python3/dist-packages'
if os.path.isdir(_ros_sys_path) and _ros_sys_path not in sys.path:
    sys.path.insert(1, _ros_sys_path)
    # Xóa cache module message rospypi để Python load lại từ đường dẫn hệ thống
    for mod_name in list(sys.modules.keys()):
        if any(mod_name.startswith(p) for p in ['geometry_msgs', 'nav_msgs', 'sensor_msgs',
                                                  'std_msgs', 'actionlib_msgs', 'tf2_msgs', 'move_base_msgs']):
            del sys.modules[mod_name]

import tf
from geometry_msgs.msg import PointStamped, Twist, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
import actionlib_msgs.msg
from move_base_msgs.msg import MoveBaseGoal

import subprocess
import navigationcacdiem as nav
try:
    # Import cv2 VÀ các thư viện AI cùng lúc để nếu mediapipe phá cv2, ta sẽ sửa cả 2
    import cv2
    import pyrealsense2 as rs
    import mediapipe as mp
    import lap
except ImportError:
    print("❌ THIẾU THƯ VIỆN TRONG VENV CỦA PYTHON 3.9! Hệ thống đang tiến hành Cài Đặt Tự Động (Khoảng 2-3 phút)...")
    sys.stdout.flush()
    try:
        # Bước 1: Cài tất cả thư viện AI (mediapipe sẽ kéo opencv-contrib-python theo)
        subprocess.check_call(['/opt/ai_venv/bin/python', '-m', 'pip', 'install', '--no-compile',
            'pyrealsense2', 'mediapipe==0.10.14', 'matplotlib', 'PyQt5', 'websocket-client', 'lapx', 'lap'])
        # Bước 2: Gỡ bỏ opencv-contrib-python (có Qt plugin xung đột) và opencv-python
        subprocess.check_call(['/opt/ai_venv/bin/python', '-m', 'pip', 'uninstall', '-y',
            'opencv-contrib-python', 'opencv-python', 'opencv-python-headless'])
        # Bước 3: Cài lại opencv-python-headless SẠCH (không còn metadata rác)
        subprocess.check_call(['/opt/ai_venv/bin/python', '-m', 'pip', 'install', '--no-compile',
            'opencv-python-headless'])
        print("✅ Cài đặt thành công! Đang khởi động lại ứng dụng...")
        sys.stdout.flush()
        os.execv('/opt/ai_venv/bin/python', ['/opt/ai_venv/bin/python'] + sys.argv)
    except Exception as e:
        print(f"❌ Lỗi tự động cài đặt: {e}")
        sys.exit(1)

# Nếu vẫn tồn tại thư mục Qt plugin giả của OpenCV, xóa khỏi đường dẫn
_cv2_qt_dir = os.path.join(os.path.dirname(cv2.__file__), 'qt', 'plugins')
if os.path.isdir(_cv2_qt_dir):
    # Buộc phải dọn dẹp: ghi đè QT_PLUGIN_PATH về đúng PyQt5
    try:
        import PyQt5
        _pyqt5_plugins = os.path.join(os.path.dirname(PyQt5.__file__), 'Qt5', 'plugins')
        if os.path.isdir(_pyqt5_plugins):
            os.environ['QT_PLUGIN_PATH'] = _pyqt5_plugins
    except Exception:
        pass

from ultralytics import YOLO
from tf.transformations import euler_from_quaternion

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# import mir_tts

# ==============================================================================
# QUY ĐỊNH & THAM SỐ TOÀN CỤC
# ==============================================================================
os.environ['YOLO_OFFLINE'] = 'True' 
STOP_DISTANCE = 1.5 # Nâng khoảng cách an toàn lên 1.5m để tránh vi phạm Inflation Radius của CostMap

map_data = None
map_resolution = 0.05
map_origin_x = 0
map_origin_y = 0
robot_pose = None
goal_pose = None # (x, y) để vẽ điểm đén lên Map
user_pose = None # (x, y) để vẽ vị trí người dùng lên Map
map_queue = queue.Queue()
tf_listener = None
robot_planned_path = []  # Chứa các điểm x, y của quỹ đạo

class SignalBus(QObject):
    frame_update = pyqtSignal(np.ndarray)
    status_update = pyqtSignal(str)

signal_bus = SignalBus()

# ==============================================================================
# GIAO DIỆN BẢN ĐỒ (MapCanvas) & KẾT NỐI GUI
# ==============================================================================
class MapCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#e0e0e0')
        self.ax.set_facecolor('#e0e0e0')
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.current_xlim = None
        self.current_ylim = None
        self.panning = False
        self.pan_start = None

        self.mpl_connect('scroll_event', self.on_scroll)
        self.mpl_connect('button_press_event', self.on_button_press)
        self.mpl_connect('motion_notify_event', self.on_motion)
        self.mpl_connect('button_release_event', self.on_button_release)

    def draw_map(self):
        global map_data, map_resolution, map_origin_x, map_origin_y, robot_pose, goal_pose
        if map_data is None or map_resolution == 0:
            return
        try:
            self.ax.clear()
            self.ax.set_aspect('equal')
            self.ax.imshow(map_data, cmap='gray', origin='lower', extent=[0, map_data.shape[1], 0, map_data.shape[0]])

            if robot_pose:
                robot_width_px = 0.88 / map_resolution
                robot_length_px = 0.55 / map_resolution
                map_x = (robot_pose[0] - map_origin_x) / map_resolution
                map_y = (robot_pose[1] - map_origin_y) / map_resolution

                rect = plt.Rectangle((map_x - robot_width_px / 2, map_y - robot_length_px / 2),
                                     robot_width_px, robot_length_px,
                                     angle=np.degrees(robot_pose[2]), rotation_point='center', color='#3498db',
                                     alpha=0.8)
                self.ax.add_patch(rect)
                arrow_length = robot_length_px * 0.8
                dx = arrow_length * np.cos(robot_pose[2])
                dy = arrow_length * np.sin(robot_pose[2])
                self.ax.arrow(map_x, map_y, dx, dy, head_width=robot_width_px * 0.4, head_length=robot_length_px * 0.3, fc='red', ec='red')

            if goal_pose is not None:
                gmap_x = (goal_pose[0] - map_origin_x) / map_resolution
                gmap_y = (goal_pose[1] - map_origin_y) / map_resolution
                circle = plt.Circle((gmap_x, gmap_y), 0.3 / map_resolution, color='#2ecc71', fill=True, alpha=0.7)
                self.ax.add_patch(circle)

            if user_pose is not None:
                umap_x = (user_pose[0] - map_origin_x) / map_resolution
                umap_y = (user_pose[1] - map_origin_y) / map_resolution
                circle2 = plt.Circle((umap_x, umap_y), 0.15 / map_resolution, color='#e74c3c', fill=True, alpha=1.0)
                self.ax.add_patch(circle2)
            if robot_planned_path:
                path_x = [(p[0] - map_origin_x) / map_resolution for p in robot_planned_path]
                path_y = [(p[1] - map_origin_y) / map_resolution for p in robot_planned_path]
                self.ax.plot(path_x, path_y, color='#f1c40f', linewidth=3, linestyle='-')

            if self.current_xlim is None or self.current_ylim is None:
                self.current_xlim = [0, map_data.shape[1]]
                self.current_ylim = [0, map_data.shape[0]]

            self.ax.set_xlim(self.current_xlim)
            self.ax.set_ylim(self.current_ylim)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.draw()
        except Exception as e:
            pass

    def on_scroll(self, event):
        if event.inaxes != self.ax: return
        x_mouse = event.xdata; y_mouse = event.ydata
        if x_mouse is None or y_mouse is None: return
        zoom_factor = 1.2 if event.button == 'up' else 0.8
        w = self.current_xlim[1] - self.current_xlim[0]; h = self.current_ylim[1] - self.current_ylim[0]
        new_w, new_h = w / zoom_factor, h / zoom_factor
        r_x = (x_mouse - self.current_xlim[0]) / w; r_y = (y_mouse - self.current_ylim[0]) / h
        self.current_xlim = [x_mouse - r_x * new_w, x_mouse + (1 - r_x) * new_w]
        self.current_ylim = [y_mouse - r_y * new_h, y_mouse + (1 - r_y) * new_h]
        self.draw_map()

    def on_button_press(self, event):
        if event.button == 1 and event.inaxes == self.ax:
            self.panning = True
            self.pan_start = (event.xdata, event.ydata)

    def on_motion(self, event):
        if self.panning and event.inaxes == self.ax:
            dx = event.xdata - self.pan_start[0]; dy = event.ydata - self.pan_start[1]
            self.current_xlim = [self.current_xlim[0] - dx, self.current_xlim[1] - dx]
            self.current_ylim = [self.current_ylim[0] - dy, self.current_ylim[1] - dy]
            self.pan_start = (event.xdata, event.ydata)
            self.draw_map()

    def on_button_release(self, event):
        if event.button == 1: self.panning = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MiR Robot Target Tracker & Navigator")
        self.setGeometry(100, 100, 1200, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Trái: Camera
        self.left_panel = QFrame()
        self.left_panel.setMinimumWidth(660)
        self.left_panel.setStyleSheet("background-color: #333333;")
        self.main_layout.addWidget(self.left_panel)
        self.left_layout = QVBoxLayout(self.left_panel)

        self.video_label = QLabel("Khởi động Camera 3D...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: #000; color: white; font-size: 20px;")
        self.left_layout.addWidget(self.video_label)

        self.status_label = QLabel("Trạng thái: Đang khởi động AI")
        self.status_label.setStyleSheet("font-size: 16pt; color: #FFF; font-weight: bold;")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.left_layout.addWidget(self.status_label)

        # Phải: Map
        self.right_panel = QFrame()
        self.main_layout.addWidget(self.right_panel)
        self.right_layout = QVBoxLayout(self.right_panel)
        self.map_canvas = MapCanvas(self)
        self.right_layout.addWidget(self.map_canvas)

        signal_bus.frame_update.connect(self.update_camera_frame)
        signal_bus.status_update.connect(self.update_status)

        self.map_timer = QTimer()
        self.map_timer.timeout.connect(self.check_map_queue)
        self.map_timer.start(100)

        self.position_timer = QTimer()
        self.position_timer.timeout.connect(self.update_position)
        self.position_timer.start(100)

    def update_camera_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(640, 480, Qt.KeepAspectRatio))

    def update_status(self, text):
        self.status_label.setText(text)

    def check_map_queue(self):
        try:
            while not map_queue.empty():
                data, resolution, origin_x, origin_y = map_queue.get_nowait()
                global map_data, map_resolution, map_origin_x, map_origin_y
                map_data = data
                map_resolution = resolution
                map_origin_x = origin_x
                map_origin_y = origin_y
                self.map_canvas.draw_map()
        except queue.Empty: pass

    def update_position(self):
        global robot_pose, tf_listener
        if tf_listener is not None:
            try:
                (trans, rot) = tf_listener.lookupTransform('/map', '/base_link', rospy.Time(0))
                robot_pose = (trans[0], trans[1], euler_from_quaternion(rot)[2])
                self.map_canvas.draw_map()
            except: pass

    def closeEvent(self, event):
        rospy.signal_shutdown("GUI closed")
        event.accept()

# ==============================================================================
# MODULE ĐIỀU KHIỂN ROBOT (Gọi ActionLib qua Python 3.8 hệ thống)
# ==============================================================================
class MirNavigator:
    def __init__(self, ip="192.168.0.177"):
        rospy.loginfo("[MirNav] KHOI DONG KET NOI REST API (Giong test.py)...")
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.is_navigating = False
        
        self.robot = nav.ws_connect()
        self.mir_headers = nav.api_login()

    def __del__(self):
        pass

    def ensure_ready(self):
        try:
            if hasattr(self, 'mir_headers') and self.mir_headers:
                nav.api_ensure_ready(self.mir_headers)
        except Exception as e:
            rospy.logerr(f"[MirNav] Lỗi ensure_ready: {e}")

    def cancel_all(self):
        """Hủy mọi lệnh và dừng xe"""
        self.is_navigating = False
        global robot_planned_path, goal_pose
        robot_planned_path = []
        goal_pose = None
        
        try:
            if hasattr(self, 'mir_headers') and self.mir_headers:
                requests.delete(f"http://192.168.0.177/api/v2.0.0/mission_queue", headers=self.mir_headers, timeout=2)
            self.cmd_vel_pub.publish(Twist())
        except Exception as e:
            rospy.logerr(f"Lỗi cancel: {e}")

    def send_goal(self, goal_x, goal_y, goal_yaw=0.0):
        self.is_navigating = True
        rospy.loginfo(f"[MirNav REST] Gửi lệnh: {goal_x},{goal_y},{goal_yaw}")
        
        q = tf.transformations.quaternion_from_euler(0, 0, goal_yaw)
        diem_dong = {"x": goal_x, "y": goal_y, "qz": q[2], "qw": q[3], "arrive_dist": 0.15}
        
        rest_ok = False
        if hasattr(self, 'mir_headers') and self.mir_headers:
            self.ensure_ready()
            rest_ok = nav.api_navigate(self.mir_headers, diem_dong, "diem_dong")
        
        if not rest_ok and self.robot:
            nav.ws_send_goal(self.robot, diem_dong)
            
        return True

    def send_goal_cmd_vel(self, goal_x, goal_y):
        """Sử dụng cmd_vel chạy thẳng trực tiếp bypass MiR planner"""
        self.is_navigating = True
        
        def _cmd_vel_loop():
            rospy.loginfo(f"[MirNav CMD_VEL] Bắt đầu đi tới ({goal_x:.2f}, {goal_y:.2f})")
            rate = rospy.Rate(10)
            start_time = time.time()
            global robot_pose
            
            while not rospy.is_shutdown() and getattr(self, 'is_navigating', False):
                if time.time() - start_time > 60.0:
                    rospy.logwarn("[MirNav CMD_VEL] Timeout quá 60s, hủy bỏ.")
                    break
                    
                if robot_pose is None:
                    rate.sleep()
                    continue
                    
                rx, ry, ryaw = robot_pose
                dx, dy = goal_x - rx, goal_y - ry
                dist = math.hypot(dx, dy)
                
                if dist < 0.25:
                    rospy.loginfo("[MirNav CMD_VEL] 🎯 ĐÃ TỚI NƠI!")
                    self.cmd_vel_pub.publish(Twist())
                    break
                    
                target_yaw = math.atan2(dy, dx)
                angle_diff = target_yaw - ryaw
                while angle_diff > math.pi: angle_diff -= 2 * math.pi
                while angle_diff < -math.pi: angle_diff += 2 * math.pi
                
                twist = Twist()
                if abs(angle_diff) > math.radians(20):
                    twist.angular.z = max(-0.5, min(0.5, angle_diff * 1.5))
                else:
                    twist.linear.x = max(0.08, min(0.3, dist * 0.6))
                    twist.angular.z = max(-0.4, min(0.4, angle_diff * 1.0))
                    
                self.cmd_vel_pub.publish(twist)
                rate.sleep()
                
            self.cmd_vel_pub.publish(Twist())
            self.is_navigating = False

        threading.Thread(target=_cmd_vel_loop, daemon=True).start()


# ==============================================================================
# ỨNG DỤNG LÕI (Camera, YOLO, MediaPipe, Math Engine, Threading)
# ==============================================================================
def get_depth_distance_m(depth_frame, box, frame_w, frame_h):
    x1, y1, x2, y2 = map(int, box)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    roi_size = int(max(8, min(x2 - x1, y2 - y1) // 4))
    distances = []
    for dx in range(-roi_size, roi_size + 1, 8):
        for dy in range(-roi_size, roi_size + 1, 8):
            px, py = center_x + dx, center_y + dy
            orig_px = frame_w - 1 - px
            if 0 <= orig_px < frame_w and 0 <= py < frame_h:
                d = depth_frame.get_distance(orig_px, py)
                if 0.3 < d < 6.0: distances.append(d)
    return float(np.median(distances)) if distances else -1.0

def get_depth_distance_m_seg(depth_frame, poly_pts, frame_w, frame_h):
    import numpy as np
    import cv2
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(poly_pts, dtype=np.int32)], 1)
    ys, xs = np.where(mask == 1)
    distances = []
    for idx in range(0, len(xs), 10):
        px, py = xs[idx], ys[idx]
        orig_px = frame_w - 1 - px
        if 0 <= orig_px < frame_w and 0 <= py < frame_h:
            d = depth_frame.get_distance(orig_px, py)
            if 0.3 < d < 6.0: distances.append(d)
    return float(np.median(distances)) if distances else -1.0


def get_person_relative_position_m(depth_frame, center_pt, frame_w, frame_h, depth_intrinsics, distance_m):
    import math
    import pyrealsense2 as rs
    
    if len(center_pt) == 4:
        x1, y1, x2, y2 = map(int, center_pt)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
    else:
        center_x, center_y = map(int, center_pt)
        
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
    forward_m = z_opt * math.cos(pitch_rad) - y_opt * math.sin(pitch_rad)
    left_m = -x_opt
    
    down_m = z_opt * math.sin(pitch_rad) + y_opt * math.cos(pitch_rad)
    camera_height_m = 1.8
    z_m = camera_height_m - down_m
    
    return forward_m, left_m, z_m

class tracking_loop:
    def __init__(self):
        import torch
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.nav = MirNavigator()
        self.camera_ready = False
        self.depth_intrinsics = None
        self.robot_state = "IDLE"  # IDLE, LOCKED, COLLECTING, MOVING
        self.locked_target_id = None

        self.robot_state = "IDLE"  # IDLE, LOCKED, COLLECTING, MOVING
        self.locked_target_id = None
        
        self.hand_raise_start = {}
        self.open5_confirm_count = {}
        self.fist_hold_start = None
        self.fist_confirm_count = 0

        self.worker_thread = threading.Thread(target=self.run, daemon=True)
        self.worker_thread.start()

    def run(self):
        global goal_pose, user_pose, robot_planned_path
        print("⏳ Đang khởi động mô hình AI (YOLO11n & MediaPipe) trong luồng ngầm...")
        # Tải mô hình YOLO Pose và Segmentation
        self.model_pose = YOLO('yolo11n-pose.pt')
        self.model_seg = YOLO('yolo11n-seg.pt')
        if self.device == 0:
            rospy.loginfo("[AI] Phát hiện GPU RTX! Đang đưa model lên CUDA...")
            self.model_pose.to('cuda')
            self.model_seg.to('cuda')
        
        # Khởi tạo Camera RealSense
        print("⏳ Đang kết nối Camera RealSense 3D...")
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        try:
            self.pipeline.start(config)
            self.camera_ready = True
            self.align = rs.align(rs.stream.color)
            profile = self.pipeline.get_active_profile()
            depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            self.depth_intrinsics = depth_profile.get_intrinsics()
            print("✅ Đã kết nối Camera RealSense thành công!")
        except RuntimeError as e:
            rospy.logerr(f"❌ KHÔNG TÌM THẤY CAMERA REALSENSE: {e}")
            self.depth_intrinsics = None

        import mediapipe as mp
        self.hands_detector = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        print("🚀 Đã khởi động toàn bộ AI thành công! Bắt đầu quét mục tiêu...")
        
        # Thiết lập cập nhật UI nếu không có camera
        if not self.camera_ready:
            signal_bus.status_update.emit(f"Lỗi: Không kết nối Camera 3D! Vẫn chờ /map")
            print("❌ LỖI: Camera không khả dụng. Giao diện sẽ hiển thị màn hình chờ.")
        
        while not rospy.is_shutdown():
            if not self.camera_ready:
                rospy.sleep(1)
                continue
            
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame: continue

            frame = np.asanyarray(color_frame.get_data())
            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]

            # 1. YOLO Nhận diện Người (Tối ưu RTX 3050 - Tensor Core FP16)
            # Dùng Half Precision (FP16) giúp RTX 3050 chạy inference nhanh gấp đôi
            # Tracker chỉ cần trên Pose để giữ ID liên tục
            results_pose = self.model_pose.track(frame, conf=0.45, persist=True, tracker="bytetrack.yaml", verbose=False, half=(self.device==0), device=self.device)
            # Segmentation không cần tracker, chỉ cần cắt mặt nạ frame hiện tại
            results_seg = self.model_seg.predict(frame, conf=0.45, verbose=False, half=(self.device==0), device=self.device)

            # 2. Tối ưu MediaPipe: CHỈ bật khi Robot đang đứng yên (IDLE) hoặc đã áp sát mục tiêu (đến bàn)
            need_mediapipe = False
            if self.robot_state == "IDLE":
                need_mediapipe = True
                
            if self.locked_target_id is not None:
                need_mediapipe = True

            detected_hands = []
            if need_mediapipe:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hand_results = self.hands_detector.process(rgb_frame)

                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        def get_dist(i1, i2):
                            p1, p2 = hand_landmarks.landmark[i1], hand_landmarks.landmark[i2]
                            return math.hypot(p1.x - p2.x, p1.y - p2.y)
                        
                        tip_ids = [4, 8, 12, 16, 20]; pip_ids = [2, 6, 10, 14, 18]
                        tip_dists = [get_dist(t, 0) for t in tip_ids]
                        pip_dists = [get_dist(p, 0) for p in pip_ids]
                        
                        fingers = sum(1 for td, pd in zip(tip_dists, pip_dists) if td > pd)
                        all_ext = all(td >= 1.3 * pd for td, pd in zip(tip_dists, pip_dists))
                        thumb_sp = get_dist(4, 8) > 0.45 * max(1e-6, get_dist(5, 17))
                        open5_strict = (fingers == 5) and all_ext and thumb_sp

                        wrist = hand_landmarks.landmark[0]
                        hx, hy = int(wrist.x * frame_w), int(wrist.y * frame_h)
                        detected_hands.append((hx, hy, fingers, open5_strict))
            curr_time = time.time()
            
            # --- LOGIC TỰ CHUYỂN ID KHI MẤT DẤU (Chống đổi ID) ---
            current_people = []
            is_target_in_frame = False
            for result_pose in results_pose:
                if result_pose.boxes is None: continue
                boxes = result_pose.boxes
                keypoints = getattr(result_pose, "keypoints", None)

                # Match with segmentation result
                seg_result = results_seg[0] if len(results_seg) > 0 else None
                seg_boxes = seg_result.boxes if seg_result and seg_result.boxes else None
                
                for i, box in enumerate(boxes):
                    cls = int(box.cls[0].cpu().item())
                    if cls != 0: continue # Chi theo con nguoi
                    
                    track_id = int(box.id[0].item()) if box.id is not None else -1
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Segmentation Depth
                    poly_pts = None
                    if seg_boxes is not None and seg_result.masks is not None:
                        # Find matching box in seg_boxes by IoU or center distance
                        best_j = -1
                        min_dist = float('inf')
                        for j, sbox in enumerate(seg_boxes):
                            sx1, sy1, sx2, sy2 = sbox.xyxy[0].cpu().numpy()
                            sc_x, sc_y = (sx1+sx2)/2, (sy1+sy2)/2
                            bc_x, bc_y = (x1+x2)/2, (y1+y2)/2
                            dist = (sc_x-bc_x)**2 + (sc_y-bc_y)**2
                            if dist < min_dist and dist < 2500: # Centers must be within 50px
                                min_dist = dist
                                best_j = j
                        
                        if best_j != -1 and len(seg_result.masks.xy) > best_j:
                            poly_pts = seg_result.masks.xy[best_j]
                            
                    if poly_pts is not None and len(poly_pts) > 0:
                        d_m = get_depth_distance_m_seg(depth_frame, poly_pts, frame_w, frame_h)
                    else:
                        d_m = get_depth_distance_m(depth_frame, (x1, y1, x2, y2), frame_w, frame_h)
                        
                    delta_h = 1.8 - 0.6
                    d_ngang_m = math.sqrt(d_m**2 - delta_h**2) if d_m > delta_h else d_m

                    # Kiem tra Gio tay (YOLO Pose)
                    is_raising = False
                    if keypoints and keypoints.data is not None and i < len(keypoints.data):
                        kpts = keypoints.data[i].cpu().numpy()
                        if len(kpts) >= 11:
                            ls, r_sho, lw, rw = kpts[5], kpts[6], kpts[9], kpts[10]
                            def v_kpt(k): return k[2]>0.4 if len(k)>=3 else (k[0]>0 and k[1]>0)
                            if v_kpt(ls) and v_kpt(lw) and lw[1] < ls[1]: is_raising = True
                            if v_kpt(r_sho) and v_kpt(rw) and rw[1] < r_sho[1]: is_raising = True
                            
                        # Lay tam nguoi dung tu 2 vai, hoac mui
                        person_center_x, person_center_y = (x1 + x2) // 2, (y1 + y2) // 2
                        if len(kpts) >= 11:
                            ls, r_sho, nose = kpts[5], kpts[6], kpts[0]
                            if v_kpt(ls) and v_kpt(r_sho):
                                person_center_x = int((ls[0] + r_sho[0]) / 2)
                                person_center_y = int((ls[1] + r_sho[1]) / 2)
                            elif v_kpt(nose):
                                person_center_x, person_center_y = int(nose[0]), int(nose[1])
                    else:
                        person_center_x, person_center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    has_open_five = False
                    has_fist = False
                    open5_flags = []
                    fing_for_unlock = []
                    
                    for hx, hy, f, o5 in detected_hands:
                        if x1 <= hx <= x2 and y1 <= hy <= y2:
                            is_my_hand = False
                            
                            # Lớp lọc 1: Khoảng cách từ Bàn tay (MediaPipe) đến Cổ tay (YOLO Pose)
                            # Nếu đây là tay của người này, nó phải nằm sát cổ tay của họ.
                            if len(kpts) >= 11:
                                dist_l = math.hypot(hx - lw[0], hy - lw[1]) if v_kpt(lw) else float('inf')
                                dist_r = math.hypot(hx - rw[0], hy - rw[1]) if v_kpt(rw) else float('inf')
                                if min(dist_l, dist_r) < 100: # Tay nằm trong phạm vi 100 pixel từ cổ tay
                                    is_my_hand = True
                                    
                            # Lớp lọc 2: Khoảng cách không gian (Depth)
                            # Lấy chiều sâu thực tế của bàn tay
                            orig_hx = frame_w - 1 - hx
                            if 0 <= orig_hx < frame_w and 0 <= hy < frame_h:
                                hand_depth = depth_frame.get_distance(orig_hx, hy)
                                if hand_depth > 0 and abs(hand_depth - d_m) < 0.6: # Bàn tay không thể cách thân người quá 60cm
                                    is_my_hand = True
                                    
                            # Nếu có khung xương Pose nhưng tay không sát cổ tay, VÀ độ sâu không khớp -> ĐÂY LÀ TAY NGƯỜI KHÁC!
                            if not is_my_hand and len(kpts) >= 11:
                                continue 
                                
                            fing_for_unlock.append(f)
                            # Kiem tra neu ban tay duoc gio len cao (khuyu tay > 45% than)
                            if hy < (y1 + 0.45 * (y2 - y1)):
                                is_raising = True
                                if o5:
                                    has_open_five = True

                    if open5_flags: has_open_five = any(open5_flags)
                    if fing_for_unlock: has_fist = any(f <= 1 for f in fing_for_unlock)
                    elif is_raising: has_fist = True # Gio tay ma k thay ban tay thi kha nang dam

                    # --- LOGIC GẮN KHÓA (LOCK TARGET) ---
                    if track_id != -1 and self.locked_target_id is None:
                        if is_raising and has_open_five and d_ngang_m <= 5.0:
                            self.open5_confirm_count[track_id] = self.open5_confirm_count.get(track_id, 0) + 1
                            if track_id not in self.hand_raise_start:
                                self.hand_raise_start[track_id] = curr_time
                        else:
                            count = self.open5_confirm_count.get(track_id, 0)
                            if count > 0: self.open5_confirm_count[track_id] = count - 1
                            else: self.hand_raise_start.pop(track_id, None)
                        
                        if self.open5_confirm_count.get(track_id, 0) >= 2:
                            if track_id in self.hand_raise_start:
                                hold_time = curr_time - self.hand_raise_start[track_id]
                                cv2.putText(frame, f"DANG KHOA TARGET: {hold_time:.1f}s/2s", (int(x1), int(y1)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                if hold_time >= 2.0:
                                    self.locked_target_id = track_id
                                    self.locked_bbox = (x1, y1, x2, y2)
                                    self.robot_state = "COLLECTING"
                                    signal_bus.status_update.emit(f"ĐÃ KHÓA ! Đang tải dữ liệu không gian...")
                                    self.locked_center_pt = (person_center_x, person_center_y)
                                    threading.Thread(target=self.acquire_coords_and_navigate, args=(d_m, self.locked_center_pt), daemon=True).start()

                    # --- LOGIC MỞ KHÓA (UNLOCK TARGET - Bằng Năm đấm) ---
                    if track_id != -1 and track_id == self.locked_target_id:
                        if is_raising and has_fist:
                            self.fist_confirm_count += 1
                        else:
                            self.fist_confirm_count = 0
                            self.fist_hold_start = None

                        if self.fist_confirm_count > 3:
                            if self.fist_hold_start is None: self.fist_hold_start = curr_time
                            ho_time = curr_time - self.fist_hold_start
                            cv2.putText(frame, f"HUY LENH: {ho_time:.1f}s/2s", (int(x1), int(y1)-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            if ho_time >= 2.0:
                                self.nav.cancel_all()
                                self.locked_target_id = None
                                self.robot_state = "IDLE"
                                goal_pose = None; user_pose = None; robot_planned_path = []
                                signal_bus.status_update.emit(f"Trạng thái: Đang theo dõi người dùng")

                    # Cập nhật Giao diện Box
                    is_too_close = (0 < d_ngang_m < 1.0)
                    is_invalid = (d_ngang_m <= 0.0 or d_ngang_m > 5.0)
                    
                    if self.locked_target_id is not None:
                        if track_id == self.locked_target_id:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 3)
                            cv2.putText(frame, "LOCKED TARGET", (int(x1), int(y1)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            
                            if is_invalid:
                                dist_str = "Khoang cach: Khong ro / >5m"
                                txt_color = (0, 165, 255)
                            else:
                                dist_str = f"Khoang cach: {d_ngang_m:.2f}m"
                                txt_color = (0, 0, 255) if is_too_close else (0, 255, 255)
                                if is_too_close: dist_str += " (QUA GAN - KHONG DI)"
                            
                            cv2.putText(frame, dist_str, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 2)
                        else:
                            # Không hiển thị box người ngoài 5m để đỡ rối mắt
                            if not is_invalid:
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (100, 100, 100), 1)
                                cv2.putText(frame, f"{d_ngang_m:.2f}m", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
                    else:
                        if not is_invalid:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            dist_str = f"{d_ngang_m:.2f}m"
                            txt_color = (0, 0, 255) if is_too_close else (0, 255, 0)
                            if is_too_close: dist_str += " (Qua gan)"
                            cv2.putText(frame, dist_str, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 2)
                        
                        # Chỉ in log lên Terminal sau mỗi 1 giây để đỡ giật màn hình
                        if not hasattr(self, 'last_print_time') or curr_time - self.last_print_time > 1.0:
                            if not is_invalid:
                                print(f"[Tracking] Người dùng ID {track_id} đang ở khoảng cách: {d_ngang_m:.2f}m")
                            self.last_print_time = curr_time

            # Tự động mở khóa nếu mục tiêu mất dấu quá 3 giây
            if self.locked_target_id is not None:
                detected_ids = [int(box.id[0].item()) for r in results_pose if r.boxes and r.boxes.id is not None for box in r.boxes]
                if self.locked_target_id not in detected_ids:
                    if not hasattr(self, 'target_lost_time'):
                        self.target_lost_time = curr_time
                    elif curr_time - self.target_lost_time > 3.0:
                        if getattr(self, 'robot_state', '') != "MOVING":
                            self.nav.cancel_all()
                            self.locked_target_id = None
                            self.robot_state = "IDLE"
                            goal_pose = None; user_pose = None; robot_planned_path = []
                            signal_bus.status_update.emit("Mất dấu mục tiêu > 3s! Đã tự mở khóa.")
                        if hasattr(self, 'target_lost_time'):
                            del self.target_lost_time
                else:
                    if hasattr(self, 'target_lost_time'):
                        del self.target_lost_time

            signal_bus.frame_update.emit(frame)

    def acquire_coords_and_navigate(self, distance_m, center_pt):
        global goal_pose, user_pose, robot_planned_path, tf_listener, map_data, map_resolution, map_origin_x, map_origin_y
        rel = get_person_relative_position_m(None, center_pt, 640, 480, self.depth_intrinsics, distance_m)
        if rel is None:
            self.robot_state = "IDLE"
            self.locked_target_id = None
            goal_pose = None; user_pose = None; robot_planned_path = []
            return
        
        time.sleep(2)

        camera_offset_x = 0.475
        forward_m, left_m = rel[0] - camera_offset_x, rel[1]
        
        if tf_listener is not None:
            try:
                msg = PointStamped()
                msg.header.stamp = rospy.Time(0)
                msg.header.frame_id = "base_link"
                msg.point.x = forward_m
                msg.point.y = left_m
                msg.point.z = 0.0

                self.robot_state = "MOVING"
                tf_listener.waitForTransform("/map", "base_link", rospy.Time(0), rospy.Duration(1.0))
                pt = tf_listener.transformPoint("/map", msg)
                target_x, target_y = pt.point.x, pt.point.y
                
                user_pose = (target_x, target_y)
                
                robot_x, robot_y = robot_pose[0], robot_pose[1]
                dx, dy = target_x - robot_x, target_y - robot_y
                dist_to_target = math.hypot(dx, dy)
                
                # Chạy xa hơn inflation radius của costmap (Khoảng 1.0m - 1.2m)
                # Dưới 1.0m robot sẽ nghĩ điểm đến là một vật cản và sinh mã Code 8 (Reject)
                STOP_DISTANCE = 1.2
                if dist_to_target <= STOP_DISTANCE:
                    rospy.loginfo(f"[MirNav] Đã ở khoảng cách {STOP_DISTANCE}m. IDLE.")
                    self.robot_state = "IDLE"
                    self.locked_target_id = None
                    return
                # -- THUẬT TOÁN TÌM ĐIỂM DỪNG THÔNG MINH TRÁNH BÀN GHẾ --
                
                valid_goal_found = False
                search_distance = STOP_DISTANCE
                final_goal_x, final_goal_y = robot_x, robot_y # Fallback
                
                while search_distance < dist_to_target:
                    ratio = (dist_to_target - search_distance) / dist_to_target
                    test_x = robot_x + dx * ratio
                    test_y = robot_y + dy * ratio
                    
                    if map_data is not None and map_resolution > 0:
                        grid_x = int((test_x - map_origin_x) / map_resolution)
                        grid_y = int((test_y - map_origin_y) / map_resolution)
                        
                        h, w = map_data.shape
                        if 0 <= grid_x < w and 0 <= grid_y < h:
                            is_safe = True
                            # Kiểm tra bán kính 40cm xung quanh điểm test xem có đâm vào bàn ghế không
                            safe_radius_px = int(0.4 / map_resolution) 
                            for check_y in range(max(0, grid_y - safe_radius_px), min(h, grid_y + safe_radius_px)):
                                for check_x in range(max(0, grid_x - safe_radius_px), min(w, grid_x + safe_radius_px)):
                                    if map_data[check_y, check_x] > 50: # Có vật cản (100)
                                        is_safe = False
                                        break
                                if not is_safe:
                                    break
                                    
                            if is_safe:
                                final_goal_x, final_goal_y = test_x, test_y
                                valid_goal_found = True
                                break
                    else:
                        # Nếu không có map, cứ lấy điểm mặc định
                        final_goal_x, final_goal_y = test_x, test_y
                        valid_goal_found = True
                        break
                        
                    # Nếu vướng bàn, lùi xa người ra thêm 15cm về phía robot và thử lại
                    search_distance += 0.15

                if not valid_goal_found:
                    rospy.logwarn("[MirNav] Không tìm thấy chỗ đứng an toàn (Người đứng quá sát bàn/tường).")
                    self.robot_state = "IDLE"
                    self.locked_target_id = None
                    return
                
                # Tính góc Yaw hướng mặt vào người dùng (Mặc định)
                look_dx = target_x - final_goal_x
                look_dy = target_y - final_goal_y
                final_yaw = math.atan2(look_dy, look_dx)
                
                # -- NÂNG CẤP: DÒ VIỀN BÀN ĐỂ ĐỖ VUÔNG GÓC (Từ GitHub) --
                try:
                    if map_data is not None and map_resolution > 0:
                        import numpy as np
                        import cv2
                        px_t = int((target_x - map_origin_x) / map_resolution)
                        py_t = int((target_y - map_origin_y) / map_resolution)
                        
                        # Cắt vùng bản đồ 4x4m quanh khách hàng
                        win_px = int(4.0 / map_resolution)
                        h_map, w_map = map_data.shape
                        x1_m = max(0, px_t - win_px//2)
                        x2_m = min(w_map, px_t + win_px//2)
                        y1_m = max(0, py_t - win_px//2)
                        y2_m = min(h_map, py_t + win_px//2)
                        
                        # Rút trích viền bàn (Vật cản > 50)
                        local_mask = np.where(map_data[y1_m:y2_m, x1_m:x2_m] > 50, 255, 0).astype(np.uint8)
                        contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        best_contour = None
                        min_dist = float('inf')
                        for cnt in contours:
                            if cv2.contourArea(cnt) < 5: continue
                            # Chuyển contour về hệ pixel của toàn map
                            cnt_global = cnt + np.array([[[x1_m, y1_m]]])
                            dist = cv2.pointPolygonTest(cnt_global, (px_t, py_t), True)
                            
                            # Nếu khách đè lên mép bàn (dist >= 0) hoặc lấy mép bàn gần nhất
                            if dist >= 0:
                                best_contour = cnt_global
                                break
                            if abs(dist) < min_dist:
                                min_dist = abs(dist)
                                best_contour = cnt_global
                        
                        if best_contour is not None:
                            # Bao khung hình chữ nhật cho cái bàn
                            rect = cv2.minAreaRect(best_contour)
                            box = cv2.boxPoints(rect)
                            
                            goal_px_x = (final_goal_x - map_origin_x) / map_resolution
                            goal_px_y = (final_goal_y - map_origin_y) / map_resolution
                            
                            closest_edge_center = None
                            min_edge_dist = float('inf')
                            
                            # Tìm cạnh bàn gần nhất với điểm robot đỗ
                            for i in range(4):
                                p1 = box[i]
                                p2 = box[(i+1)%4]
                                edge_center = ((p1[0]+p2[0])/2.0, (p1[1]+p2[1])/2.0)
                                dist_to_goal = math.hypot(edge_center[0] - goal_px_x, edge_center[1] - goal_px_y)
                                if dist_to_goal < min_edge_dist:
                                    min_edge_dist = dist_to_goal
                                    closest_edge_center = edge_center
                            
                            if closest_edge_center is not None:
                                # Tính Yaw chĩa vuông góc tắp vào mép bàn thay vì chĩa vào người
                                look_dx_edge = closest_edge_center[0] - goal_px_x
                                look_dy_edge = closest_edge_center[1] - goal_px_y
                                final_yaw = math.atan2(look_dy_edge, look_dx_edge)
                                rospy.loginfo("[MirNav] Đã kích hoạt Yaw nội suy mép bàn (Vuông góc)!")
                except Exception as e:
                    rospy.logwarn(f"[MirNav] Lỗi nội suy mép bàn (Sẽ dùng Yaw HRI mặc định): {e}")
                    
                goal_pose = (final_goal_x, final_goal_y)
                
                signal_bus.status_update.emit(f"Chỉ định Đích: Navigation ({final_goal_x:.1f}, {final_goal_y:.1f})")
                
                # Thử gửi chỉ dẫn cho Move Base Daemon (Nó sẽ tự Retry nếu bị Code 8)
                if self.nav.send_goal(final_goal_x, final_goal_y, final_yaw):
                    signal_bus.status_update.emit("Robot đang di chuyển theo lộ trình Planner!")
                else:
                    rospy.logerr("[MirNav] Không kích hoạt được Daemon MoveBase!")

            except Exception as e:
                rospy.logerr(f"[Nav] Lỗi TF: {e}")
                signal_bus.status_update.emit(f"Lỗi tính toán không gian Map (TF)")
                self.robot_state = "IDLE"
                self.locked_target_id = None

    def map_callback(self, msg):
        try:
            width = msg.info.width
            height = msg.info.height
            resolution = msg.info.resolution
            origin_x = msg.info.origin.position.x
            origin_y = msg.info.origin.position.y
            data = np.array(msg.data).reshape((height, width))
            map_queue.put((data, resolution, origin_x, origin_y))
        except Exception as e:
            rospy.logerr(f"[Map] Lỗi xử lý map: {e}")

    def path_callback(self, msg):
        global robot_planned_path
        try:
            pts = []
            for pose in msg.poses:
                pts.append((pose.pose.position.x, pose.pose.position.y))
            robot_planned_path = pts
        except Exception as e:
            rospy.logerr(f"[Path] Lỗi xử lý lộ trình: {e}")

# ==============================================================================
# ENTRY POINT CỦA SUPER APP
# ==============================================================================
if __name__ == "__main__":
    rospy.init_node('xlanav_superapp', anonymous=True)
    
    app = QApplication(sys.argv)
    tf_listener = tf.TransformListener()
    
    window = MainWindow()
    window.show()
    
    core_logic = tracking_loop()
    rospy.Subscriber('/map', OccupancyGrid, core_logic.map_callback)
    
    # Lắng nghe nhiều topic Global Plan phổ biến để luôn vẽ được đường đi
    rospy.Subscriber('/move_base_node/SBPLLatticePlanner/plan', Path, core_logic.path_callback)
    rospy.Subscriber('/move_base_node/GlobalPlanner/plan', Path, core_logic.path_callback)
    rospy.Subscriber('/move_base/GlobalPlanner/plan', Path, core_logic.path_callback)
    rospy.Subscriber('/move_base_node/mir_global_planner/plan', Path, core_logic.path_callback)
    rospy.Subscriber('/move_base/NavfnROS/plan', Path, core_logic.path_callback)
    rospy.Subscriber('/mir_planner/global_path', Path, core_logic.path_callback)
    
    rospy.loginfo("[Map] Đã đăng ký subscriber /map & PATH")
    threading.Thread(target=rospy.spin, daemon=True).start()
    
    sys.exit(app.exec_())