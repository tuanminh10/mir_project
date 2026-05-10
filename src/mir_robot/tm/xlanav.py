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
        rospy.loginfo("[MirNav] KHOI DONG DAEMON SUBPROCESS... (Tranh loi ROS Node bi ngat va thoat giua chung) ")
        self._helper_script = os.path.join(os.path.dirname(__file__), 'send_goal_helper.py')
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        self.ip = ip
        self.api_url = f"http://{self.ip}/api/v2.0.0"
        auth = "Basic YWRtaW46OGM2OTc2ZTViNTQxMDQxNWJkZTkwOGJkNGRlZTE1ZGZiMTY3YTljODczZmM0YmI4YTgxZjZmMmFiNDQ4YTkxOA=="
        self.headers = {"Content-Type": "application/json", "Authorization": auth}

        self.is_navigating = False
        
        # CHAY BACKGROUND DAEMON 1 LAN DUY NHAT
        clean_env = os.environ.copy()
        clean_env.pop('PYTHONHOME', None)
        clean_env.pop('VIRTUAL_ENV', None)
        if 'PATH' in clean_env:
            paths = clean_env['PATH'].split(':')
            clean_env['PATH'] = ':'.join([p for p in paths if 'ai_venv' not in p and '.venv' not in p])
        if 'PYTHONPATH' in clean_env:
            ppaths = clean_env['PYTHONPATH'].split(':')
            clean_env['PYTHONPATH'] = ':'.join([p for p in ppaths if 'ai_venv' not in p and '.venv' not in p])

        cmd = ['/usr/bin/python3', self._helper_script]
        self._daemon_process = subprocess.Popen(
            cmd, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            env=clean_env,
            bufsize=1 # Line buffered
        )

        def read_stdout():
            while self._daemon_process.poll() is None:
                line = self._daemon_process.stdout.readline()
                if line:
                    rospy.loginfo(f"[Daemon] {line.strip()}")
                    if line.startswith("STATE_3"):
                        rospy.loginfo(f"[MirNav move_base] 🎯 Đã cập bến điểm tĩnh (Helper báo thành công)!")
                        self.is_navigating = False
                        global goal_pose
                        goal_pose = None
                    elif line.startswith("STATE_8"):
                        rospy.loginfo(f"[MirNav move_base] ❌ Lệnh bị hủy ngang/Preempt code 8.")
                        self.is_navigating = False
                    elif line.startswith("ERROR_"):
                        rospy.logerr(f"[MirNav move_base] LỖI DAEMON: {line.strip()}")
        
        def read_stderr():
            while self._daemon_process.poll() is None:
                err = self._daemon_process.stderr.readline()
                if err:
                    rospy.logwarn(f"[Daemon Err] {err.strip()}")

        threading.Thread(target=read_stdout, daemon=True).start()
        threading.Thread(target=read_stderr, daemon=True).start()

    def __del__(self):
        if hasattr(self, '_daemon_process') and self._daemon_process:
            self._daemon_process.terminate()

    def ensure_ready(self):
        """Đảm bảo MiR ở trạng thái Ready (State 3) để nhận ROS Goal"""
        try:
            # Chỉ gỡ lỗi và ép về Ready, không Pause để tránh Preempt code 8
            requests.delete(f"{self.api_url}/status", headers=self.headers, timeout=2)
            time.sleep(0.2)
            requests.put(f"{self.api_url}/status", headers=self.headers, json={"state_id": 3}, timeout=2)
            time.sleep(0.5)
            rospy.loginfo("[MirNav] Đã ép MiR Web Dashboard sang trạng thái Ready (Màu xanh)")
        except Exception as e:
            rospy.logerr(f"[MirNav] Lỗi ensure_ready: {e}")

    def cancel_all(self):
        """Hủy mọi lệnh và dừng xe"""
        self.is_navigating = False
        global robot_planned_path, goal_pose
        robot_planned_path = []
        goal_pose = None
        
        # Gửi lệnh CANCEL tới daemon
        if hasattr(self, '_daemon_process') and self._daemon_process.poll() is None:
            try:
                self._daemon_process.stdin.write("CANCEL\n")
                self._daemon_process.stdin.flush()
            except Exception as e:
                rospy.logerr(f"Lỗi gửi CANCEL tới daemon: {e}")

        try:
            self.cmd_vel_pub.publish(Twist())
        except Exception as e:
            rospy.logerr(f"Lỗi cancel cmd_vel: {e}")

    def send_goal(self, goal_x, goal_y, goal_yaw=0.0):
        """Gửi goal qua file helper Python 3.8 để tránh lỗi deserialize ActionLib"""
        self.is_navigating = True
        
        if hasattr(self, '_daemon_process') and self._daemon_process.poll() is None:
            rospy.loginfo(f"[MirNav move_base] Gửi lệnh: {goal_x},{goal_y},{goal_yaw}")
            try:
                self._daemon_process.stdin.write(f"{goal_x},{goal_y},{goal_yaw}\n")
                self._daemon_process.stdin.flush()
                return True
            except Exception as e:
                rospy.logerr(f"Lỗi gửi lệnh tới daemon: {e}")
                return False
        else:
            rospy.logerr("[MirNav move_base] Daemon đã chết, không thể gửi lệnh!")
            return False

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

def get_person_relative_position_m(depth_frame, box, frame_w, frame_h, depth_intrinsics, distance_m):
    x1, y1, x2, y2 = map(int, box)
    center_x = (x1 + x2) // 2
    orig_px = frame_w - 1 - center_x
    if distance_m <= 0: return None
    if depth_intrinsics is None:
        hfov_rad = math.radians(69.0)
        angle = ((orig_px - frame_w / 2.0) / frame_w) * hfov_rad
        x_cam = distance_m * math.tan(angle)
    else:
        x_cam = (orig_px - depth_intrinsics.ppx) / depth_intrinsics.fx * distance_m
    return (distance_m, -x_cam) # (forward_m, left_m)

class tracking_loop:
    def __init__(self):
        self.nav = MirNavigator()
        # Tải mô hình YOLO Pose chuyên dụng cho con người để lấy khung xương (Keypoints)
        self.model = YOLO('yolo11n-pose.pt')
        
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        self.camera_ready = False
        try:
            self.pipeline.start(config)
            self.camera_ready = True
            self.align = rs.align(rs.stream.color)
            
            profile = self.pipeline.get_active_profile()
            depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            self.depth_intrinsics = depth_profile.get_intrinsics()
        except RuntimeError as e:
            rospy.logerr(f"❌ KHÔNG TÌM THẤY CAMERA REALSENSE: {e}")
            self.depth_intrinsics = None

        mp_hands = mp.solutions.hands
        self.hands_detector = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.robot_state = "IDLE"  # IDLE, LOCKED, COLLECTING, MOVING
        self.locked_target_id = None
        
        self.hand_raise_start = {}
        self.open5_confirm_count = {}
        self.fist_hold_start = None
        self.fist_confirm_count = 0

        self.worker_thread = threading.Thread(target=self.run, daemon=True)
        self.worker_thread.start()

    def run(self):
        # Thiết lập cập nhật UI nếu không có camera
        if not self.camera_ready:
            signal_bus.status_update.emit(f"Lỗi: Không kết nối Camera 3D! Vẫn chờ /map")
        
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

            # 1. MediaPipe ngón tay
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = self.hands_detector.process(rgb_frame)
            detected_hands = []

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

            # 2. YOLO Nhan dien nguoi
            results = self.model.track(frame, conf=0.45, persist=True, tracker="bytetrack.yaml", verbose=False)
            curr_time = time.time()
            
            # --- LOGIC TỰ CHUYỂN ID KHI MẤT DẤU (Chống đổi ID) ---
            current_people = []
            is_target_in_frame = False
            for result in results:
                if result.boxes is None: continue
                for box in result.boxes:
                    if int(box.cls[0].item()) == 0:
                        tid = int(box.id[0].item()) if box.id is not None else -1
                        bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                        current_people.append({"id": tid, "box": (bx1, by1, bx2, by2)})
                        if tid == self.locked_target_id:
                            is_target_in_frame = True
                            self.locked_bbox = (bx1, by1, bx2, by2)

            if self.locked_target_id is not None and not is_target_in_frame and hasattr(self, 'locked_bbox') and self.locked_bbox is not None:
                lx1, ly1, lx2, ly2 = self.locked_bbox
                lcx, lcy = (lx1 + lx2) / 2, (ly1 + ly2) / 2
                best_match_id = None
                min_dist = float('inf')
                for p in current_people:
                    px1, py1, px2, py2 = p["box"]
                    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
                    dist = ((pcx - lcx)**2 + (pcy - lcy)**2)**0.5
                    if dist < 150 and dist < min_dist:
                        min_dist = dist
                        best_match_id = p["id"]
                        self.locked_bbox = p["box"]
                
                if best_match_id is not None:
                    print(f"[TARGET] Đã cập nhật ID mục tiêu: {self.locked_target_id} -> {best_match_id} (Do nhầm lẫn ID)")
                    self.locked_target_id = best_match_id
                    if hasattr(self, 'target_lost_time'):
                        del self.target_lost_time

            for result in results:
                if result.boxes is None: continue
                boxes = result.boxes
                keypoints = getattr(result, "keypoints", None)

                for i, box in enumerate(boxes):
                    cls = int(box.cls[0].cpu().item())
                    if cls != 0: continue # Chi theo con nguoi
                    
                    track_id = int(box.id[0].item()) if box.id is not None else -1
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Tinh khoang cach
                    d_m = get_depth_distance_m(depth_frame, (x1, y1, x2, y2), frame_w, frame_h)
                    delta_h = 1.8 - 0.6
                    d_ngang_m = math.sqrt(d_m**2 - delta_h**2) if d_m > delta_h else d_m

                    # Kiem tra Gio tay (qua vai)
                    is_raising = False
                    if keypoints and keypoints.data is not None and i < len(keypoints.data):
                        kpts = keypoints.data[i].cpu().numpy()
                        if len(kpts) >= 11:
                            ls, rs, lw, rw = kpts[5], kpts[6], kpts[9], kpts[10]
                            def v_kpt(k): return k[2]>0.4 if len(k)>=3 else (k[0]>0 and k[1]>0)
                            if v_kpt(ls) and v_kpt(lw) and lw[1] < ls[1]: is_raising = True
                            if v_kpt(rs) and v_kpt(rw) and rw[1] < rs[1]: is_raising = True

                    # Kiem tra open5 & fist cho rieng khoi box nay
                    has_open_five = False
                    has_fist = False
                    open5_flags = []
                    fing_for_unlock = []
                    
                    if is_raising:
                        for hx, hy, f, o5 in detected_hands:
                            if x1 <= hx <= x2 and y1 <= hy <= y2:
                                open5_flags.append(o5)
                                fing_for_unlock.append(f)
                    else:
                        for hx, hy, f, o5 in detected_hands:
                            if x1 <= hx <= x2 and y1 <= hy <= y2:
                                fing_for_unlock.append(f)

                    if open5_flags: has_open_five = any(open5_flags)
                    if fing_for_unlock: has_fist = any(f <= 1 for f in fing_for_unlock)
                    elif is_raising: has_fist = True # Gio tay ma k thay ban tay thi kha nang dam

                    # --- LOGIC GẮN KHÓA (LOCK TARGET) ---
                    if track_id != -1 and self.locked_target_id is None:
                        if is_raising and has_open_five:
                            self.open5_confirm_count[track_id] = self.open5_confirm_count.get(track_id, 0) + 1
                        else:
                            self.open5_confirm_count.pop(track_id, None)
                            self.hand_raise_start.pop(track_id, None)

                        if self.open5_confirm_count.get(track_id, 0) >= 3:
                            if track_id not in self.hand_raise_start: self.hand_raise_start[track_id] = curr_time
                            hold_time = curr_time - self.hand_raise_start[track_id]
                            
                            cv2.putText(frame, f"DANG KHOA TARGET: {hold_time:.1f}s/5s", (int(x1), int(y1)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            if hold_time >= 5.0:
                                self.locked_target_id = track_id
                                self.locked_bbox = (x1, y1, x2, y2)
                                self.robot_state = "COLLECTING"
                                signal_bus.status_update.emit(f"ĐÃ KHÓA ! Đang tải dữ liệu không gian...")
                                threading.Thread(target=self.acquire_coords_and_navigate, args=(d_m, x1, y1, x2, y2), daemon=True).start()

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
                            cv2.putText(frame, f"HUY LENH: {ho_time:.1f}s/5s", (int(x1), int(y1)-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            if ho_time >= 5.0:
                                self.nav.cancel_all()
                                self.locked_target_id = None
                                self.robot_state = "IDLE"
                                signal_bus.status_update.emit(f"Trạng thái: Đang theo dõi người dùng")

                    # Cập nhật Giao diện Box
                    is_too_close = d_ngang_m < 1.0
                    
                    if self.locked_target_id is not None:
                        if track_id == self.locked_target_id:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 3)
                            cv2.putText(frame, "LOCKED TARGET", (int(x1), int(y1)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            
                            dist_str = f"Khoang cach: {d_ngang_m:.2f}m"
                            txt_color = (0, 0, 255) if is_too_close else (0, 255, 255)
                            if is_too_close: dist_str += " (QUA GAN - KHONG DI)"
                            
                            cv2.putText(frame, dist_str, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 2)
                        else:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (100, 100, 100), 1)
                            cv2.putText(frame, f"{d_ngang_m:.2f}m", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
                    else:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        dist_str = f"{d_ngang_m:.2f}m"
                        txt_color = (0, 0, 255) if is_too_close else (0, 255, 0)
                        if is_too_close: dist_str += " (Qua gan)"
                        cv2.putText(frame, dist_str, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 2)

            # Tự động mở khóa nếu mục tiêu mất dấu quá 3 giây
            if self.locked_target_id is not None:
                detected_ids = [int(box.id[0].item()) for r in results if r.boxes and r.boxes.id is not None for box in r.boxes]
                if self.locked_target_id not in detected_ids:
                    if not hasattr(self, 'target_lost_time'):
                        self.target_lost_time = curr_time
                    elif curr_time - self.target_lost_time > 3.0:
                        self.nav.cancel_all()
                        self.locked_target_id = None
                        self.robot_state = "IDLE"
                        signal_bus.status_update.emit("Mất dấu mục tiêu > 3s! Đã tự mở khóa.")
                        del self.target_lost_time
                else:
                    if hasattr(self, 'target_lost_time'):
                        del self.target_lost_time

            signal_bus.frame_update.emit(frame)

    def acquire_coords_and_navigate(self, distance_m, x1, y1, x2, y2):
        rel = get_person_relative_position_m(None, (x1, y1, x2, y2), 640, 480, self.depth_intrinsics, distance_m)
        if rel is None:
            self.robot_state = "IDLE"
            self.locked_target_id = None
            return
        
        time.sleep(2)

        camera_offset_x = 0.475
        forward_m, left_m = rel[0] - camera_offset_x, rel[1]
        
        global tf_listener, goal_pose
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
                ratio = (dist_to_target - STOP_DISTANCE) / dist_to_target
                final_goal_x = robot_x + dx * ratio
                final_goal_y = robot_y + dy * ratio
                final_yaw = math.atan2(dy, dx)
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
