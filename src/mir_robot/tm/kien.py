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
import navigationcacdiem as nav
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
STOP_DISTANCE = 0.65 # Khoảng cách dừng cách người giơ tay


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
       rospy.loginfo("[MirNav] KHOI DONG DAEMON SUBPROCESS... (Tranh loi ROS Node bi ngat va thoat giua chung) ")
       self._helper_script = os.path.join(os.path.dirname(__file__), 'send_goal_helper.py')
       self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
      
       self.ip = ip
       self.api_url = f"http://{self.ip}/api/v2.0.0"
       auth = "Basic YWRtaW46OGM2OTc2ZTViNTQxMDQxNWJkZTkwOGJkNGRlZTE1ZGZiMTY3YTljODczZmM0YmI4YTgxZjZmMmFiNDQ4YTkxOA=="
       self.headers = {"Content-Type": "application/json", "Authorization": auth}


       self.is_navigating = False
       self.mir_headers = nav.api_login()  # REST API login cho api_navigate
      
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
# ỨNG DỤNG LÕI (Camera, YOLO, Math Engine, Threading)
# ==============================================================================
# YOLO Pose keypoint indices (COCO 17-keypoint format)
KP_NOSE = 0
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_ELBOW, KP_R_ELBOW = 7, 8
KP_L_WRIST, KP_R_WRIST = 9, 10


def extract_3d_coordinates_from_pc(vertices, box, frame_w, frame_h):
   """Trích xuất cụm tọa độ 3D từ mảng Vertices của PointCloud nằm trong Bounding Box YOLO"""
   x1, y1, x2, y2 = map(int, box)
   width = x2 - x1
   height = y2 - y1
  
   # Cắt lấy nửa trên của bounding box để CHẮC CHẮN né cái bàn phía trước
   roi_x1 = max(0, int(x1 + width * 0.20))
   roi_x2 = min(frame_w, int(x2 - width * 0.20))
   roi_y1 = max(0, int(y1 + height * 0.05))
   roi_y2 = min(frame_h, int(y1 + height * 0.45))
  
   if roi_x2 <= roi_x1 or roi_y2 <= roi_y1: return None
      
   roi_pts = vertices[roi_y1:roi_y2, roi_x1:roi_x2]
   valid_mask = (roi_pts[:, :, 2] > 0.3) & (roi_pts[:, :, 2] < 6.0)
   valid_pts = roi_pts[valid_mask]
  
   if len(valid_pts) < 10: return None
      
   # GIẢI THUẬT LỌC FOREGROUND (Khách hàng):
   Z_values = valid_pts[:, 2]
   p15_Z = np.percentile(Z_values, 15) # Dùng phân vị 15% để bỏ qua nhiễu hạt
  
   person_mask = (Z_values >= p15_Z - 0.1) & (Z_values <= p15_Z + 0.5)
   person_pts = valid_pts[person_mask]
  
   if len(person_pts) < 5:
       person_pts = valid_pts # Fallback an toàn
      
   median_pt = np.median(person_pts, axis=0)
   X, Y, Z = median_pt
  
   return {
       'Z': float(Z),
       'X_raw': float(X),
       'Y_raw': float(Y),
       'X': float(-X)
   }


def find_3d_obstacle_in_path(vertices, target_Z, floor_y_thresh=1.6, width_m=0.35):
   """Quét dọc theo hành lang trước robot để tìm vật cản 3D lơ lửng"""
   mask_Z = (vertices[:, :, 2] > 1.0) & (vertices[:, :, 2] < target_Z - 0.2)
   mask_X = (vertices[:, :, 0] > -width_m) & (vertices[:, :, 0] < width_m)
   mask_Y = (vertices[:, :, 1] > 0.0) & (vertices[:, :, 1] < 1.4)
  
   valid_mask = mask_Z & mask_X & mask_Y
   valid_pts = vertices[valid_mask]
  
   if len(valid_pts) > 50:
       min_idx = np.argmin(valid_pts[:, 2])
       obs_pt = valid_pts[min_idx]
       return float(-obs_pt[0]), float(obs_pt[2])
   return None, None


class tracking_loop:
   def __init__(self):
       import torch
       self.device = 0 if torch.cuda.is_available() else 'cpu'
       self.nav = MirNavigator()
       self.camera_ready = False
       self.robot_state = "IDLE"  # IDLE, LOCKED, COLLECTING, MOVING
       self.locked_target_id = None
      
       self.is_scanning = False
       self.raised_hand_trackers = {}
      
       self.worker_thread = threading.Thread(target=self.run, daemon=True)
       self.worker_thread.start()


   def run(self):
       global goal_pose, user_pose, robot_planned_path
       print("⏳ Đang khởi động mô hình AI (YOLO11s-Pose) trong luồng ngầm...")
       self.model_pose = YOLO('yolo11n-pose.pt')
       if self.device == 0:
           rospy.loginfo("[AI] Phát hiện GPU RTX! Đang đưa model lên CUDA...")
           self.model_pose.to('cuda')
      
       print("⏳ Đang kết nối Camera RealSense 3D...")
       self.pipeline = rs.pipeline()
       config = rs.config()
       config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
       config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
       try:
           self.pipeline.start(config)
           self.camera_ready = True
           self.align = rs.align(rs.stream.color)
           self.pc = rs.pointcloud()
           print("✅ Đã kết nối Camera RealSense thành công!")
       except RuntimeError as e:
           rospy.logerr(f"❌ KHÔNG TÌM THẤY CAMERA REALSENSE: {e}")
           self.depth_intrinsics = None


       print("🚀 Đã khởi động toàn bộ AI thành công! Bắt đầu quét mục tiêu...")
      
       if not self.camera_ready:
           signal_bus.status_update.emit(f"Lỗi: Không kết nối Camera 3D! Vẫn chờ /map")
           print("❌ LỖI: Camera không khả dụng. Giao diện sẽ hiển thị màn hình chờ.")
      
       while not rospy.is_shutdown():
           if not self.camera_ready:
               rospy.sleep(1)
               continue
          
           try: frames = self.pipeline.wait_for_frames(timeout_ms=1000)
           except: continue
              
           aligned = self.align.process(frames)
           depth_frame = aligned.get_depth_frame()
           color_frame = aligned.get_color_frame()
           if not depth_frame or not color_frame: continue


           frame = np.asanyarray(color_frame.get_data())
           frame = cv2.flip(frame, 1)
           frame_h, frame_w = frame.shape[:2]


           annotated_frame = frame.copy()
          
           # KHI ĐANG DI CHUYỂN -> TẮT QUÉT YOLO ĐỂ TIẾT KIỆM TÀI NGUYÊN VÀ KHÔNG BỊ LOẠN
           if self.robot_state == "MOVING":
               cv2.putText(annotated_frame, "ROBOT IS NAVIGATING...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
               signal_bus.frame_update.emit(annotated_frame)
               continue


           results_pose = self.model_pose.track(frame, classes=[0], conf=0.45, iou=0.6, persist=True, tracker="bytetrack.yaml", verbose=False, half=(self.device==0), device=self.device)


           hand_detected_box = None
           current_time = time.time()
          
           if results_pose[0].boxes and results_pose[0].keypoints is not None:
               boxes = results_pose[0].boxes.xyxy.cpu().numpy()
               keypoints = results_pose[0].keypoints
               kpts_xy = keypoints.xy.cpu().numpy()
               kpts_conf = keypoints.conf.cpu().numpy() if keypoints.conf is not None else None
              
               track_ids = results_pose[0].boxes.id.int().cpu().tolist() if results_pose[0].boxes.id is not None else [-1] * len(boxes)
               new_trackers = {}
              
               raised_arms = []
               for i in range(len(boxes)):
                   if kpts_conf is None: continue
                   kp = kpts_xy[i]
                   cf = kpts_conf[i]
                   box_h = boxes[i][3] - boxes[i][1]
                   native_t_id = track_ids[i]
                  
                   def get_valid_arm(kp_wrist, kp_elbow, side):
                       if cf[kp_wrist] > 0.25 and cf[kp_elbow] > 0.25:
                           wx, wy = kp[kp_wrist]
                           ex, ey = kp[kp_elbow]
                           forearm_len = math.hypot(wx - ex, wy - ey)
                          
                           # 1. Cẳng tay dốc thẳng đứng tắp (> 85%) thay vì chỉ 75% như cũ
                           is_pointing_up = (ey - wy) > (forearm_len * 0.85)
                          
                           # 2. Chiều dài cẳng tay đủ lớn
                           is_long_enough = forearm_len > max(30, box_h * 0.1)
                          
                           # 3. YÊU CẦU: Cùi chỏ phải cao hơn vai (điều kiện cốt lõi của "giơ cao hết mức")
                           shoulder_idx = KP_L_SHOULDER if side == 'L' else KP_R_SHOULDER
                           if cf[shoulder_idx] > 0.3:
                               is_elbow_raised = ey < kp[shoulder_idx][1]
                           elif cf[KP_NOSE] > 0.3:
                               # Nếu không thấy vai, nhượng bộ cùi chỏ tiệm cận mũi
                               is_elbow_raised = ey < (kp[KP_NOSE][1] + forearm_len * 0.5)
                           else:
                               is_elbow_raised = ey < (boxes[i][1] + box_h * 0.3)
                          
                           # 4. YÊU CẦU: Cổ tay vươn thẳng vút lên trời (cao hơn hẳn đỉnh đầu/mũi)
                           if cf[KP_NOSE] > 0.3:
                               # Cổ tay phải cao hơn mũi MỘT ĐOẠN gần bằng độ dài cẳng tay
                               is_high_enough = wy < (kp[KP_NOSE][1] - forearm_len * 0.4)
                           else:
                               is_high_enough = wy < boxes[i][1]
                          
                           if is_pointing_up and is_long_enough and is_elbow_raised and is_high_enough:
                               return (side, wx, wy, ex, ey, forearm_len)
                       return None
                      
                   l_arm = get_valid_arm(KP_L_WRIST, KP_L_ELBOW, 'L')
                   if l_arm: raised_arms.append((native_t_id, l_arm))
                   r_arm = get_valid_arm(KP_R_WRIST, KP_R_ELBOW, 'R')
                   if r_arm: raised_arms.append((native_t_id, r_arm))
              
               raising_hands_ids = set()
               person_data = []
               for i, box in enumerate(boxes):
                   t_id = track_ids[i]
                   x1, y1, x2, y2 = map(int, box)
                   kp = kpts_xy[i]
                   cf = kpts_conf[i] if kpts_conf is not None else None
                  
                   if cf is not None and cf[KP_NOSE] > 0.4:
                       ref_x, ref_y = kp[KP_NOSE]
                   else:
                       ref_x = (x1 + x2) / 2
                       ref_y = y1 + (y2 - y1) * 0.1
                      
                   fallback_shoulder_y = ref_y + (y2 - y1) * 0.15
                  
                   if cf is not None and cf[KP_L_SHOULDER] > 0.3:
                       l_shoulder_x, l_shoulder_y = kp[KP_L_SHOULDER]
                   else:
                       l_shoulder_x, l_shoulder_y = ref_x, fallback_shoulder_y
                      
                   if cf is not None and cf[KP_R_SHOULDER] > 0.3:
                       r_shoulder_x, r_shoulder_y = kp[KP_R_SHOULDER]
                   else:
                       r_shoulder_x, r_shoulder_y = ref_x, fallback_shoulder_y
                      
                   person_data.append({
                       'id': t_id,
                       'l_shoulder_x': l_shoulder_x, 'l_shoulder_y': l_shoulder_y,
                       'r_shoulder_x': r_shoulder_x, 'r_shoulder_y': r_shoulder_y,
                       'box': box, 'height': y2 - y1
                   })
              
               for native_t_id, (side, arm_wx, arm_wy, arm_ex, arm_ey, forearm_len) in raised_arms:
                   best_person_id = None
                   best_score = float('inf')
                  
                   for p in person_data:
                       # Ràng buộc X: Cùi chỏ không được chìa ra quá xa so với vai
                       shoulder_width = abs(p['l_shoulder_x'] - p['r_shoulder_x'])
                       if shoulder_width < 10: shoulder_width = p['height'] * 0.2
                      
                       target_shoulder_x = p['l_shoulder_x'] if side == 'L' else p['r_shoulder_x']
                       target_shoulder_y = p['l_shoulder_y'] if side == 'L' else p['r_shoulder_y']
                      
                       if abs(arm_ex - target_shoulder_x) > shoulder_width * 1.5:
                           continue
                          
                       forearm_ratio = forearm_len / p['height']
                       if 0.10 < forearm_ratio < 0.65:
                           if arm_wy < target_shoulder_y - (forearm_len * 0.2):
                               dist_to_shoulder = math.hypot(arm_ex - target_shoulder_x, arm_ey - target_shoulder_y)
                               anatomical_score = dist_to_shoulder / forearm_len
                              
                               # Ưu tiên dự đoán gốc của YOLO
                               if p['id'] == native_t_id:
                                   anatomical_score *= 0.5
                                  
                               if anatomical_score < 2.0 and anatomical_score < best_score:
                                   best_score = anatomical_score
                                   best_person_id = p['id']
                              
                   if best_person_id is not None:
                       raising_hands_ids.add(best_person_id)
              
               for i, box in enumerate(boxes):
                   t_id = track_ids[i]
                   x1, y1, x2, y2 = map(int, box)
                  
                   if self.locked_target_id == t_id:
                       hand_detected_box = box
                       if self.robot_state != "MOVING":
                           self.is_scanning = True
                       cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                       cv2.putText(annotated_frame, "TARGET LOCKED!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                       continue
                      
                   first_detected, last_seen = self.raised_hand_trackers.get(t_id, (current_time, 0))
                  
                   if t_id in raising_hands_ids:
                       last_seen = current_time
                       new_trackers[t_id] = (first_detected, last_seen)
                       duration = current_time - first_detected
                      
                       if duration >= 2.5:
                           self.locked_target_id = t_id
                           hand_detected_box = box
                           if self.robot_state != "MOVING":
                               self.is_scanning = True
                           cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                           cv2.putText(annotated_frame, "TARGET LOCKED!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                       else:
                           cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                           cv2.putText(annotated_frame, f"Locking: {2.5 - duration:.1f}s", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                          
                   elif current_time - last_seen < 0.5:
                       new_trackers[t_id] = (first_detected, last_seen)
                       duration = current_time - first_detected
                       cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                       cv2.putText(annotated_frame, f"Locking: {2.5 - duration:.1f}s", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                   else:
                       cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
              
               self.raised_hand_trackers = new_trackers
          
           if self.is_scanning:
               if hand_detected_box is not None:
                   # QUAN TRỌNG: YOLO chạy trên frame đã flip ngang (cv2.flip code=1),
                   # nhưng PointCloud vertices căn theo frame GỐC (chưa flip).
                   # Phải un-flip tọa độ bounding box trước khi tra cứu vào vertices.
                   fx1, fy1, fx2, fy2 = hand_detected_box
                   unflipped_box = np.array([frame_w - fx2, fy1, frame_w - fx1, fy2])
                  
                   self.pc.map_to(color_frame)
                   points = self.pc.calculate(depth_frame)
                   vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(480, 640, 3)
                   pc_data = extract_3d_coordinates_from_pc(vertices, unflipped_box, 640, 480)
                  
                   if pc_data is not None:
                       self.is_scanning = False
                       self.robot_state = "MOVING"
                      
                       rel_Z = pc_data['Z']
                       rel_X = pc_data['X']
                       X_raw = pc_data['X_raw']
                       Y_raw = pc_data['Y_raw']


                       euclid_d = math.sqrt(X_raw**2 + Y_raw**2 + rel_Z**2)
                       delta_h = 1.8 - 0.7
                       if euclid_d > delta_h:
                           horizontal_sq = euclid_d**2 - delta_h**2
                           d_ngang_toan_phan = math.sqrt(horizontal_sq)
                           if d_ngang_toan_phan**2 > rel_X**2:
                               d_forward = math.sqrt(d_ngang_toan_phan**2 - rel_X**2)
                           else:
                               d_forward = d_ngang_toan_phan
                       else:
                           d_forward = rel_Z
                          
                       threading.Thread(target=self.acquire_coords_and_navigate, args=(d_forward, rel_X), daemon=True).start()


           if self.locked_target_id is not None and self.robot_state != "MOVING":
               detected_ids = [int(box.id[0].item()) for r in results_pose if r.boxes and r.boxes.id is not None for box in r.boxes]
               if self.locked_target_id not in detected_ids:
                   if not hasattr(self, 'target_lost_time'):
                       self.target_lost_time = current_time
                   elif current_time - self.target_lost_time > 3.0:
                       self.nav.cancel_all()
                       self.locked_target_id = None
                       self.robot_state = "IDLE"
                       goal_pose = None; user_pose = None; robot_planned_path = []
                       self.raised_hand_trackers.clear()
                       signal_bus.status_update.emit("Mất dấu mục tiêu > 3s! Đã tự mở khóa.")
                       del self.target_lost_time
               else:
                   if hasattr(self, 'target_lost_time'):
                       del self.target_lost_time


           cv2.putText(annotated_frame, "SCANNING" if self.is_scanning else "IDLE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) if self.is_scanning else (0,255,0), 2)
           signal_bus.frame_update.emit(annotated_frame)


   def acquire_coords_and_navigate(self, forward_m, left_m):
       global goal_pose, user_pose, robot_planned_path, tf_listener, map_data, map_resolution, map_origin_x, map_origin_y
      
       signal_bus.status_update.emit("Đã lấy được tọa độ 3D. Đang lập bản đồ...")
       time.sleep(1)


       camera_offset_x = 0.475
       forward_m = forward_m - camera_offset_x
      
       if tf_listener is not None:
           try:
               msg = PointStamped()
               msg.header.stamp = rospy.Time(0)
               msg.header.frame_id = "/base_link"
               msg.point.x = forward_m
               msg.point.y = left_m
               msg.point.z = 0.0


               self.robot_state = "MOVING"
               tf_listener.waitForTransform("/map", "/base_link", rospy.Time(0), rospy.Duration(1.0))
               pt = tf_listener.transformPoint("/map", msg)
               target_x, target_y = pt.point.x, pt.point.y
              
               user_pose = (target_x, target_y)
              
               if robot_pose is None:
                   rospy.logwarn("[MirNav] Chưa nhận được vị trí robot (robot_pose is None). Thử lại sau.")
                   self.robot_state = "IDLE"
                   self.locked_target_id = None
                   return
                  
               robot_x, robot_y = robot_pose[0], robot_pose[1]
               dx, dy = target_x - robot_x, target_y - robot_y
               dist_to_target = math.hypot(dx, dy)
              
               STOP_DISTANCE = 1.0
               if dist_to_target <= STOP_DISTANCE:
                   rospy.loginfo(f"[MirNav] Đã ở khoảng cách {STOP_DISTANCE}m. IDLE.")
                   self.robot_state = "IDLE"
                   self.locked_target_id = None
                   return
              
               # =============================================================
               # THUẬT TOÁN TÌM ĐIỂM ĐỖ 360° QUANH VỊ TRÍ NGƯỜI
               # Ưu tiên 1: Điểm đỗ rộng rãi (an toàn 0.7m) để MiR không phải "suy nghĩ lâu"
               # Ưu tiên 2: Điểm đỗ hẹp hơn (0.55m -> 0.45m) nếu khách ngồi ở góc kẹt
               # =============================================================
               valid_goal_found = False
               final_goal_x, final_goal_y = robot_x, robot_y
               best_dist_to_robot = float('inf')
              
               for desired_safe_radius in [0.7, 0.55, 0.45]:
                   if valid_goal_found: break
                   for search_r in [STOP_DISTANCE, STOP_DISTANCE + 0.3, STOP_DISTANCE + 0.6, STOP_DISTANCE + 0.9, STOP_DISTANCE + 1.2]:
                       if valid_goal_found: break
                       for angle_deg in range(0, 360, 5):
                           angle_rad = math.radians(angle_deg)
                           test_x = target_x + search_r * math.cos(angle_rad)
                           test_y = target_y + search_r * math.sin(angle_rad)
                          
                           if map_data is not None and map_resolution > 0:
                               grid_x = int((test_x - map_origin_x) / map_resolution)
                               grid_y = int((test_y - map_origin_y) / map_resolution)
                              
                               h, w = map_data.shape
                               if not (0 <= grid_x < w and 0 <= grid_y < h):
                                   continue
                              
                               is_safe = True
                               safe_radius_px = int(desired_safe_radius / map_resolution)
                               for check_y in range(max(0, grid_y - safe_radius_px), min(h, grid_y + safe_radius_px)):
                                   for check_x in range(max(0, grid_x - safe_radius_px), min(w, grid_x + safe_radius_px)):
                                       cell = map_data[check_y, check_x]
                                       if cell > 50 or cell < 0:
                                           is_safe = False
                                           break
                                   if not is_safe: break
                              
                               if is_safe:
                                   d_to_robot = math.hypot(test_x - robot_x, test_y - robot_y)
                                   if d_to_robot < best_dist_to_robot:
                                       best_dist_to_robot = d_to_robot
                                       final_goal_x, final_goal_y = test_x, test_y
                                       valid_goal_found = True
                           else:
                               ratio = (dist_to_target - STOP_DISTANCE) / dist_to_target
                               final_goal_x = robot_x + dx * ratio
                               final_goal_y = robot_y + dy * ratio
                               valid_goal_found = True
                               break


               if not valid_goal_found:
                   rospy.logwarn("[MirNav] Không tìm thấy chỗ đứng an toàn quanh người (360°).")
                   self.robot_state = "IDLE"
                   self.locked_target_id = None
                   return
              
               rospy.loginfo(f"[MirNav] ✅ Tìm thấy điểm đỗ 360° cách người {math.hypot(final_goal_x - target_x, final_goal_y - target_y):.2f}m, cách robot {best_dist_to_robot:.2f}m")
              
               # Hướng quay: luôn nhìn về phía người
               look_dx = target_x - final_goal_x
               look_dy = target_y - final_goal_y
               final_yaw = math.atan2(look_dy, look_dx)
                  
               goal_pose = (final_goal_x, final_goal_y)
               signal_bus.status_update.emit(f"Chỉ định Đích: Navigation ({final_goal_x:.1f}, {final_goal_y:.1f})")
              
               import tf.transformations
               q = tf.transformations.quaternion_from_euler(0, 0, final_yaw)
               diem_dong = {
                   "x": final_goal_x,
                   "y": final_goal_y,
                   "qz": q[2],
                   "qw": q[3],
                   "arrive_dist": 0.15
               }
              
               rest_ok = False
               if self.nav.mir_headers:
                   rest_ok = nav.api_navigate(self.nav.mir_headers, diem_dong, "diem_dong")
              
               if rest_ok:
                   signal_bus.status_update.emit("Robot đang di chuyển theo lộ trình Planner!")
                  
                   # KHÔNG gọi api_set_state(3) ở đây nữa!
                   # api_navigate đã chờ 3s và robot ĐÃ đang Executing rồi.
                   # Gọi thêm sẽ gây nhiễu khiến MiR dừng lại đánh giá lại.
                  
                   start_wait = time.time()
                   stuck_count = 0
                   has_started = False
                   last_state_id = -1
                   resume_cooldown_until = 0
                   requeue_count = 0
                  
                   while self.robot_state == "MOVING":
                       time.sleep(0.5)
                       if time.time() - start_wait > 180:
                           rospy.logwarn("[MirNav] Quá thời gian di chuyển (180s), tự động hủy.")
                           break
                      
                       # Kiểm tra khoảng cách thực tế đến đích
                       current_dist = float('inf')
                       if robot_pose is not None:
                           current_dist = math.hypot(robot_pose[0] - final_goal_x, robot_pose[1] - final_goal_y)
                           if current_dist < 0.4:
                               rospy.loginfo("[MirNav] ✅ Đã đến đích thành công (khoảng cách < 0.4m)!")
                               signal_bus.status_update.emit("Đã đến vị trí khách. Đang chờ gọi món...")
                               break
                          
                       if self.nav.mir_headers:
                           st = nav.api_status(self.nav.mir_headers)
                           if st:
                               state_id = st.get("state_id")
                              
                               if state_id != last_state_id:
                                   rospy.loginfo(f"[MirNav] Trạng thái MiR: {state_id} ({st.get('state_text', '')}) | Cách đích: {current_dist:.2f}m")
                                   last_state_id = state_id
                              
                               if state_id == 5: # Executing - Robot đang chạy bình thường
                                   has_started = True
                                   stuck_count = 0
                                  
                               elif state_id == 3: # Ready
                                   if not has_started:
                                       # Robot chưa bắt đầu chạy → ấn Play để khởi động
                                       nav.api_set_state(self.nav.mir_headers, 3)
                                       continue
                                      
                                   if time.time() < resume_cooldown_until:
                                       # Đang trong cooldown sau resume → State 3 chỉ là chuyển tiếp
                                       continue
                                  
                                   if current_dist < 0.6:
                                       # Gần đích → Đã hoàn thành thật sự!
                                       rospy.loginfo("[MirNav] ✅ Robot Ready + gần đích → Hoàn thành!")
                                       signal_bus.status_update.emit("Đã đến vị trí khách. Đang chờ gọi món...")
                                       break
                                   elif current_dist < 2.0:
                                       # GẦN ĐÍCH nhưng bị kẹt vùng tím → Chuyển sang CMD_VEL lách vào!
                                       rospy.logwarn(f"[MirNav] ⚠ Planner bỏ cuộc (còn {current_dist:.1f}m). Chuyển CMD_VEL lách vào!")
                                       signal_bus.status_update.emit("Planner kẹt vùng cấm. Đang lách vào bằng CMD_VEL...")
                                       try: requests.delete(f"http://{self.nav.ip}/api/v2.0.0/mission_queue", headers=self.nav.mir_headers, timeout=2)
                                       except: pass
                                       self.nav.send_goal_cmd_vel(final_goal_x, final_goal_y)
                                       # Chờ cmd_vel hoàn thành
                                       while self.nav.is_navigating and self.robot_state == "MOVING":
                                           time.sleep(0.5)
                                           if robot_pose is not None:
                                               d = math.hypot(robot_pose[0] - final_goal_x, robot_pose[1] - final_goal_y)
                                               if d < 0.3:
                                                   break
                                       signal_bus.status_update.emit("Đã đến vị trí khách. Đang chờ gọi món...")
                                       break
                                   else:
                                       # Còn xa đích → gửi lại mission
                                       requeue_count += 1
                                       if requeue_count > 3:
                                           # Gửi lại 3 lần không được → CMD_VEL toàn tuyến
                                           rospy.logwarn("[MirNav] ⚠ Planner thất bại. Chuyển CMD_VEL toàn tuyến!")
                                           try: requests.delete(f"http://{self.nav.ip}/api/v2.0.0/mission_queue", headers=self.nav.mir_headers, timeout=2)
                                           except: pass
                                           self.nav.send_goal_cmd_vel(final_goal_x, final_goal_y)
                                           while self.nav.is_navigating and self.robot_state == "MOVING":
                                               time.sleep(0.5)
                                               if robot_pose is not None:
                                                   d = math.hypot(robot_pose[0] - final_goal_x, robot_pose[1] - final_goal_y)
                                                   if d < 0.3:
                                                       break
                                           signal_bus.status_update.emit("Đã đến vị trí khách. Đang chờ gọi món...")
                                           break
                                       rospy.logwarn(f"[MirNav] ⚠ Robot dừng giữa đường (còn {current_dist:.1f}m)! Gửi lại mission lần {requeue_count}...")
                                       # Gửi lại mission
                                       nav.api_navigate(self.nav.mir_headers, diem_dong, "diem_dong")
                                       resume_cooldown_until = time.time() + 5.0
                                       has_started = False
                                      
                               elif state_id in (10, 12): # Error / Obstacle Pause
                                   stuck_count += 1
                                  
                                   # Nếu gần đích và bị kẹt liên tục → chuyển CMD_VEL
                                   if stuck_count > 6 and current_dist < 2.0:
                                       rospy.logwarn(f"[MirNav] ⚠ Kẹt vùng cấm gần đích. Chuyển CMD_VEL!")
                                       signal_bus.status_update.emit("Kẹt vùng cấm. Đang lách vào bằng CMD_VEL...")
                                       try: requests.delete(f"http://{self.nav.ip}/api/v2.0.0/mission_queue", headers=self.nav.mir_headers, timeout=2)
                                       except: pass
                                       self.nav.send_goal_cmd_vel(final_goal_x, final_goal_y)
                                       while self.nav.is_navigating and self.robot_state == "MOVING":
                                           time.sleep(0.5)
                                           if robot_pose is not None:
                                               d = math.hypot(robot_pose[0] - final_goal_x, robot_pose[1] - final_goal_y)
                                               if d < 0.3:
                                                   break
                                       signal_bus.status_update.emit("Đã đến vị trí khách. Đang chờ gọi món...")
                                       break
                                      
                                   if stuck_count > 12:
                                       rospy.logwarn("[MirNav] ❌ Robot bị kẹt vật cản quá nhiều lần. HỦY LỆNH!")
                                       signal_bus.status_update.emit("Đường đi bị bịt kín. Đã hủy lệnh!")
                                       try: requests.delete(f"http://{self.nav.ip}/api/v2.0.0/mission_queue", headers=self.nav.mir_headers, timeout=2)
                                       except: pass
                                       break
                                      
                                   if state_id == 10:
                                       rospy.logwarn(f"[MirNav] ⚠ Laser phòng thủ. Ấn Play...")
                                       nav.api_set_state(self.nav.mir_headers, 3)
                                       resume_cooldown_until = time.time() + 3.0
                                   else:
                                       rospy.logwarn(f"[MirNav] ⚠ Lỗi kẹt đường (Lần {stuck_count}/12). Xóa lỗi...")
                                       try:
                                           requests.put(f"http://{self.nav.ip}/api/v2.0.0/status", headers=self.nav.mir_headers, json={"clear_error": True}, timeout=2)
                                       except: pass
                                       time.sleep(0.3)
                                       nav.api_set_state(self.nav.mir_headers, 3)
                                       resume_cooldown_until = time.time() + 3.0
                                  
                               elif state_id == 4: # Manual Pause
                                   rospy.logwarn("[MirNav] Robot bị TẠM DỪNG! Đang ép chạy lại...")
                                   nav.api_set_state(self.nav.mir_headers, 3)
                                   resume_cooldown_until = time.time() + 3.0
                              
                   # Tự động mở khóa mục tiêu sau khi đến nơi để nhận lệnh mới
                   self.robot_state = "IDLE"
                   self.locked_target_id = None
                   goal_pose = None
                   robot_planned_path = []
                  
                   # QUAN TRỌNG: Phải xóa bộ đếm thời gian giơ tay.
                   # Nếu không, khung hình tiếp theo AI thấy tay khách vẫn đang giơ sẽ lập tức khóa lại (vì thời gian đã > 2.5s)
                   self.raised_hand_trackers.clear()
               else:
                   # Fallback: thử gửi qua daemon move_base
                   if self.nav.send_goal(final_goal_x, final_goal_y, final_yaw):
                       signal_bus.status_update.emit("Robot đang di chuyển (MoveBase daemon)!")
                   else:
                       rospy.logerr("[MirNav] Cả REST API và Daemon đều thất bại!")
                       signal_bus.status_update.emit("❌ Lỗi: Không gửi được lệnh di chuyển!")


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



