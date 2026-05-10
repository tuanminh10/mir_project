#!/usr/bin/env python3

import cv2
import sys
import numpy as np
import torch
import pyrealsense2 as rs
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Path
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import pygame
import time
import mediapipe as mp
from collections import deque
import tf
from tf.transformations import euler_from_quaternion
import matplotlib.pyplot as plt
import threading
import queue
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped

# PyQt5 imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QSizePolicy, QStatusBar, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
from PyQt5.QtCore import Qt, QTimer, QPointF
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# --- Constants ---
DESIRED_DISTANCE = 1.4  # meters
DISTANCE_TOLERANCE = 0.03  # ±3cm tolerance
MIN_DISTANCE = 0.5  # minimum safe distance
MAX_SPEED = 1.8
ANGULAR_SPEED = 1.5
DISTANCE_SMOOTHING = 5  # number of samples for moving average
GESTURE_CONFIRM_FRAMES_STOP = 5  # frames for stop gesture
GESTURE_CONFIRM_FRAMES_FOLLOW = 10  # frames for follow gesture
GESTURE_CONFIRM_FRAMES_PARKING = 20  # frames for parking gesture
FACE_DISTANCE_THRESHOLD = 0.4
DELAY_TIME = 3  # seconds before alert
GOAL_DISTANCE_THRESHOLD = 2.5  # meters for gesture-based goal

# --- ROS Setup ---

# ============================================================================
# IP is now provided via environment variable ROBOT_IP (set by run_mir.sh)
# No need for dialog in Python code
import os


# Robot IP already set by run_mir.sh via environment variable
print("=" * 70)
print("🤖 MiR ROBOT PERSON FOLLOWER - STARTING")
print("=" * 70)

# Get robot IP from environment (set by run_mir.sh)
robot_ip = os.environ.get('ROBOT_IP', '192.168.0.174')

# Create QApplication for GUI
app = QApplication(sys.argv)
app.setApplicationName("MiR Robot Person Follower")

print(f"📡 Robot IP: {robot_ip}")
print(f"🔗 ROS Master: http://localhost:11311 (container)")
print("=" * 70)

# ============================================================================
# CONTINUE WITH ORIGINAL ROS INIT BELOW
# ============================================================================

rospy.init_node('mir_follower')
cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

# --- Model Initialization ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
yolo_model = YOLO("/app/assets/models/yolo11s.pt").to(device)

# Initialize FaceAnalysis
face_app = FaceAnalysis(name="buffalo_s", allowed_modules=['detection', 'recognition'])
face_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))

# --- MediaPipe Hands ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- RealSense Setup ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
color_sensor = profile.get_device().first_color_sensor()
color_sensor.set_option(rs.option.enable_auto_exposure, 1)  # Bật auto-exposure
color_sensor.set_option(rs.option.enable_auto_white_balance, 1)  # Bật auto-white balance
align = rs.align(rs.stream.color)  # Align depth and color streams

# --- Global Variables ---
person_list = []
target_id = None
target_encodings = []  # List to store three face embeddings
encoding_step = 0  # 0: initial, 1: left, 2: right, 3: done
is_tracking = False
is_recognizing = False
lost_time = None
current_frame = None
distance_history = deque(maxlen=DISTANCE_SMOOTHING)
gesture_command = None
gesture_frames = 0
current_gesture = None
robot_pose = None  # (x, y, theta)
person_map_pos = None
map_data = None
map_resolution = 0.05
map_origin_x = 0
map_origin_y = 0

# --- Sound Setup ---
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("/app/assets/alert.mp3")
alert_sound.set_volume(0.8)
alert2_sound = pygame.mixer.Sound("/app/assets/alert2.mp3")
alert2_sound.set_volume(0.8)
is_sound_playing = False
is_alert2_playing = False

# --- TF Listener ---
tf_listener = tf.TransformListener()

# Queue for thread-safe communication
map_queue = queue.Queue()

class MapCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#e0e0e0')  # Màu nền xám
        self.ax.set_facecolor('#e0e0e0')
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

        # Variables for zoom and pan
        self.current_xlim = None
        self.current_ylim = None
        self.panning = False
        self.pan_start = None
        self.app = parent

        # Connect events
        self.mpl_connect('scroll_event', self.on_scroll)
        self.mpl_connect('button_press_event', self.on_button_press)
        self.mpl_connect('motion_notify_event', self.on_motion)
        self.mpl_connect('button_release_event', self.on_button_release)

    def draw_map(self):
        global map_data, map_resolution, map_origin_x, map_origin_y, robot_pose, person_map_pos
        if map_data is None or map_resolution == 0:
            return
        try:
            self.ax.clear()
            self.ax.set_aspect('equal')
            self.ax.imshow(map_data, cmap='gray', origin='lower', extent=[0, map_data.shape[1], 0, map_data.shape[0]])

            # Draw robot
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
                self.ax.arrow(map_x, map_y, dx, dy, head_width=robot_width_px * 0.4, head_length=robot_length_px * 0.3,
                              fc='red', ec='red')

            # Draw person
            if person_map_pos and is_tracking and not self.app.is_moving_to_goal:
                px = (person_map_pos[0] - map_origin_x) / map_resolution
                py = (person_map_pos[1] - map_origin_y) / map_resolution
                self.ax.plot(px, py, 'go', markersize=10)

                if robot_pose:
                    rx = (robot_pose[0] - map_origin_x) / map_resolution
                    ry = (robot_pose[1] - map_origin_y) / map_resolution
                    self.ax.plot([rx, px], [ry, py], 'g--', linewidth=1)

            # Draw goal
            if self.app.goal_position and self.app.is_moving_to_goal:
                gx = (self.app.goal_position[0] - map_origin_x) / map_resolution
                gy = (self.app.goal_position[1] - map_origin_y) / map_resolution
                self.ax.plot(gx, gy, 'ro', markersize=10)

            # Draw path
            if self.app.path_points and self.app.is_moving_to_goal:
                path_x = [(x - map_origin_x) / map_resolution for x, y in self.app.path_points]
                path_y = [(y - map_origin_y) / map_resolution for x, y in self.app.path_points]
                self.ax.plot(path_x, path_y, 'b-', linewidth=1)

            # Draw parking spot
            if self.app.parking_position:
                grid_x = (self.app.parking_position[0] - map_origin_x) / map_resolution
                grid_y = (self.app.parking_position[1] - map_origin_y) / map_resolution
                self.ax.plot(grid_x, grid_y, 'yo', markersize=10)
                if self.app.parking_orientation:
                    quaternion = (0, 0, self.app.parking_orientation[0], self.app.parking_orientation[1])
                    euler = euler_from_quaternion(quaternion)
                    yaw = euler[2]
                    arrow_length = 10
                    dx = arrow_length * np.cos(yaw)
                    dy = arrow_length * np.sin(yaw)
                    self.ax.arrow(grid_x, grid_y, dx, dy, head_width=5, head_length=5, fc='yellow', ec='yellow')

            # Draw rotation target
            if self.app.rotation_target:
                rx = (self.app.rotation_target[0] - map_origin_x) / map_resolution
                ry = (self.app.rotation_target[1] - map_origin_y) / map_resolution
                self.ax.plot(rx, ry, 'r*', markersize=15)
                if robot_pose:
                    rxx = (robot_pose[0] - map_origin_x) / map_resolution
                    ryy = (robot_pose[1] - map_origin_y) / map_resolution
                    self.ax.plot([rxx, rx], [ryy, ry], 'r-', linewidth=2)
                    # Draw arrow
                    angle = np.arctan2(ry - ryy, rx - rxx)
                    arrow_length = 20
                    dx = arrow_length * np.cos(angle)
                    dy = arrow_length * np.sin(angle)
                    self.ax.arrow(rxx, ryy, dx, dy, head_width=10, head_length=10, fc='red', ec='red')

            # Zoom and pan logic
            if gesture_command == "stop":
                if self.current_xlim is None or self.current_ylim is None:
                    self.current_xlim = [0, map_data.shape[1]]
                    self.current_ylim = [0, map_data.shape[0]]
            else:
                if gesture_command in ["follow", "parking"] and robot_pose:
                    robot_map_x = (robot_pose[0] - map_origin_x) / map_resolution
                    robot_map_y = (robot_pose[1] - map_origin_y) / map_resolution
                    w = map_data.shape[1]
                    h = map_data.shape[0]
                    zoom_factor = 2
                    view_w = w / zoom_factor
                    view_h = h / zoom_factor
                    x_min = robot_map_x - view_w / 2
                    x_max = robot_map_x + view_w / 2
                    y_min = robot_map_y - view_h / 2
                    y_max = robot_map_y + view_h / 2

                    if x_min < 0:
                        x_max -= x_min
                        x_min = 0
                    if x_max > w:
                        x_min -= (x_max - w)
                        x_max = w
                    if y_min < 0:
                        y_max -= y_min
                        y_min = 0
                    if y_max > h:
                        y_min -= (y_max - h)
                        y_max = h

                    self.current_xlim = [x_min, x_max]
                    self.current_ylim = [y_min, y_max]
                else:
                    self.current_xlim = [0, map_data.shape[1]]
                    self.current_ylim = [0, map_data.shape[0]]

            self.ax.set_xlim(self.current_xlim)
            self.ax.set_ylim(self.current_ylim)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.draw()

        except Exception as e:
            rospy.logerr(f"Map drawing error: {str(e)}")

    def on_scroll(self, event):
        global gesture_command
        if gesture_command == "stop":
            if event.inaxes != self.ax:
                return
            x_mouse = event.xdata
            y_mouse = event.ydata
            if x_mouse is None or y_mouse is None:
                return

            if event.button == 'up':
                zoom_factor = 1.2
            else:
                zoom_factor = 0.8

            x_min, x_max = self.current_xlim
            y_min, y_max = self.current_ylim
            w = x_max - x_min
            h = y_max - y_min

            r_x = (x_mouse - x_min) / w
            r_y = (y_mouse - y_min) / h

            original_w = map_data.shape[1]
            original_h = map_data.shape[0]

            new_w = w / zoom_factor
            new_h = h / zoom_factor

            if new_w < original_w / 5:
                new_w = original_w / 5
                new_h = original_h / 5
            elif new_w > original_w:
                new_w = original_w
                new_h = original_h

            x_min_new = x_mouse - r_x * new_w
            x_max_new = x_min_new + new_w
            y_min_new = y_mouse - r_y * new_h
            y_max_new = y_min_new + new_h

            if x_min_new < 0:
                x_max_new -= x_min_new
                x_min_new = 0
            if x_max_new > original_w:
                x_min_new -= (x_max_new - original_w)
                x_max_new = original_w
            if y_min_new < 0:
                y_max_new -= y_min_new
                y_min_new = 0
            if y_max_new > original_h:
                y_min_new -= (y_max_new - original_h)
                y_max_new = original_h

            self.current_xlim = [x_min_new, x_max_new]
            self.current_ylim = [y_min_new, y_max_new]
            self.draw_map()

    def on_button_press(self, event):
        global gesture_command, target_id
        if gesture_command == "stop" and event.button == 1 and event.inaxes == self.ax:
            self.panning = True
            self.pan_start = (event.xdata, event.ydata)
            # Handle parking mode clicks
            if self.app.parking_mode:
                self.app.handle_parking_click(event.xdata, event.ydata)
            # Handle rotation target clicks
        if self.app.rotation_mode and not target_id:
            self.app.handle_rotation_click(event.xdata, event.ydata)

    def on_motion(self, event):
        global gesture_command
        if self.panning and gesture_command == "stop" and event.inaxes == self.ax:
            dx = event.xdata - self.pan_start[0]
            dy = event.ydata - self.pan_start[1]
            x_min, x_max = self.current_xlim
            y_min, y_max = self.current_ylim

            new_x_min = x_min - dx
            new_x_max = x_max - dx
            new_y_min = y_min - dy
            new_y_max = y_max - dy

            original_w = map_data.shape[1]
            original_h = map_data.shape[0]

            if new_x_min < 0:
                new_x_max = new_x_max - new_x_min
                new_x_min = 0
            if new_x_max > original_w:
                new_x_min = new_x_min - (new_x_max - original_w)
                new_x_max = original_w
            if new_y_min < 0:
                new_y_max = new_y_max - new_y_min
                new_y_min = 0
            if new_y_max > original_h:
                new_y_min = new_y_min - (new_y_max - original_h)
                new_y_max = original_h

            self.current_xlim = [new_x_min, new_x_max]
            self.current_ylim = [new_y_min, new_y_max]
            self.pan_start = (event.xdata, event.ydata)
            self.draw_map()

    def on_button_release(self, event):
        if event.button == 1:  # Đổi từ chuột phải (3) sang chuột trái (1)
            self.panning = False
            self.pan_start = None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MiR100 Person Follower with Navigation")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize navigation variables
        self.goal_position = None
        self.is_moving_to_goal = False
        self.path_points = []
        self.last_person_goal = None
        self.parking_mode = False
        self.parking_click_count = 0
        self.parking_position = None
        self.parking_orientation = None
        self.is_registering_face = False  # Biến mới để theo dõi trạng thái đăng ký khuôn mặt
        self.is_idle = True  # Biến mới để theo dõi trạng thái chờ đăng ký khuôn mặt

        # Rotation variables
        self.rotation_mode = False
        self.rotation_target = None
        self.is_rotating = False
        self.target_yaw = 0.0

        # Create main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Left panel (camera and controls)
        self.left_panel = QFrame()
        self.left_panel.setFrameShape(QFrame.StyledPanel)
        self.left_panel.setMinimumWidth(600)
        self.left_panel.setStyleSheet("background-color: #A9A9A9;")
        self.main_layout.addWidget(self.left_panel)
        self.left_layout = QVBoxLayout(self.left_panel)

        # Camera display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: #66CDAA;")
        self.left_layout.addWidget(self.video_label)

        # Status labels
        self.status_label = QLabel("Nhìn thẳng vào camera và nhấp chuột vào người")
        self.status_label.setStyleSheet("font-size: 12pt; color: white;")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.left_layout.addWidget(self.status_label)
        self.face_status_label = QLabel("Trạng thái nhận diện: Chưa đăng ký khuôn mặt")
        self.face_status_label.setStyleSheet("font-size: 11pt; color: white;")
        self.face_status_label.setAlignment(Qt.AlignCenter)
        self.left_layout.addWidget(self.face_status_label)
        self.distance_label = QLabel("Khoảng cách: --.-- m")
        self.distance_label.setStyleSheet("font-size: 12pt; font-weight: bold; color: white;")
        self.distance_label.setAlignment(Qt.AlignCenter)
        self.left_layout.addWidget(self.distance_label)
        self.gesture_label = QLabel("Cử chỉ: --")
        self.gesture_label.setStyleSheet("font-size: 11pt; color: white;")
        self.gesture_label.setAlignment(Qt.AlignCenter)
        self.left_layout.addWidget(self.gesture_label)

        # Nút "Hủy theo dõi"
        self.cancel_tracking_btn = QPushButton("Hủy theo dõi")
        self.cancel_tracking_btn.setStyleSheet(
            "QPushButton {"
            "   background-color: #e74c3c;"
            "   color: white;"
            "   border: none;"
            "   padding: 8px;"
            "   font-weight: bold;"
            "   border-radius: 4px;"
            "}"
            "QPushButton:hover {"
            "   background-color: #c0392b;"
            "}"
        )
        self.cancel_tracking_btn.clicked.connect(self.cancel_tracking)
        self.left_layout.addWidget(self.cancel_tracking_btn)

        # Right panel (map)
        self.right_panel = QFrame()
        self.right_panel.setFrameShape(QFrame.StyledPanel)
        self.main_layout.addWidget(self.right_panel)
        self.right_layout = QVBoxLayout(self.right_panel)

        # Map canvas
        self.map_canvas = MapCanvas(self)
        self.right_layout.addWidget(self.map_canvas)

        # Rotation status label
        self.rotation_status_label = QLabel("Chưa chọn hướng quay")
        self.rotation_status_label.setStyleSheet("font-size: 13pt; color: purple;")
        self.rotation_status_label.setAlignment(Qt.AlignCenter)
        self.right_layout.addWidget(self.rotation_status_label)

        # Rotation button
        self.rotate_btn = QPushButton("Quay robot")
        self.rotate_btn.setStyleSheet(
            "QPushButton {"
            "   background-color: #9b59b6;"
            "   color: white;"
            "   border: none;"
            "   padding: 8px;"
            "   font-weight: bold;"
            "   border-radius: 4px;"
            "}"
            "QPushButton:hover {"
            "   background-color: #8e44ad;"
            "}"
            "QPushButton:disabled {"
            "   background-color: #bdc3c7;"
            "}"
        )
        self.rotate_btn.clicked.connect(self.activate_rotation_mode)
        self.right_layout.addWidget(self.rotate_btn)

        # Parking status label
        self.parking_status_label = QLabel("Chưa có vị trí đỗ xe")
        self.parking_status_label.setStyleSheet("font-size: 13pt; color: red;")
        self.parking_status_label.setAlignment(Qt.AlignCenter)
        self.right_layout.addWidget(self.parking_status_label)

        # Parking button
        self.parking_btn = QPushButton("Parking")
        self.parking_btn.setStyleSheet(
            "QPushButton {"
            "   background-color: #3498db;"
            "   color: white;"
            "   border: none;"
            "   padding: 8px;"
            "   font-weight: bold;"
            "   border-radius: 4px;"
            "}"
            "QPushButton:hover {"
            "   background-color: #2980b9;"
            "}"
        )
        self.parking_btn.clicked.connect(self.start_parking_mode)
        self.right_layout.addWidget(self.parking_btn)

        # Set up timers
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(30)  # ~30 FPS
        self.position_timer = QTimer()
        self.position_timer.timeout.connect(self.update_position_continuously)
        self.position_timer.start(500)
        self.map_timer = QTimer()
        self.map_timer.timeout.connect(self.check_map_queue)
        self.map_timer.start(100)
        self.rotation_timer = QTimer()
        self.rotation_timer.timeout.connect(self.update_rotation)
        self.rotation_timer.setInterval(100)  # 10 Hz

        # ROS Subscribers
        rospy.Subscriber('/map', OccupancyGrid, self.map_callback_threaded)
        rospy.Subscriber('/move_base_node/SBPLLatticePlanner/plan', Path, self.path_callback)

        # Connect video label click event
        self.video_label.mousePressEvent = self.on_video_click

        # Start ROS spin thread
        self.ros_thread = threading.Thread(target=rospy.spin, daemon=True)
        self.ros_thread.start()

    def activate_rotation_mode(self):
        """Kích hoạt chế độ chọn hướng quay cho robot"""
        if is_tracking:
            QMessageBox.warning(self, "Lỗi", "Không thể quay robot khi đang theo dõi người!")
            return
        if self.is_moving_to_goal:
            QMessageBox.warning(self, "Lỗi", "Không thể quay robot khi đang di chuyển đến mục tiêu!")
            return
        if self.is_rotating:
            QMessageBox.warning(self, "Lỗi", "Robot đang trong quá trình quay!")
            return

        self.rotation_mode = True
        self.rotation_status_label.setText("Vui lòng chọn hướng quay trên bản đồ")
        self.rotation_status_label.setStyleSheet("font-size: 13pt; color: purple;")

    def handle_rotation_click(self, x, y):
        """Xử lý sự kiện nhấp chuột để chọn hướng quay"""
        if x is not None and y is not None:
            # Chuyển đổi tọa độ pixel sang tọa độ bản đồ
            x_map = x * map_resolution + map_origin_x
            y_map = y * map_resolution + map_origin_y
            self.rotation_target = (x_map, y_map)
            self.rotation_mode = False
            self.start_rotation()

    def start_rotation(self):
        """Bắt đầu quá trình quay robot sau 3 giây trễ"""
        if robot_pose is None or self.rotation_target is None:
            return

        # Tính góc quay mục tiêu
        rx, ry, current_yaw = robot_pose
        tx, ty = self.rotation_target

        # Tính vector từ robot đến mục tiêu
        dx = tx - rx
        dy = ty - ry

        # Tính góc quay mục tiêu (radians)
        self.target_yaw = np.arctan2(dy, dx)

        # Hiển thị thông báo chuẩn bị quay
        self.rotation_status_label.setText("Chuẩn bị quay robot sau 5 giây...")
        self.rotation_status_label.setStyleSheet("font-size: 12pt; color: purple;")

        # Trì hoãn 5 giây trước khi quay
        QTimer.singleShot(5000, self.execute_rotation)

    def execute_rotation(self):
        """Thực hiện quá trình quay robot"""
        if robot_pose is None or self.rotation_target is None:
            return

        # Bắt đầu quay
        self.is_rotating = True
        self.rotation_status_label.setText("Đang quay robot")
        self.rotation_status_label.setStyleSheet("font-size: 12pt; color: purple;")
        self.rotation_timer.start()

    def update_rotation(self):
        """Cập nhật quá trình quay robot với tốc độ bằng một nửa tối đa"""
        if not self.is_rotating or robot_pose is None:
            self.rotation_timer.stop()
            self.rotation_status_label.setText("Chưa chọn hướng quay")
            self.rotation_status_label.setStyleSheet("font-size: 13pt; color: purple;")
            return

        rx, ry, current_yaw = robot_pose

        # Tính sai số góc
        error = self.target_yaw - current_yaw

        # Đưa sai số về khoảng [-pi, pi]
        while error > np.pi:
            error -= 2 * np.pi
        while error < -np.pi:
            error += 2 * np.pi

        # Ngưỡng dừng (5 độ)
        if abs(error) < np.radians(5):
            self.send_velocity_command(0, 0)
            self.is_rotating = False
            self.rotation_target = None
            self.rotation_timer.stop()
            self.rotation_status_label.setText("Hoàn tất quay")
            self.rotation_status_label.setStyleSheet("font-size: 13pt; color: green;")
            QTimer.singleShot(2000, lambda: self.rotation_status_label.setText("Chưa chọn hướng quay"))
            QTimer.singleShot(2000, lambda: self.rotation_status_label.setStyleSheet("font-size: 13pt; color: purple;"))
            return

        # Điều khiển P
        Kp = 0.75
        angular_speed = Kp * error

        # Giới hạn tốc độ góc bằng một nửa tốc độ tối đa
        max_angular_speed = ANGULAR_SPEED / 2  # Giảm tốc độ quay xuống một nửa
        angular_speed = np.clip(angular_speed, -max_angular_speed, max_angular_speed)

        # Gửi lệnh quay
        self.send_velocity_command(0, angular_speed)

    def map_callback_threaded(self, msg):
        try:
            width = msg.info.width
            height = msg.info.height
            resolution = msg.info.resolution
            origin_x = msg.info.origin.position.x
            origin_y = msg.info.origin.position.y
            data = np.array(msg.data).reshape((height, width))
            map_queue.put((data, resolution, origin_x, origin_y))
        except Exception as e:
            rospy.logerr(f"Map processing error: {str(e)}")

    def check_map_queue(self):
        try:
            while not map_queue.empty():
                data, resolution, origin_x, origin_y = map_queue.get_nowait()
                self.process_map_data(data, resolution, origin_x, origin_y)
        except queue.Empty:
            pass

    def process_map_data(self, data, resolution, origin_x, origin_y):
        global map_data, map_resolution, map_origin_x, map_origin_y
        map_data = data
        map_resolution = resolution
        map_origin_x = origin_x
        map_origin_y = origin_y
        self.map_canvas.draw_map()

    def start_parking_mode(self):
        self.parking_mode = True
        self.parking_click_count = 0
        self.parking_status_label.setText("Vui lòng chọn vị trí đỗ xe")
        self.parking_status_label.setStyleSheet("font-size: 13pt; color: red;")

    def handle_parking_click(self, x, y):
        if self.parking_mode and x is not None and y is not None:
            if self.parking_click_count == 0:
                x_parking = x * map_resolution + map_origin_x
                y_parking = y * map_resolution + map_origin_y
                self.parking_position = (x_parking, y_parking)
                self.parking_click_count = 1
                self.parking_status_label.setText("Đã xác nhận vị trí đỗ xe, vui lòng chọn hướng đỗ xe")
                self.parking_status_label.setStyleSheet("font-size: 13pt; color: red;")
            elif self.parking_click_count == 1:
                direction_x = x * map_resolution + map_origin_x
                direction_y = y * map_resolution + map_origin_y
                z_parking, w_parking = self.calculate_orientation(self.parking_position, (direction_x, direction_y))
                self.parking_orientation = (z_parking, w_parking)
                self.parking_mode = False
                self.parking_click_count = 0
                self.parking_status_label.setText("Đã xác nhận vị trí và hướng đỗ xe")
                self.parking_status_label.setStyleSheet("font-size: 13pt; color: green;")
                self.map_canvas.draw_map()

    def calculate_orientation(self, start, end):
        orientation = np.arctan2(end[1] - start[1], end[0] - start[0])
        z = np.sin(orientation / 2)
        w = np.cos(orientation / 2)
        return z, w

    def path_callback(self, path_msg):
        if self.is_moving_to_goal:
            try:
                self.path_points = [(pose.pose.position.x, pose.pose.position.y) for pose in path_msg.poses]
                self.map_canvas.draw_map()
            except Exception as e:
                rospy.logerr(f"Error in path_callback: {str(e)}")

    def send_velocity_command(self, linear_x=0, angular_z=0):
        cmd = Twist()
        cmd.linear.x = np.clip(linear_x, -MAX_SPEED, MAX_SPEED)
        cmd.angular.z = np.clip(angular_z, -ANGULAR_SPEED, ANGULAR_SPEED)
        cmd_vel_pub.publish(cmd)

    def get_smoothed_distance(self, depth_frame, box):
        x1, y1, x2, y2 = box
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        roi_size = min(x2 - x1, y2 - y1) // 4
        distances = []
        for dx in range(-roi_size, roi_size, 5):
            for dy in range(-roi_size, roi_size, 5):
                px = center_x + dx
                py = center_y + dy
                if 0 <= px < 640 and 0 <= py < 480:
                    dist = depth_frame.get_distance(px, py)
                    if 0.3 < dist < 6.0:
                        distances.append(dist)
        return np.median(distances) if distances else 0

    def face_recognition_handler(self, frame, person):
        global target_id, is_tracking, is_recognizing, lost_time, is_sound_playing
        x1, y1, x2, y2 = person['box']
        box_x_mid = (x1 + x2) // 2
        face_x_mid = None

        box_height = y2 - y1
        y_mid = y1 + int(box_height * 0.5)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(640, x2)
        y_mid = min(480, y_mid)
        face_region = frame[y1:y_mid, x1:x2]
        if face_region.size == 0:
            return False
        face_region_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
        faces = face_app.get(face_region_rgb)

        if len(faces) >= 5:
            return False
        if len(faces) == 0:
            return False

        max_area_face = None
        max_area = 0
        for face in faces:
            bbox = face.bbox.astype(int)
            top, right, bottom, left = bbox[1], bbox[2], bbox[3], bbox[0]
            face_area = (right - left) * (bottom - top)
            if face_area > max_area:
                max_area = face_area
                max_area_face = face

        if max_area_face is None:
            return False

        face_embedding = max_area_face.embedding
        face_bbox = max_area_face.bbox.astype(int)
        face_x_mid = x1 + (face_bbox[0] + face_bbox[2]) // 2

        X_MID_THRESHOLD = 80
        if face_x_mid is None or abs(face_x_mid - box_x_mid) > X_MID_THRESHOLD:
            return False

        for encoding in target_encodings:
            distance = 1 - np.dot(face_embedding, encoding) / (
                    np.linalg.norm(face_embedding) * np.linalg.norm(encoding))
            if distance < FACE_DISTANCE_THRESHOLD:
                target_id = person['id']
                is_tracking = True
                is_recognizing = False
                lost_time = None
                self.face_status_label.setText("Đã xác nhận danh tính")
                self.face_status_label.setStyleSheet("font-size: 11pt; color: green;")
                if is_sound_playing:
                    alert_sound.stop()
                    is_sound_playing = False
                return True
        return False

    def process_frame(self, color_frame, depth_frame):
        global target_id, person_list, is_tracking, is_recognizing, lost_time, current_frame, is_sound_playing
        color_image = np.asanyarray(color_frame.get_data())
        current_frame = color_image.copy()
        results = yolo_model.track(color_image, persist=True, verbose=False, conf=0.5)
        person_list = []
        for result in results:
            for box in result.boxes:
                if int(box.cls) != 0:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_area = (x2 - x1) * (y2 - y1)
                if box_area < 10000:
                    continue
                person_id = f"ID{int(box.id)}" if box.id is not None else "ID?"
                distance = self.get_smoothed_distance(depth_frame, (x1, y1, x2, y2))
                person_list.append({
                    'id': person_id,
                    'box': (x1, y1, x2, y2),
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                    'distance': distance,
                    'is_target': (person_id == target_id)
                })

        if not self.is_moving_to_goal and not self.is_rotating:
            if is_tracking:
                if target_id not in [p['id'] for p in person_list]:
                    is_tracking = False
                    is_recognizing = True
                    target_id = None
                    lost_time = time.time()
                    self.face_status_label.setText("Mất dấu! Đang nhận diện...")
                    self.face_status_label.setStyleSheet("font-size: 11pt; color: red;")
            elif is_recognizing and target_encodings:
                for person in sorted(person_list, key=lambda x: x['distance']):
                    if person['distance'] < 2.5:
                        if self.face_recognition_handler(color_image, person):
                            break
                if gesture_command == "follow" and lost_time and (
                        time.time() - lost_time) > DELAY_TIME and not is_sound_playing:
                    alert_sound.play(loops=-1)
                    is_sound_playing = True

        if target_id and not self.is_moving_to_goal and not self.is_rotating:
            target_person = next((p for p in person_list if p['id'] == target_id), None)
            if target_person:
                self.detect_gestures(color_image, target_person['box'], depth_frame)
        return color_image

    def detect_gestures(self, image, target_box, depth_frame):
        global gesture_command, gesture_frames, current_gesture
        if self.is_moving_to_goal or self.is_rotating:
            return

        x1, y1, x2, y2 = target_box
        box_height = y2 - y1
        y_top = y1
        y_bottom = y1 + box_height // 2
        roi = image[max(0, y_top):min(480, y_bottom), max(0, x1):min(640, x2)]
        if roi.size == 0:
            return
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = hands.process(roi_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                fingers_up = self.count_fingers(hand_landmarks)

                if fingers_up == 3:
                    gesture = "stop"
                    confirm_frames = GESTURE_CONFIRM_FRAMES_STOP
                elif fingers_up == 2:
                    gesture = "follow"
                    confirm_frames = GESTURE_CONFIRM_FRAMES_FOLLOW
                elif fingers_up == 5:
                    gesture = "parking"
                    confirm_frames = GESTURE_CONFIRM_FRAMES_PARKING
                else:
                    gesture = None
                    confirm_frames = 0

                if gesture:
                    if gesture == current_gesture:
                        gesture_frames += 1
                        if gesture_frames >= confirm_frames:
                            if gesture != "parking":
                                gesture_command = gesture
                            gesture_frames = 0
                            if gesture == "parking" and self.parking_position is not None and self.parking_orientation is not None:
                                target_person = next((p for p in person_list if p['id'] == target_id), None)
                                if target_person and gesture_command == "stop" and target_person['distance'] < 2.0:
                                    gesture_command = "parking"
                                    threading.Thread(target=self.send_goal_parking,
                                                     args=(self.parking_position[0], self.parking_position[1],
                                                           self.parking_orientation[0], self.parking_orientation[1]),
                                                     daemon=True).start()
                            elif gesture == "follow" and not self.is_moving_to_goal:
                                target_person = next((p for p in person_list if p['id'] == target_id), None)
                                if target_person and target_person['distance'] > GOAL_DISTANCE_THRESHOLD:
                                    person_pos = self.get_person_map_position(depth_frame, target_person['box'])
                                    if person_pos:
                                        self.goal_position = person_pos
                                        self.last_person_goal = person_pos
                                        threading.Thread(target=self.send_goal, args=(person_pos[0], person_pos[1]),
                                                         daemon=True).start()
                    else:
                        current_gesture = gesture
                        gesture_frames = 1

    def count_fingers(self, hand_landmarks):
        tip_ids = [4, 8, 12, 16, 20]
        fingers = []
        if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1, 5):
            if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)
        return sum(fingers)

    def control_robot(self):
        global target_id, distance_history, gesture_command
        # Ưu tiên trạng thái đăng ký khuôn mặt
        if self.is_registering_face and encoding_step in [0, 1, 2]:
            return
        if self.is_idle:
            self.send_velocity_command(0, 0)
            return
        if self.is_rotating or self.rotation_mode:
            return

        if target_id and person_list:
            target_person = next((p for p in person_list if p['id'] == target_id), None)
            if target_person:
                current_distance = target_person['distance']
                distance_history.append(current_distance)
                smoothed_distance = np.mean(distance_history) if distance_history else current_distance
                self.distance_label.setText(f"Khoảng cách: {smoothed_distance:.2f}m")
            else:
                self.distance_label.setText("Khoảng cách: --.-- m")
        else:
            self.distance_label.setText("Khoảng cách: --.-- m")

        if self.is_moving_to_goal:
            self.status_label.setText("Đang di chuyển đến mục tiêu")
            self.status_label.setStyleSheet("font-size: 12pt; color: blue;")
            self.gesture_label.setText("Cử chỉ: --")
            self.gesture_label.setStyleSheet("font-size: 11pt; color: white;")
            self.send_velocity_command(0, 0)
            return

        if gesture_command == "stop":
            self.send_velocity_command(0, 0)
            self.status_label.setText("ĐANG TẠM DỪNG")
            self.status_label.setStyleSheet("font-size: 12pt; color: orange;")
            self.gesture_label.setText("Cử chỉ: DỪNG (3 ngón)")
            self.gesture_label.setStyleSheet("font-size: 11pt; color: red;")
            return

        if gesture_command == "follow":
            self.gesture_label.setText("Cử chỉ: THEO DÕI (2 ngón)")
            self.gesture_label.setStyleSheet("font-size: 11pt; color: green;")

        if not target_id or not person_list:
            self.send_velocity_command(0, 0)
            return

        smoothed_distance = np.mean(distance_history) if distance_history else target_person['distance']
        cx, cy = target_person['center']
        center_error = cx - 320
        distance_error = smoothed_distance - DESIRED_DISTANCE

        angular_z = -ANGULAR_SPEED * (center_error / 320)
        linear_x = 0

        if smoothed_distance > DESIRED_DISTANCE + DISTANCE_TOLERANCE:
            speed_factor = min(1.0, (smoothed_distance - DESIRED_DISTANCE) / 0.5)
            linear_x = MAX_SPEED * 0.5 * speed_factor
            self.status_label.setText(f"Đang tiến lại gần")
            self.status_label.setStyleSheet("font-size: 12pt; color: green;")
        elif smoothed_distance < DESIRED_DISTANCE - DISTANCE_TOLERANCE:
            speed_factor = min(1.0, (DESIRED_DISTANCE - smoothed_distance) / 0.5)
            linear_x = -MAX_SPEED * 0.5 * speed_factor
            self.status_label.setText(f"Đang lùi ra xa")
            self.status_label.setStyleSheet("font-size: 12pt; color: red;")
        else:
            self.status_label.setText(f"Giữ khoảng cách tốt")
            self.status_label.setStyleSheet("font-size: 12pt; color: blue;")

        if smoothed_distance < MIN_DISTANCE:
            linear_x = -0.3
            self.status_label.setText(f"QUÁ GẦN! DỪNG LẠI!")
            self.status_label.setStyleSheet("font-size: 12pt; color: red;")

        self.send_velocity_command(linear_x, angular_z)

    def update_gui(self):
        try:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                return
            color_image = self.process_frame(color_frame, depth_frame)

            # Calculate person_map_pos
            global person_map_pos
            if is_tracking and target_id and not self.is_moving_to_goal and not self.is_rotating:
                target_person = next((p for p in person_list if p['id'] == target_id), None)
                if target_person:
                    person_map_pos = self.get_person_map_position(depth_frame, target_person['box'])
                else:
                    person_map_pos = None
            else:
                person_map_pos = None

            # Draw bounding boxes
            if is_tracking and target_id:
                target_person = next((p for p in person_list if p['id'] == target_id), None)
                if target_person:
                    x1, y1, x2, y2 = target_person['box']
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(color_image, "OWNER", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # Convert to QImage and display
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.video_label.setPixmap(pixmap)
            self.control_robot()
        except Exception as e:
            print(f"Error: {e}")

    def on_video_click(self, event):
        global target_id, target_encodings, encoding_step, is_tracking, is_recognizing, gesture_command, target_brightness, target_histogram
        if current_frame is None:
            return
        x = event.pos().x()
        y = event.pos().y()

        # Adjust for label centering
        label_size = self.video_label.size()
        frame_width, frame_height = 640, 480
        offset_x = (label_size.width() - frame_width) // 2
        offset_y = (label_size.height() - frame_height) // 2
        x -= offset_x
        y -= offset_y
        if x < 0 or y < 0 or x >= frame_width or y >= frame_height:
            return

        for person in person_list:
            x1, y1, x2, y2 = person['box']
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.is_registering_face = True  # Bật trạng thái đăng ký khuôn mặt
                self.is_idle = False  # Tắt trạng thái chờ khi bắt đầu đăng ký
                face_region = current_frame[y1:y2, x1:x2]
                face_region_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                faces = face_app.get(face_region_rgb)
                if len(faces) == 0:
                    self.face_status_label.setText("Không tìm thấy khuôn mặt, vui lòng thử lại")
                    self.face_status_label.setStyleSheet("font-size: 11pt; color: red;")
                    self.is_registering_face = False
                    return

                # Chọn khuôn mặt có diện tích lớn nhất
                max_area_face = None
                max_area = 0
                for face in faces:
                    bbox = face.bbox.astype(int)
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if area > max_area:
                        max_area = area
                        max_area_face = face
                if max_area_face is None:
                    self.is_registering_face = False
                    return

                face_embedding = max_area_face.embedding

                if encoding_step == 0:
                    target_encodings.append(face_embedding)
                    encoding_step = 1
                    self.status_label.setText("Quay mặt sang trái (~45°) và nhấp chuột vào người")
                    self.status_label.setStyleSheet("font-size: 12pt; color: yellow;")
                    self.face_status_label.setText("Hoàn thành bước 1/3")
                    self.face_status_label.setStyleSheet("font-size: 11pt; color: yellow;")
                elif encoding_step == 1:
                    target_encodings.append(face_embedding)
                    encoding_step = 2
                    self.status_label.setText("Quay mặt sang phải (~45°) và nhấp chuột vào người")
                    self.status_label.setStyleSheet("font-size: 12pt; color: yellow;")
                    self.face_status_label.setText("Hoàn thành bước 2/3")
                    self.face_status_label.setStyleSheet("font-size: 11pt; color: yellow;")
                elif encoding_step == 2:
                    target_encodings.append(face_embedding)
                    encoding_step = 3
                    target_id = person['id']
                    is_tracking = True
                    self.is_registering_face = False  # Tắt trạng thái đăng ký sau khi hoàn tất
                    self.is_idle = False  # Đảm bảo robot không ở trạng thái chờ sau khi đăng ký
                    gesture_command = "stop"
                    self.face_status_label.setText("Hoàn thành bước 3/3")
                    self.face_status_label.setStyleSheet("font-size: 11pt; color: green;")
                    QTimer.singleShot(2000, lambda: self.face_status_label.setText("Đã đăng ký khuôn mặt"))
                    QTimer.singleShot(2000,
                                      lambda: self.face_status_label.setStyleSheet("font-size: 11pt; color: green;"))
                break

    def cancel_tracking(self):
        global target_id, encoding_step, gesture_command, target_encodings, is_tracking, is_recognizing, is_sound_playing
        target_id = None
        encoding_step = 0
        gesture_command = "stop"
        target_encodings = []
        is_tracking = False
        is_recognizing = False
        self.is_registering_face = False  # Đặt lại trạng thái đăng ký khuôn mặt
        self.is_idle = True  # Đặt robot vào trạng thái chờ
        if is_sound_playing:
            alert_sound.stop()
            is_sound_playing = False
        self.send_velocity_command(0, 0)
        self.status_label.setText("Nhìn thẳng vào camera và nhấp chuột vào người")
        self.status_label.setStyleSheet("font-size: 12pt; color: white;")
        self.face_status_label.setText("Chưa đăng ký khuôn mặt")
        self.face_status_label.setStyleSheet("font-size: 11pt; color: white;")
        self.gesture_label.setText("Cử chỉ: --")
        self.gesture_label.setStyleSheet("font-size: 11pt; color: white;")
        # Reset rotation if active
        if self.is_rotating:
            self.rotation_timer.stop()
            self.is_rotating = False
            self.rotation_target = None
            self.rotation_status_label.setText("Chưa chọn hướng quay")
            self.rotation_status_label.setStyleSheet("font-size: 13pt; color: purple;")

    def update_position_continuously(self):
        global robot_pose
        if not rospy.is_shutdown():
            try:
                (trans, rot) = tf_listener.lookupTransform('/map', '/base_link', rospy.Time(0))
                robot_pose = (trans[0], trans[1], euler_from_quaternion(rot)[2])
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                pass
            self.map_canvas.draw_map()

    def get_person_map_position(self, depth_frame, box):
        global robot_pose
        if robot_pose is None or depth_frame is None:
            return None
        try:
            x1, y1, x2, y2 = box
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            distance = depth_frame.get_distance(center_x, center_y)
            if distance <= 0:
                return None
            fov = 69
            angle = (center_x - 320) * (fov / 640)
            angle_rad = np.radians(angle)
            z_cam = distance * np.cos(angle_rad)
            x_cam = distance * np.sin(angle_rad)
            x_rel = z_cam
            y_rel = -x_cam
            robot_x, robot_y, robot_theta = robot_pose
            map_x = robot_x + x_rel * np.cos(robot_theta) - y_rel * np.sin(robot_theta)
            map_y = robot_y + x_rel * np.sin(robot_theta) + y_rel * np.cos(robot_theta)
            return (map_x, map_y)
        except Exception as e:
            rospy.logerr(f"Person position calculation error: {str(e)}")
            return None

    def send_goal(self, x, y):
        global gesture_command, is_alert2_playing
        self.is_moving_to_goal = True
        while self.is_moving_to_goal and not rospy.is_shutdown():
            try:
                client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
                client.wait_for_server()
                goal = MoveBaseGoal()
                goal.target_pose.header.frame_id = "map"
                goal.target_pose.header.stamp = rospy.Time.now()
                goal.target_pose.pose.position.x = x
                goal.target_pose.pose.position.y = y
                z, w = self.calculate_orientation(robot_pose[:2], (x, y))
                goal.target_pose.pose.orientation.z = z
                goal.target_pose.pose.orientation.w = w
                client.send_goal(goal)
                client.wait_for_result()
                state = client.get_state()
                if state == GoalStatus.SUCCEEDED:
                    rospy.loginfo("Robot reached the goal")
                    self.is_moving_to_goal = False
                    self.goal_position = None
                    self.path_points = []
                    gesture_command = "stop"
                    if is_alert2_playing:
                        alert2_sound.stop()
                        is_alert2_playing = False
                    self.map_canvas.draw_map()
                    break
                else:
                    rospy.logwarn("Failed to reach goal, state: %d. Retrying in 1.5 seconds...", state)
                    if not pygame.mixer.get_busy():
                        alert2_sound.play(loops=0)
                        is_alert2_playing = True
                    rospy.sleep(2.0)
            except Exception as e:
                rospy.logerr(f"Error sending goal: {str(e)}")
                rospy.sleep(1)

    def send_goal_parking(self, x_parking, y_parking, z_parking, w_parking):
        global is_alert2_playing, gesture_command
        self.is_moving_to_goal = True
        self.goal_position = (x_parking, y_parking)
        while self.is_moving_to_goal and not rospy.is_shutdown():
            try:
                client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
                client.wait_for_server()
                goal = MoveBaseGoal()
                goal.target_pose.header.frame_id = "map"
                goal.target_pose.header.stamp = rospy.Time.now()
                goal.target_pose.pose.position.x = x_parking
                goal.target_pose.pose.position.y = y_parking
                goal.target_pose.pose.orientation.z = z_parking
                goal.target_pose.pose.orientation.w = w_parking
                client.send_goal(goal)
                client.wait_for_result()
                state = client.get_state()
                if state == GoalStatus.SUCCEEDED:
                    rospy.loginfo("Robot reached the parking spot")
                    self.is_moving_to_goal = False
                    self.goal_position = None
                    self.path_points = []
                    gesture_command = "stop"
                    if is_alert2_playing:
                        alert2_sound.stop()
                        is_alert2_playing = False
                    self.map_canvas.draw_map()
                    break
                else:
                    rospy.logwarn("Failed to reach parking spot, state: %d. Retrying in 1.5 seconds...", state)
                    if not pygame.mixer.get_busy():
                        alert2_sound.play(loops=0)
                        is_alert2_playing = True
                    rospy.sleep(2.0)
            except Exception as e:
                rospy.logerr(f"Error sending parking goal: {str(e)}")
                rospy.sleep(1)

    def closeEvent(self, event):
        self.send_velocity_command(0, 0)
        pipeline.stop()
        if is_sound_playing:
            alert_sound.stop()
        if is_alert2_playing:
            alert2_sound.stop()
        rospy.signal_shutdown("GUI closed")
        event.accept()


if __name__ == "__main__":
    # app already created in get_robot_ip()
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

