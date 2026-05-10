import cv2
import numpy as np
import matplotlib.pyplot as plt
import rospy

from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import config

class SignalBus(QObject):
    frame_update = pyqtSignal(np.ndarray)
    status_update = pyqtSignal(str)

signal_bus = SignalBus()

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
        
        # State lưu trữ cục bộ thay vì global
        self.map_data = None
        self.map_resolution = 0.05
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0
        self.robot_pose = None
        self.goal_pose = None
        self.robot_planned_path = []

        self.mpl_connect('scroll_event', self.on_scroll)
        self.mpl_connect('button_press_event', self.on_button_press)
        self.mpl_connect('motion_notify_event', self.on_motion)
        self.mpl_connect('button_release_event', self.on_button_release)

    def set_map_data(self, data, resolution, origin_x, origin_y):
        self.map_data = data
        self.map_resolution = resolution
        self.map_origin_x = origin_x
        self.map_origin_y = origin_y
        self.draw_map()

    def set_robot_pose(self, pose):
        self.robot_pose = pose
        self.draw_map()

    def set_goal_and_path(self, goal, path):
        self.goal_pose = goal
        self.robot_planned_path = path
        self.draw_map()

    def draw_map(self):
        if self.map_data is None or self.map_resolution == 0:
            return
        try:
            self.ax.clear()
            self.ax.set_aspect('equal')
            self.ax.imshow(self.map_data, cmap='gray', origin='lower', 
                           extent=[0, self.map_data.shape[1], 0, self.map_data.shape[0]])

            if self.robot_pose:
                robot_width_px = config.ROBOT_WIDTH_M / self.map_resolution
                robot_length_px = config.ROBOT_LENGTH_M / self.map_resolution
                map_x = (self.robot_pose[0] - self.map_origin_x) / self.map_resolution
                map_y = (self.robot_pose[1] - self.map_origin_y) / self.map_resolution

                rect = plt.Rectangle((map_x - robot_width_px / 2, map_y - robot_length_px / 2),
                                     robot_width_px, robot_length_px,
                                     angle=np.degrees(self.robot_pose[2]), rotation_point='center', color='#3498db', alpha=0.8)
                self.ax.add_patch(rect)
                
                # Mũi tên hướng
                dx = (robot_length_px * 0.8) * np.cos(self.robot_pose[2])
                dy = (robot_length_px * 0.8) * np.sin(self.robot_pose[2])
                self.ax.arrow(map_x, map_y, dx, dy, head_width=robot_width_px * 0.4, head_length=robot_length_px * 0.3, fc='red', ec='red')

            if self.goal_pose:
                gmap_x = (self.goal_pose[0] - self.map_origin_x) / self.map_resolution
                gmap_y = (self.goal_pose[1] - self.map_origin_y) / self.map_resolution
                circle = plt.Circle((gmap_x, gmap_y), 0.3 / self.map_resolution, color='#2ecc71', fill=True, alpha=0.7)
                self.ax.add_patch(circle)

            if self.robot_planned_path:
                path_x = [(p[0] - self.map_origin_x) / self.map_resolution for p in self.robot_planned_path]
                path_y = [(p[1] - self.map_origin_y) / self.map_resolution for p in self.robot_planned_path]
                self.ax.plot(path_x, path_y, color='#f1c40f', linewidth=3, linestyle='-')

            if self.current_xlim is None or self.current_ylim is None:
                self.current_xlim = [0, self.map_data.shape[1]]
                self.current_ylim = [0, self.map_data.shape[0]]

            self.ax.set_xlim(self.current_xlim)
            self.ax.set_ylim(self.current_ylim)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.fig.canvas.draw_idle()
        except Exception as e:
            pass

    def on_scroll(self, event):
        if event.inaxes != self.ax: return
        if event.xdata is None or event.ydata is None: return
        zoom_factor = 1.2 if event.button == 'up' else 0.8
        w = self.current_xlim[1] - self.current_xlim[0]
        h = self.current_ylim[1] - self.current_ylim[0]
        new_w, new_h = w / zoom_factor, h / zoom_factor
        r_x = (event.xdata - self.current_xlim[0]) / w
        r_y = (event.ydata - self.current_ylim[0]) / h
        self.current_xlim = [event.xdata - r_x * new_w, event.xdata + (1 - r_x) * new_w]
        self.current_ylim = [event.ydata - r_y * new_h, event.ydata + (1 - r_y) * new_h]
        self.draw_map()

    def on_button_press(self, event):
        if event.button == 1 and event.inaxes == self.ax:
            self.panning = True
            self.pan_start = (event.xdata, event.ydata)

    def on_motion(self, event):
        if self.panning and event.inaxes == self.ax:
            dx = event.xdata - self.pan_start[0]
            dy = event.ydata - self.pan_start[1]
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

        self.right_panel = QFrame()
        self.main_layout.addWidget(self.right_panel)
        self.right_layout = QVBoxLayout(self.right_panel)
        self.map_canvas = MapCanvas(self)
        self.right_layout.addWidget(self.map_canvas)

        signal_bus.frame_update.connect(self.update_camera_frame)
        signal_bus.status_update.connect(self.update_status)

    def update_camera_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(640, 480, Qt.KeepAspectRatio))

    def update_status(self, text):
        self.status_label.setText(text)

    def update_map_data(self, data, resolution, origin_x, origin_y):
        self.map_canvas.set_map_data(data, resolution, origin_x, origin_y)
        
    def update_robot_pose(self, pose):
        self.map_canvas.set_robot_pose(pose)
        
    def update_goal_and_path(self, goal_pose, path_pts):
        self.map_canvas.set_goal_and_path(goal_pose, path_pts)

    def closeEvent(self, event):
        rospy.signal_shutdown("GUI closed")
        event.accept()
