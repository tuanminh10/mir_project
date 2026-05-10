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
# SỬA XUNG ĐỘT QT PLUGIN GIỮA OPENCV VÀ PYQT5 & ROS MD5 HASH
# ==============================================================================
os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
os.environ['QT_API'] = 'pyqt5'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

_ros_sys_path = '/opt/ros/noetic/lib/python3/dist-packages'
if os.path.isdir(_ros_sys_path) and _ros_sys_path not in sys.path:
    sys.path.insert(1, _ros_sys_path)
    for mod_name in list(sys.modules.keys()):
        if any(mod_name.startswith(p) for p in ['geometry_msgs', 'nav_msgs', 'sensor_msgs',
                                                  'std_msgs', 'actionlib_msgs', 'tf2_msgs', 'move_base_msgs']):
            del sys.modules[mod_name]

import cv2
_cv2_qt_dir = os.path.join(os.path.dirname(cv2.__file__), 'qt', 'plugins')
if os.path.isdir(_cv2_qt_dir):
    try:
        import PyQt5
        _pyqt5_plugins = os.path.join(os.path.dirname(PyQt5.__file__), 'Qt5', 'plugins')
        if os.path.isdir(_pyqt5_plugins):
            os.environ['QT_PLUGIN_PATH'] = _pyqt5_plugins
    except Exception:
        pass

# ==============================================================================
# IMPORTS HỆ THỐNG VÀ MODULES
# ==============================================================================
import time
import math
import threading
import numpy as np

import rospy
import tf
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid, Path
from tf.transformations import euler_from_quaternion

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

# Import kiến trúc refactored
import config
from gui import MainWindow, signal_bus
from nav import MirNavigator
from vis import VisionTracker

class AppController:
    """Bộ điều khiển trung tâm (Orchestrator) nối phần AI, Navigation và GUI lại với nhau"""
    def __init__(self, main_window):
        self.window = main_window
        self.navigator = MirNavigator()
        
        # Khởi tạo Camera AI và truyền callback
        self.vision = VisionTracker(
            on_lock_callback=self.on_target_locked,
            on_unlock_callback=self.on_target_unlocked
        )
        
        # TF & State
        self.tf_listener = tf.TransformListener()
        self.robot_pose = None
        self.goal_pose = None
        self.robot_path = []
        
        # Timer lấy vị trí robot real-time từ TF
        self.tf_timer = QTimer()
        self.tf_timer.timeout.connect(self.update_robot_pose)
        self.tf_timer.start(100)

    def on_target_unlocked(self):
        rospy.loginfo("[Controller] Mở khóa. Hủy bỏ Navigation.")
        self.navigator.cancel_all()
        self.goal_pose = None
        self.window.update_goal_and_path(None, self.robot_path)

    def on_target_locked(self, forward_m, left_m):
        """Khi AI tìm thấy người và khóa (Raise Hand + Open 5), tính điểm đến và đi"""
        if self.robot_pose is None:
            rospy.logwarn("[Controller] Chưa lấy được tọa độ Robot từ TF, chưa thể ra lệnh đi.")
            self.on_target_unlocked()
            return
            
        try:
            # 1. Tạo điểm đến tương đối trên base_link
            msg = PointStamped()
            msg.header.stamp = rospy.Time(0)
            msg.header.frame_id = "base_link"
            # Trừ hao bù khoảng cách từ camera đến base_link
            msg.point.x = forward_m - config.CAMERA_OFFSET_X_M
            msg.point.y = left_m
            msg.point.z = 0.0

            # 2. Quy đổi ra tọa độ thế giới (Map)
            self.tf_listener.waitForTransform("/map", "base_link", rospy.Time(0), rospy.Duration(1.0))
            pt = self.tf_listener.transformPoint("/map", msg)
            target_x, target_y = pt.point.x, pt.point.y
            
            robot_x, robot_y = self.robot_pose[0], self.robot_pose[1]
            dx, dy = target_x - robot_x, target_y - robot_y
            dist_to_target = math.hypot(dx, dy)
            
            # 3. Trừ đi Offset tĩnh (Khoảng đệm an toàn Costmap)
            if dist_to_target <= config.STOP_DISTANCE_M:
                rospy.loginfo(f"[Controller] Đã ở gần người ({config.STOP_DISTANCE_M}m). IDLE.")
                self.on_target_unlocked()
                return
                
            ratio = (dist_to_target - config.STOP_DISTANCE_M) / dist_to_target
            final_goal_x = robot_x + dx * ratio
            final_goal_y = robot_y + dy * ratio
            final_yaw = math.atan2(dy, dx)
            
            self.goal_pose = (final_goal_x, final_goal_y)
            signal_bus.status_update.emit(f"Chỉ định Đích: Navigation ({final_goal_x:.1f}, {final_goal_y:.1f})")
            
            # 4. Ra lệnh cho Navigator
            if self.navigator.send_goal(final_goal_x, final_goal_y, final_yaw):
                signal_bus.status_update.emit("Robot đang di chuyển theo lộ trình Planner!")
            else:
                rospy.logerr("[Controller] Gửi lệnh Navigator thất bại!")
                
            # Cập nhật GUI
            self.window.update_goal_and_path(self.goal_pose, self.robot_path)

        except Exception as e:
            rospy.logerr(f"[Controller] Lỗi TF/Navigation: {e}")
            self.on_target_unlocked()

    def update_robot_pose(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform('/map', '/base_link', rospy.Time(0))
            self.robot_pose = (trans[0], trans[1], euler_from_quaternion(rot)[2])
            self.window.update_robot_pose(self.robot_pose)
        except Exception:
            pass

    def map_callback(self, msg):
        try:
            width = msg.info.width
            height = msg.info.height
            resolution = msg.info.resolution
            origin_x = msg.info.origin.position.x
            origin_y = msg.info.origin.position.y
            # Tránh lag cho Main GUI Window, việc truyền nparray là ổn nếu tốc độ không quá gắt
            data = np.array(msg.data).reshape((height, width))
            self.window.update_map_data(data, resolution, origin_x, origin_y)
        except Exception as e:
            pass

    def path_callback(self, msg):
        try:
            pts = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
            self.robot_path = pts
            self.window.update_goal_and_path(self.goal_pose, self.robot_path)
        except Exception:
            pass


def main():
    rospy.init_node('xlanav_refactored', anonymous=True)
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()
    
    controller = AppController(window)

    rospy.Subscriber('/map', OccupancyGrid, controller.map_callback)
    
    path_topics = [
        '/move_base_node/SBPLLatticePlanner/plan',
        '/move_base_node/GlobalPlanner/plan',
        '/move_base/GlobalPlanner/plan',
        '/move_base_node/mir_global_planner/plan',
        '/mir_planner/global_path'
    ]
    for topic in path_topics:
        rospy.Subscriber(topic, Path, controller.path_callback)

    rospy.loginfo("[Main] Khởi động hệ thống Refactored Xlanav.")
    threading.Thread(target=rospy.spin, daemon=True).start()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
