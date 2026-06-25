#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
import math
import time
import threading

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
import config

# ================= Utils =================

from vision import VideoThread
from gui import MapLabel

class MainApp(QMainWindow):
    map_signal = pyqtSignal(object)
    pose_signal = pyqtSignal(float, float, float)
    request_gui_update_signal = pyqtSignal()
    retry_nav_signal = pyqtSignal(float, float, object, float, int, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MiR Auto Navigation - V4 (Hybrid Safe Goal + Motor + Voice)")
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

        rospy.init_node('main_control_v4', anonymous=True)
        
        rospy.loginfo("Đang tải YOLO Laptop (Đồ uống)...")
        self.laptop_yolo = None
        try:
            self.laptop_yolo = YOLO('../tm/data/detect_laviecoca.v1-laviecoca.yolov11/runs/drink_v11s/weights/best.pt')
        except Exception as e:
            rospy.logwarn(f"Không tìm thấy model laptop best.pt: {e}")

        self.task_queue = queue.PriorityQueue()
        self.task_counter = 0
        self.active_orders = {} 
        self.saved_locations = {}
        self.current_location = "sac"
        
        self.wait_event = threading.Event()
        self.scanning_event = threading.Event()
        self.charging_cancel_event = threading.Event()
        self.target_locked_coords = None
        self.nav_arrived_event = threading.Event()

        try:
            self.servo = dongco.ServoController(pin=18, min_angle=0, max_angle=180)
            self.servo.set_angle(95)
        except Exception as e:
            print("Lỗi Servo:", e)
            self.servo = None

        self.robot = nav.ws_connect()
        self.mir_headers = nav.api_login()
        if self.mir_headers:
            nav.api_ensure_ready(self.mir_headers)

        self.map_signal.connect(self.map_label.set_map)
        self.pose_signal.connect(self.map_label.set_robot_pose)
        self.request_gui_update_signal.connect(self.map_label.update_view)
        self.retry_nav_signal.connect(self.calculate_hybrid_safe_goal)

        self.video_thread = VideoThread()
        self.video_thread.change_pixmap_signal.connect(self.update_camera_image)
        self.video_thread.target_locked_signal.connect(self.on_hand_locked)
        self.video_thread.is_scanning_for_hand = False
        self.video_thread.start()

        self.pub_arrived = rospy.Publisher('/robot_arrived_table', String, queue_size=10)

        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        rospy.Subscriber('/robot_pose', Pose, self.pose_callback)
        rospy.Subscriber('/table_call_buttons', String, self.on_guest_call)
        rospy.Subscriber('/robot_orders', String, self.on_web_order)
        rospy.Subscriber('/kitchen_commands', String, self.on_kitchen_cmd)
        
        self.local_obstacles_cells = []
        rospy.Subscriber('/move_base_node/local_costmap/obstacles', GridCells, self.local_costmap_callback)

        rospy.on_shutdown(self.on_shutdown_hook)

        self.worker_thread = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker_thread.start()
        mir_tts.speak_on_mir("Hệ thống phiên bản 4 đã sẵn sàng.")

    def cleanup_temp_positions(self):
        try:
            import requests
            if not self.mir_headers: return
            r = requests.get(f"{config.MIR_API_URL}/positions", headers=self.mir_headers, timeout=2)
            if r.status_code == 200:
                for p in r.json():
                    if p.get("name", "").startswith("_nav_") or p.get("name", "").startswith("_test_"):
                        requests.delete(f"{config.MIR_API_URL}/positions/{p['guid']}", headers=self.mir_headers, timeout=1)
                print("[CLEANUP] ✅ Đã dọn sạch các điểm _nav_ và _test_ trên MiR Map sau khi kết thúc cuốc xe.")
        except Exception as e:
            print(f"[CLEANUP ERROR] Lỗi khi dọn rác map: {e}")

    def on_shutdown_hook(self):
        rospy.logwarn("ĐANG DỪNG KHẨN CẤP...")
        self.video_thread.stop()
        if self.servo: self.servo.cleanup()
        if self.mir_headers: 
            nav.api_set_state(self.mir_headers, 4) # Đưa xe về Pause
            try:
                import requests
                # XÓA TẤT CẢ MISSION TRÊN WEB (QUEUE)
                requests.delete(f"{config.MIR_API_URL}/mission_queue", headers=self.mir_headers, timeout=2)
                rospy.logwarn("Đã xóa toàn bộ Mission Queue trên MiR Web!")
                self.cleanup_temp_positions()
            except Exception as e:
                rospy.logerr(f"Lỗi khi dọn dẹp REST API: {e}")

    def update_camera_image(self, cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qImg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self.camera_label.setPixmap(QPixmap.fromImage(qImg).scaled(self.camera_label.size(), Qt.KeepAspectRatio))

    def map_callback(self, msg): self.map_signal.emit(msg)
    def local_costmap_callback(self, msg): self.local_obstacles_cells = msg.cells

    def pose_callback(self, msg):
        q = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        yaw = tf.transformations.euler_from_quaternion(q)[2]
        self.pose_signal.emit(msg.position.x, msg.position.y, yaw)

    def on_guest_call(self, msg):
        try:
            data = json.loads(msg.data)
            ban = str(data.get("ban")).strip()
            if ban.isdigit(): ban = f"ban {ban}"
            self.task_queue.put((2, self.task_counter, {"type": "GUEST_CALL", "target": ban}))
            self.task_counter += 1
            self.charging_cancel_event.set()
        except: pass

    def on_web_order(self, msg):
        try:
            data = json.loads(msg.data)
            ban = str(data.get("ban", "")).strip()
            if ban.isdigit(): ban = f"ban {ban}"
            self.active_orders[ban] = {"coca": int(data.get("coca", 0)), "lavie": int(data.get("lavie", 0))}
            
            # Đã nhận được lệnh, báo cho vòng lặp ở bàn biết để rời đi
            self.wait_event.set()
        except: pass

    def on_kitchen_cmd(self, msg):
        try:
            data = json.loads(msg.data)
            action = data.get("action")
            if action == "call_robot":
                self.task_queue.put((1, self.task_counter, {"type": "KITCHEN_CALL", "target": "bep"}))
                self.task_counter += 1
                self.charging_cancel_event.set()
            elif action == "deliver":
                ban = str(data.get("table", "")).strip()
                if ban.isdigit(): ban = f"ban {ban}"
                if self.current_location != "bep":
                    self.task_queue.put((1, self.task_counter, {"type": "KITCHEN_CALL", "target": "bep"}))
                    self.task_counter += 1
                self.task_queue.put((1, self.task_counter, {"type": "DELIVER", "target": ban}))
                self.task_counter += 1
                self.charging_cancel_event.set()
        except: pass

    def on_hand_locked(self, mx, my, obj):
        self.target_locked_coords = (mx, my, obj)
        self.scanning_event.set()

    def move_to_static_goal(self, target_name, cancel_event=None):
        if target_name == self.current_location: return True
        if target_name not in nav.DIEM: return False
        nav.handle_command(target_name, self.robot, self.mir_headers, non_interactive=True, cancel_event=cancel_event)
        if cancel_event and cancel_event.is_set(): return False
        self.current_location = target_name
        return True

    def verify_tray(self, exp_coca, exp_lavie, check_empty=False, timeout=30.0):
        if not self.laptop_yolo: return True
        self.video_thread.pause_emit = True # Chặn VideoThread tự update GUI
        start = rospy.Time.now()
        success = 0
        last_print_time = 0
        try:
            while (rospy.Time.now() - start).to_sec() < timeout:
                frame = self.video_thread.get_latest_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue
                    
                res = self.laptop_yolo.track(frame, persist=True, stream=True, conf=0.40, verbose=False)
                coca, lavie = 0, 0
                
                annotated_frame = frame.copy()
                for r in res:
                    # RENDER BOUNDING BOX TỪ YOLO
                    annotated_frame = r.plot()
                    if r.boxes:
                        for b in r.boxes:
                            if int(b.cls[0]) == 0: coca += 1
                            else: lavie += 1
                            
                # VẼ MENU VÀ CHỮ LÊN GIAO DIỆN
                ec, el = max(0, exp_coca), max(0, exp_lavie)
                if ec == 0 and el == 0: el = 1
                
                if check_empty:
                    cv2.putText(annotated_frame, "CHO KHACH LAY DO", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                    cv2.putText(annotated_frame, "Yeu cau: KHAY TRONG", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if coca > 0 or lavie > 0:
                        cv2.putText(annotated_frame, f"Con lai: {coca} Coca, {lavie} Lavie", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(annotated_frame, "DANG KIEM TRA DO UONG BEP LEN", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(annotated_frame, f"Coca: {coca}/{ec}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if coca >= ec else (0, 0, 255), 2)
                    cv2.putText(annotated_frame, f"Lavie: {lavie}/{el}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if lavie >= el else (0, 0, 255), 2)
                
                # ÉP PHÁT ẢNH NÀY LÊN GIAO DIỆN CAMERA CHÍNH
                self.video_thread.change_pixmap_signal.emit(annotated_frame)
                            
                now = time.time()
                if now - last_print_time > 1.0:
                    if check_empty:
                        print(f"[VERIFY TRAY] 📷 Đang chờ khách lấy đồ... Hiện tại trên khay: Coca: {coca}, Lavie: {lavie}")
                    else:
                        print(f"[VERIFY TRAY] 📷 Đang kiểm tra đồ uống: Coca {coca}/{ec} | Lavie {lavie}/{el}")
                    last_print_time = now
    
                if check_empty:
                    if coca == 0 and lavie == 0: success += 1
                    else: success = 0
                else:
                    if coca >= ec and lavie >= el: success += 1
                    else: success = 0
                    
                if success >= 5:
                    if check_empty: print(f"[VERIFY TRAY] ✅ KHÁCH ĐÃ LẤY HẾT ĐỒ! Tắt AI quét nước.")
                    else: print(f"[VERIFY TRAY] ✅ ĐỒ UỐNG ĐÃ ĐỦ! Tắt AI quét nước.")
                    return True
                time.sleep(0.05)
        finally:
            self.video_thread.pause_emit = False
            
        print("[VERIFY TRAY] ❌ Hết thời gian chờ!")
        return False

    def worker_loop(self):
        while not rospy.is_shutdown():
            try:
                priority, count, task = self.task_queue.get(timeout=2.0)
                try:
                    if task["type"] == "RETURN_HOME":
                        self.charging_cancel_event.clear()
                        self.execute_task(task, cancel_event=self.charging_cancel_event)
                    else:
                        self.execute_task(task)
                except Exception as e: print(e)
                finally: self.task_queue.task_done()
            except queue.Empty:
                if self.current_location not in ["sac", "moving_to_sac", "bep"]:
                    self.current_location = "moving_to_sac"
                    self.task_queue.put((3, self.task_counter, {"type": "RETURN_HOME", "target": "sac"}))
                    self.task_counter += 1

    def execute_task(self, task, cancel_event=None):
        ttype, target = task["type"], task["target"]
        
        if ttype == "GUEST_CALL":
            mir_tts.speak_on_mir(f"Đã nhận lệnh gọi từ {get_vn_name(target)}, xe đang di chuyển tới.")
            if self.servo: self.servo.set_angle(95)
            ok = self.move_to_static_goal(target, cancel_event=cancel_event)
            if not ok: return
            
            if self.servo: self.servo.set_angle(95)
            mir_tts.speak_on_mir("Chào quý khách, khách nào order thì giơ tay lên.")
            
            self.target_locked_coords = None
            self.scanning_event.clear()
            self.video_thread.is_scanning_for_hand = True
            
            if self.scanning_event.wait(timeout=30.0):
                self.video_thread.is_scanning_for_hand = False
                tx, ty, obj = self.target_locked_coords
                
                self.nav_arrived_event.clear()
                self.calculate_hybrid_safe_goal(tx, ty, obj, mode="ORDER")
                self.nav_arrived_event.wait()
                
                self.current_location = "specific_" + target
                self.saved_locations[target] = {"tx": tx, "ty": ty, "obj": obj}
                
                mir_tts.speak_on_mir("Mời khách order.")
                self.wait_event.clear()
                self.pub_arrived.publish(json.dumps({"action": "popup_menu", "ban": target}))
                
                print(f"Đang chờ {target} order qua giao diện Web...")
                self.wait_event.wait(timeout=120) 
                self.wait_event.clear()
                
                mir_tts.speak_on_mir("Đã nhận order, xe xin phép rời đi.")
            else:
                self.video_thread.is_scanning_for_hand = False
                mir_tts.speak_on_mir("Không thấy ai giơ tay, robot xin phép quay về.")

        elif ttype == "RETURN_HOME":
            mir_tts.speak_on_mir("Robot đang quay về vị trí sạc.")
            if self.servo: self.servo.set_angle(95)
            self.move_to_static_goal("sac", cancel_event=cancel_event)

        elif ttype == "KITCHEN_CALL":
            mir_tts.speak_on_mir("Đang quay về bếp để nhận món.")
            if self.servo: self.servo.set_angle(95)
            ok = self.move_to_static_goal("bep", cancel_event=cancel_event)
            if ok: 
                if self.servo: self.servo.set_angle(155) 
                mir_tts.speak_on_mir("Mời bếp đặt đồ lên xe và bấm nút xác nhận.")

        elif ttype == "DELIVER":
            if self.servo: self.servo.set_angle(155)
            exp_coca = self.active_orders.get(target, {}).get("coca", 0)
            exp_lavie = self.active_orders.get(target, {}).get("lavie", 0)
            
            mir_tts.speak_on_mir("Đang kiểm tra đồ uống trên khay.")
            has_items = self.verify_tray(exp_coca, exp_lavie)
            if not has_items:
                mir_tts.speak_on_mir("Hình như đặt thiếu đồ, xin bếp kiểm tra lại.")
                return 
            else:
                mir_tts.speak_on_mir(f"Đồ uống đã đủ, robot bắt đầu đi giao tới {get_vn_name(target)}.")
                
            if self.servo: self.servo.set_angle(95)
            
            if target in self.saved_locations:
                deliver_data = self.saved_locations[target]
                if isinstance(deliver_data, dict) and "tx" in deliver_data:
                    tx = deliver_data["tx"]
                    ty = deliver_data["ty"]
                    obj = deliver_data["obj"]
                    self.nav_arrived_event.clear()
                    self.calculate_hybrid_safe_goal(tx, ty, obj, mode="DELIVER")
                    self.nav_arrived_event.wait(timeout=90)
                else:
                    tx, ty = deliver_data
                    self.nav_arrived_event.clear()
                    self.calculate_hybrid_safe_goal(tx, ty, None, mode="DELIVER")
                    self.nav_arrived_event.wait(timeout=90)
                self.current_location = "specific_" + target
            else:
                ok = self.move_to_static_goal(target, cancel_event=cancel_event)
                if not ok: return
                
            mir_tts.speak_on_mir(f"Đã tới nơi. Mời khách lấy đồ uống.")
            
            if self.servo: 
                self.servo.set_angle(155)
                time.sleep(2.0)
            
            is_empty = False
            for i in range(3):
                is_empty = self.verify_tray(0, 0, check_empty=True, timeout=20.0)
                if is_empty: break
                if i < 2: mir_tts.speak_on_mir("Quý khách vui lòng lấy hết đồ uống trên khay để robot tiếp tục làm việc.")
            
            if self.servo: self.servo.set_angle(95)
            
            if is_empty: mir_tts.speak_on_mir("Cảm ơn quý khách. Chúc quý khách ngon miệng.")
            else: mir_tts.speak_on_mir("Đã quá thời gian chờ, robot xin phép quay về.")
                
            self.active_orders.pop(target, None)
            self.cleanup_temp_positions()

    def calculate_hybrid_safe_goal(self, target_x, target_y, obs_pt_map=None, min_dist_m=0.50, attempt=0, mode="ORDER"):
        if not self.map_label.map_info or self.map_label.robot_px is None:
            return
            
        res = self.map_label.map_info.resolution
        ox = self.map_label.map_info.origin.position.x
        oy = self.map_label.map_info.origin.position.y
        w = self.map_label.map_info.width
        h = self.map_label.map_info.height
        
        px_t = int((target_x - ox) / res)
        py_t = int((target_y - oy) / res)
        
        if not (0 <= px_t < w and 0 <= py_t < h): return

        self.map_label.target_px = (int(px_t), h - int(py_t) - 1)
        self.request_gui_update_signal.emit()

        obs_mask = np.where((self.map_label.map_data != 0), 255, 0).astype(np.uint8)
        combined_obs = obs_mask.copy()
        raycast_obs = np.where((self.map_label.map_data == 100), 255, 0).astype(np.uint8)
        
        if obs_pt_map:
            obs_px_x = int((obs_pt_map[0] - ox) / res)
            obs_px_y = int((obs_pt_map[1] - oy) / res)
            self.map_label.obs3d_px = (obs_px_x, h - obs_px_y - 1)
            if 0 <= obs_px_x < w and 0 <= obs_px_y < h:
                radius_px = int(0.15 / res)
                cv2.circle(combined_obs, (obs_px_x, obs_px_y), radius_px, 255, -1)
                cv2.circle(raycast_obs, (obs_px_x, obs_px_y), radius_px, 255, -1)
        else: self.map_label.obs3d_px = None

        if hasattr(self, 'local_obstacles_cells') and self.local_obstacles_cells:
            for p in self.local_obstacles_cells:
                obs_px_x = int((p.x - ox) / res)
                obs_px_y = int((p.y - oy) / res)
                if 0 <= obs_px_x < w and 0 <= obs_px_y < h:
                    radius_px = int(0.10 / res) 
                    cv2.circle(combined_obs, (obs_px_x, obs_px_y), radius_px, 255, -1)
                    cv2.circle(raycast_obs, (obs_px_x, obs_px_y), radius_px, 255, -1)

        inflate_m = 0.15
        inflate_px = max(1, int(inflate_m / res))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*inflate_px+1, 2*inflate_px+1))
        inflated_obs = cv2.dilate(combined_obs, kernel, iterations=1)
        
        self.map_label.cone_pixels = []
        self.map_label.ray_pixels = []
        
        best_rays = []
        max_ray_len = int(5.0 / res)
        for deg in range(0, 360, 5):
            rad = math.radians(deg)
            free_count = 0
            for step in range(1, max_ray_len):
                cx = int(px_t + step * math.cos(rad))
                cy = int(py_t + step * math.sin(rad))
                if not (0 <= cx < w and 0 <= cy < h): break
                if self.map_label.map_data[cy, cx] == 0: free_count += 1
                elif self.map_label.map_data[cy, cx] == 100: break
            best_rays.append((rad, free_count))
            
        max_free = max(c for _, c in best_rays)
        open_rays = [r for r, c in best_rays if c >= max_free * 0.9]
        sx = sum(math.cos(r) for r in open_rays)
        sy = sum(math.sin(r) for r in open_rays)
        theta_open = math.atan2(sy, sx)
        
        theta_raw_left = theta_open + math.radians(45)
        theta_raw_right = theta_open - math.radians(45)
        global_angles = [45, 135, -45, -135]
        def get_snapped_angle(raw_rad):
            deg = math.degrees(raw_rad)
            best = min(global_angles, key=lambda a: abs((a - deg + 180) % 360 - 180))
            return math.radians(best), best
            
        theta_left, deg_left = get_snapped_angle(theta_raw_left)
        theta_right, deg_right = get_snapped_angle(theta_raw_right)
        
        fp_m = [(0.506, -0.32), (0.506, 0.32), (-0.454, 0.32), (-0.454, -0.32)]
        min_step = int(max(0.50, min_dist_m) / res)
        self.map_label.all_rejected_pts = []
        
        def test_path(theta_dock_test, yaw_test):
            for step in range(min_step, max_ray_len):
                cx = int(px_t + step * math.cos(theta_dock_test))
                cy = int(py_t + step * math.sin(theta_dock_test))
                if not (0 <= cx < w and 0 <= cy < h): break
                pts = []
                for dx, dy in fp_m:
                    rx = (dx * math.cos(yaw_test) - dy * math.sin(yaw_test)) / res
                    ry = (dx * math.sin(yaw_test) + dy * math.cos(yaw_test)) / res
                    pts.append([int(cx + rx), int(cy + ry)])
                pts = np.array(pts, np.int32).reshape((-1, 1, 2))
                x_min, y_min = np.min(pts, axis=0)[0]
                x_max, y_max = np.max(pts, axis=0)[0]
                x_min = max(0, x_min); y_min = max(0, y_min)
                x_max = min(w-1, x_max); y_max = min(h-1, y_max)
                if x_min >= x_max or y_min >= y_max: continue
                roi = inflated_obs[y_min:y_max+1, x_min:x_max+1]
                local_pts = pts - np.array([x_min, y_min])
                mask = np.zeros_like(roi)
                cv2.fillPoly(mask, [local_pts], 255)
                if not np.any((roi > 0) & (mask > 0)): return step, pts
            return None, None

        yaw_o_left = (theta_left - math.pi + math.pi) % (2 * math.pi) - math.pi
        step_left, pts_left = test_path(theta_left, yaw_o_left)
        yaw_o_right = (theta_right - math.pi + math.pi) % (2 * math.pi) - math.pi
        step_right, pts_right = test_path(theta_right, yaw_o_right)
        
        theta_dock = theta_left
        target_step = None
        if step_left is not None and step_right is not None:
            if step_left <= step_right: theta_dock = theta_left; target_step = step_left
            else: theta_dock = theta_right; target_step = step_right
        elif step_left is not None: theta_dock = theta_left; target_step = step_left
        elif step_right is not None: theta_dock = theta_right; target_step = step_right
        else: theta_dock = theta_left; target_step = min_step
            
        theta_d_left = theta_open + math.radians(85)
        theta_d_right = theta_open - math.radians(85)
        def get_deliver_yaw(dock_rad):
            deg = math.degrees(dock_rad)
            best_yaw = min([0, 180], key=lambda a: abs((a - deg + 180) % 360 - 180))
            return math.radians(best_yaw)
        yaw_d_left = get_deliver_yaw(theta_d_left)
        yaw_d_right = get_deliver_yaw(theta_d_right)
        step_d_left, _ = test_path(theta_d_left, yaw_d_left)
        step_d_right, _ = test_path(theta_d_right, yaw_d_right)
        theta_dock_d = theta_d_left; target_step_d = None; yaw_d = yaw_d_left
        if step_d_left is not None and step_d_right is not None:
            if step_d_left <= step_d_right: theta_dock_d = theta_d_left; target_step_d = step_d_left; yaw_d = yaw_d_left
            else: theta_dock_d = theta_d_right; target_step_d = step_d_right; yaw_d = yaw_d_right
        elif step_d_left is not None: theta_dock_d = theta_d_left; target_step_d = step_d_left; yaw_d = yaw_d_left
        elif step_d_right is not None: theta_dock_d = theta_d_right; target_step_d = step_d_right; yaw_d = yaw_d_right
        else: theta_dock_d = theta_d_left; target_step_d = min_step; yaw_d = yaw_d_left
            
        final_step_d = (target_step_d * res + 0.10) / res
        px_x_d = int(px_t + final_step_d * math.cos(theta_dock_d))
        px_y_d = int(py_t + final_step_d * math.sin(theta_dock_d))
        q_d = tf.transformations.quaternion_from_euler(0, 0, yaw_d)
        
        self.last_calculated_deliver_diem = {
            "x": ox + px_x_d * res, 
            "y": oy + px_y_d * res, 
            "qz": q_d[2], 
            "qw": q_d[3], 
            "arrive_dist": 0.15
        }
        
        self.map_label.deliver_px = (px_x_d, h - px_y_d - 1)
        self.map_label.deliver_yaw = yaw_d
        
        yaw = theta_dock - math.pi 
        yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
        
        for step in range(min_step, target_step + 1):
            cx = int(px_t + step * math.cos(theta_dock))
            cy = int(py_t + step * math.sin(theta_dock))
            if not (0 <= cx < w and 0 <= cy < h): break
            self.map_label.ray_pixels.append((cx, h - cy - 1))

            
        target_dist_m = target_step * res
        target_dist_m += 0.10 # Lùi thêm an toàn so với điểm check cuối cùng
        
        final_step = target_dist_m / res
        
        # TÍNH TOÁN TỌA ĐỘ VÀ GÓC QUAY
        px_x = int(px_t + final_step * math.cos(theta_dock))
        px_y = int(py_t + final_step * math.sin(theta_dock))
        
        w_x = ox + px_x * res
        w_y = oy + px_y * res
        
        q = tf.transformations.quaternion_from_euler(0, 0, yaw)
        
        diem_dong = {
            "x": w_x, 
            "y": w_y, 
            "qz": q[2], 
            "qw": q[3], 
            "arrive_dist": 0.15,
            "dist_m": target_dist_m
        }
        
        # Hiển thị GUI NGAY LẬP TỨC để User thấy tọa độ dù có lỗi
        self.map_label.target_px = (int(px_t), h - int(py_t) - 1)
        self.map_label.goal_yaw = yaw
        self.map_label.goal_px = (px_x, h - px_y - 1)
        
        if mode == "DELIVER":
            print(f"[SMART NAV] 🔄 Chuyển hướng sang tọa độ ĐỖ NGANG (Deliver Point)")
            theta_dock = theta_dock_d
            yaw = yaw_d
            target_dist_m = target_step_d * res + 0.10
            q = tf.transformations.quaternion_from_euler(0, 0, yaw)
            w_x = ox + px_x_d * res
            w_y = oy + px_y_d * res
            self.map_label.goal_yaw = yaw
            self.map_label.goal_px = (px_x_d, h - px_y_d - 1)
            
        self.request_gui_update_signal.emit()
        
        print(f"[SMART NAV] ✅ Chốt điểm đỗ DUY NHẤT dựa theo Lidar: Cự ly {target_dist_m:.2f}m, Góc = {math.degrees(yaw):.1f}°")
        print(f"🚀 [NAV] Bắn lệnh tới MiR Fleet / MoveBase!")
        
        self.current_goal = (w_x, w_y)
        self.is_moving = True
        
        # CHẠY NAVIGATION TRONG THREAD RIÊNG ĐỂ KHÔNG BLOCK QT MAIN THREAD
        import threading
        def _nav_worker():
            current_dist_m = target_dist_m
            max_retries = config.MAX_NAV_RETRIES 
            final_success = False
            
            # Tính tọa độ đích
            f_step = current_dist_m / res
            n_px_x = int(px_t + f_step * math.cos(theta_dock))
            n_px_y = int(py_t + f_step * math.sin(theta_dock))
            
            n_w_x = ox + n_px_x * res
            n_w_y = oy + n_px_y * res
            
            current_diem = {
                "x": n_w_x, 
                "y": n_w_y, 
                "qz": q[2], 
                "qw": q[3], 
                "arrive_dist": 0.15,
                "dist_m": current_dist_m
            }
            
            print(f"\n========================================================")
            print(f"[SMART NAV] 🚀 BẮT ĐẦU THỬ NGHIỆM ĐỖ LẦN {attempt+1}/{max_retries}")
            print(f"📍 Tọa độ gửi xuống MiR: X={n_w_x:.2f}, Y={n_w_y:.2f} (Cự ly lùi: {current_dist_m:.2f}m)")
            print(f"========================================================\n")
            
            rest_ok = False
            
            if hasattr(self, 'mir_headers') and self.mir_headers:
                try:
                    rest_ok = nav.api_navigate(self.mir_headers, [current_diem], "diem_dong")
                except Exception as e:
                    print(f"[SMART NAV] ❌ CRASH API: {e}")
                    
            if rest_ok:
                print(f"\n🎉 QUÁ TUYỆT VỜI! MiR đã chấp nhận điểm ở cự ly {current_dist_m:.2f}m. BẮT ĐẦU THEO DÕI HÀNH TRÌNH...\n")
                
                # THEO DÕI HÀNH TRÌNH CHO TỚI KHI ĐẾN ĐÍCH (Chống Lỗi Tím giữa đường)
                reached = False
                last_moving_time = time.time()
                last_px, last_py = None, None
                
                while True:
                    time.sleep(1.0)
                    try:
                        st = nav.api_status(self.mir_headers)
                        if st:
                            s_id = st.get("state_id", -1)
                            
                            # Nếu dính Lỗi Tím (10, 12) TRONG LÚC ĐANG CHẠY
                            if s_id in (10, 12):
                                rest_ok = False # Đánh dấu LỖI để kích hoạt tính năng lùi ở block else
                                print(f"   [THEO DÕI] 💥 CHẾT RỒI! Đang chạy thì bị Lỗi Tím (State {s_id})!")
                                break
                                
                            # Lấy vị trí để check kẹt
                            if "position" in st:
                                rx, ry = st["position"].get("x"), st["position"].get("y")
                                dist_to_goal = math.hypot(rx - n_w_x, ry - n_w_y)
                                
                                # Nếu đã đến đích an toàn (State 3 - Ready và cự ly còn lại < 0.4m)
                                if s_id == 3 and dist_to_goal < 0.4:
                                    print(f"   [THEO DÕI] 🎯 Đã đến đích an toàn tuyệt đối!")
                                    reached = True
                                    break
                                    
                                if s_id == 4:
                                    # Nếu đang pause, không đếm thời gian kẹt
                                    last_moving_time = time.time()
                                    
                                if s_id == 5:
                                    # Tính toán xem có bị kẹt cứng một chỗ quá lâu không
                                    if last_px is not None:
                                        moved_dist = math.hypot(rx - last_px, ry - last_py)
                                        if moved_dist > 0.05:
                                            last_moving_time = time.time()
                                            last_px, last_py = rx, ry
                                    else:
                                        last_px, last_py = rx, ry
                                        
                                    # Nếu đã đứng im 8 giây mà vẫn đang Executing -> Bị kẹt vật cản vô hình!
                                    if time.time() - last_moving_time > 8.0:
                                        rest_ok = False
                                        print(f"   [THEO DÕI] 🐢 BỊ KẸT CỨNG MỘT CHỖ QUÁ 8 GIÂY! Tự động ép Lỗi!")
                                        break
                    except Exception as e:
                        pass
                        
                if reached:
                    final_success = True
                    self.nav_arrived_event.set()
            
            # Nếu rest_ok == False (Do Lỗi Tím hoặc Bị Kẹt Cứng)
            if not rest_ok:
                print(f"\n⚠️ [CẢNH BÁO] PHÁT HIỆN LỖI TÍM HOẶC BỊ KẸT TỪ BỘ NÃO MIR! ⚠️")
                print(f"   Nguyên nhân: Tại cự ly {current_dist_m:.2f}m vẫn bị đè lên vật cản thực tế.")
                print(f"   💡 HƯỚNG GIẢI QUYẾT: QUÉT LẠI LIDAR VÀ TÍNH TOÁN LẠI TỪ ĐẦU!")
                
                # 1 & 2. XÓA QUEUE VÀ XÓA LỖI TÍM (REST API)
                try:
                    import requests
                    requests.delete(f"{config.MIR_API_URL}/mission_queue", headers=self.mir_headers, timeout=2)
                    requests.put(f"{config.MIR_API_URL}/status", headers=self.mir_headers, json={"clear_error": True}, timeout=2)
                    print(f"   ✅ Đã hủy lệnh cũ và Xóa Lỗi thành công.")
                except:
                    print(f"   ❌ Không thể gửi lệnh Xóa Lỗi.")
                
                if attempt < max_retries - 1:
                    print(f"   🔄 Bắn tín hiệu tính toán lại mục tiêu từ cự ly {current_dist_m + 0.10:.2f}m...")
                    time.sleep(1.0)
                    self.retry_nav_signal.emit(target_x, target_y, obs_pt_map, current_dist_m + 0.10, attempt + 1, mode)
                    return # Thoát thread cũ
                else:
                    print("[SMART NAV] ⚠️ Đã hết số lần tự động lùi (6 lần). Lỗi quá nặng hoặc đường bị chặn kín!")
                    
            if not final_success and attempt >= max_retries - 1:
                if self.robot:
                    print("Ép chạy fallback bằng ROS (ws_send_goal)...\n")
                    nav.ws_send_goal(self.robot, current_diem)
                else:
                    print("Lỗi cực nặng: Không có fallback ROS. Hủy lệnh để tránh treo robot.\n")
                self.nav_arrived_event.set()
                
        threading.Thread(target=_nav_worker, daemon=True).start()





    

def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()