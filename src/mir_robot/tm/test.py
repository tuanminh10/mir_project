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

from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import pyrealsense2 as rs

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
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PointStamped, Pose

# Thay đổi bằng file import nav của bạn
import navigationcacdiem as nav
from ultralytics import YOLO
import mediapipe as mp

# ================= Utils =================
def get_depth_distance_m(depth_frame, box, frame_w, frame_h, center_pt=None):
    x1, y1, x2, y2 = map(int, box)
    
    # Quét toàn bộ khung người (Bounding Box) thay vì chỉ dải giữa ngực
    # Lý do: Nếu người mặc áo đen/phản xạ kém, vùng ngực sẽ bị nhiễu (distance=0).
    # Quét toàn bộ và lấy Percentile 5% sẽ đảm bảo lấy trúng khuôn mặt hoặc cánh tay (gần nhất).
    distances = []
    step_x = max(2, (x2 - x1) // 20)
    step_y = max(2, (y2 - y1) // 20)
    
    for px in range(x1, x2, step_x):
        for py in range(y1, y2, step_y):
            orig_px = frame_w - 1 - px
            if 0 <= orig_px < frame_w and 0 <= py < frame_h:
                d = depth_frame.get_distance(orig_px, py)
                # Bỏ qua nhiễu quá gần (<0.2m)
                if 0.2 < d < 6.0: distances.append(d)
    
    if distances:
        distances.sort()
        # Lấy Percentile 5% (Gần nhất có thể) để bỏ qua mảng background tường
        idx = int(len(distances) * 0.05)
        return float(distances[idx])
    return -1.0

# Vô hiệu hóa hàm segmentation vì nó thường xuyên nhận diện sai và lấy d_m của bức tường
# def get_depth_distance_m_seg(...)

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
    forward_m = z_opt * math.cos(pitch_rad) - y_opt * math.sin(pitch_rad)
    left_m = -x_opt
    
    return forward_m, left_m, z_opt

# ================= GUI Map =================
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
        # if hasattr(self, 'ray_pixels') and self.ray_pixels:
        #     for pt in self.ray_pixels:
        #         cv2.circle(display_img, pt, 1, (255, 255, 0), -1)

        # Vẽ vật cản 3D từ Camera
        if self.obs3d_px:
            cv2.circle(display_img, self.obs3d_px, 5, (255, 0, 255), -1) # Tím
            cv2.putText(display_img, "3D OBS", (self.obs3d_px[0]+10, self.obs3d_px[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        # Vẽ Goal an toàn
        if self.goal_px:
            cv2.circle(display_img, self.goal_px, 6, (0, 255, 0), -1) # Xanh lá
            cv2.putText(display_img, "SMART GOAL", (self.goal_px[0]+10, self.goal_px[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
            
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

        # Vẽ Robot
        if self.robot_px and self.map_info:
            res = self.map_info.resolution
            rl, rw = (0.89 / res) / 2, (0.58 / res) / 2
            pts = []
            for dx, dy in [(-rl, -rw), (rl, -rw), (rl, rw), (-rl, rw)]:
                rx = dx * math.cos(-self.robot_yaw) - dy * math.sin(-self.robot_yaw)
                ry = dx * math.sin(-self.robot_yaw) + dy * math.cos(-self.robot_yaw)
                pts.append([int(self.robot_px[0] + rx), int(self.robot_px[1] + ry)])
            pts = np.array(pts, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(display_img, [pts], (0, 165, 255))
            cv2.polylines(display_img, [pts], True, (0, 0, 0), 2)

        h, w, ch = display_img.shape
        qImg = QImage(display_img.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self.setPixmap(QPixmap.fromImage(qImg))

    def mouseReleaseEvent(self, event):
        if self.map_info is None: return
        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        h = self.map_info.height
        
        px, py = event.x(), event.y()
        wx = ox + px * res
        wy = oy + (h - py - 1) * res
        
        self.clicked_signal.emit(wx, wy, None)

# ================= Camera Thread =================
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    target_locked_signal = pyqtSignal(float, float, object)
    status_update_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.device = 0 if os.path.exists('/opt/ai_venv/bin/python') else 'cpu'
        self.model_pose = YOLO('yolo11n-pose.pt')
        self.model_seg = YOLO('yolo11n-seg.pt')
        if self.device == 0:
            self.model_pose.to('cuda')
            self.model_seg.to('cuda')
            
        self.mp_hands = mp.solutions.hands
        self.hands_detector = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        self.locked_target_id = None
        self.robot_state = "IDLE"
        self.locked_bbox = None
        
        self.hand_raise_start = {}
        self.open5_confirm_count = {}
        
        self.fist_confirm_count = 0
        self.fist_hold_start = None

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
            print("[INFO] Đã KẾT NỐI RealSense (Không dùng PointCloud để tối ưu)!")
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

            results_pose = self.model_pose.track(frame, conf=0.45, persist=True, tracker="bytetrack.yaml", verbose=False, half=(self.device==0), device=self.device)
            results_seg = self.model_seg.predict(frame, conf=0.45, verbose=False, half=(self.device==0), device=self.device)
            
            need_mediapipe = False
            if self.robot_state == "IDLE" or self.locked_target_id is not None:
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
            annotated_frame = frame.copy()

            for result_pose in results_pose:
                boxes = result_pose.boxes
                if boxes is None or boxes.id is None: continue
                
                keypoints = result_pose.keypoints.data if result_pose.keypoints is not None else None
                seg_result = results_seg[0] if len(results_seg) > 0 else None
                masks = seg_result.masks.xy if (seg_result and seg_result.masks is not None) else None

                for i, box in enumerate(boxes):
                    track_id = int(box.id[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
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
                    
                    # SỬA LỖI TỌA ĐỘ CUSTOMER CÁCH XA BÀN:
                    # Bỏ dùng Segmentation vì nó dính background (bức tường 2.49m)
                    # Bỏ quét toàn bộ Box vì 5% percentile sẽ lấy trúng "Bàn tay đang giơ ra" -> d_m bị ngắn lại!
                    # Giải pháp: Ưu tiên lấy chiều sâu (Depth) từ vùng Mặt và Vai.
                    body_distances = []
                    if keypoints is not None and i < len(keypoints):
                        kp = keypoints[i]
                        for k_idx in range(7): # Từ Mũi (0) đến Vai phải (6)
                            if len(kp) > k_idx and len(kp[k_idx]) >= 3 and kp[k_idx][2].item() > 0.4:
                                kx, ky = int(kp[k_idx][0].item()), int(kp[k_idx][1].item())
                                # Quét một vùng nhỏ quanh mỗi keypoint thân trên
                                for dx in range(-15, 16, 5):
                                    for dy in range(-15, 16, 5):
                                        px, py = kx + dx, ky + dy
                                        if 0 <= px < frame_w and 0 <= py < frame_h:
                                            orig_px = frame_w - 1 - px
                                            d = depth_frame.get_distance(orig_px, py)
                                            if 0.2 < d < 6.0: body_distances.append(d)
                                            
                    if body_distances:
                        body_distances.sort()
                        # Lấy Median (50%) của thân trên để đảm bảo đo đúng người
                        # CỘNG THÊM 0.15m: Bù trừ độ dày cơ thể (từ bề mặt ngực đến mép bàn)
                        d_m = float(body_distances[int(len(body_distances) * 0.5)]) + 0.15
                    else:
                        # Fallback nếu không thấy keypoint thân trên
                        d_m = get_depth_distance_m(depth_frame, (x1, y1, x2, y2), frame_w, frame_h, (person_center_x, person_center_y)) + 0.15
                    d_ngang_m = d_m
                    
                    is_raising = False
                    if keypoints is not None and i < len(keypoints):
                        kp = keypoints[i]
                        if len(kp) >= 11:
                            wrist_y = min(kp[9][1].item(), kp[10][1].item())
                            shoulder_y = min(kp[5][1].item(), kp[6][1].item())
                            if wrist_y < shoulder_y and wrist_y > 0:
                                is_raising = True
                    
                    has_open_five = False
                    has_fist = False
                    for hx, hy, fingers, open5 in detected_hands:
                        if x1 - 30 < hx < x2 + 30 and y1 - 30 < hy < y2 + 30:
                            if open5: has_open_five = True
                            if fingers <= 1: has_fist = True

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
                                cv2.putText(annotated_frame, f"DANG KHOA TARGET: {hold_time:.1f}s/2s", (int(x1), int(y1)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                if hold_time >= 2.0:
                                    self.hand_raise_start.pop(track_id, None)
                                    self.open5_confirm_count[track_id] = 0
                                    
                                    self.locked_target_id = track_id
                                    self.locked_bbox = (x1, y1, x2, y2)
                                    self.robot_state = "COLLECTING"
                                    self.status_update_signal.emit(f"ĐÃ KHÓA ! Đang tải dữ liệu không gian...")
                                    
                                    # Chuyển đổi tọa độ ngay lập tức và emit signal
                                    rel = get_person_relative_position_m(depth_frame, (person_center_x, person_center_y), frame_w, frame_h, self.depth_intrinsics, d_m)
                                    if rel is not None:
                                        # BÊ NGUYÊN XI CÔNG THỨC TỪ dung.py ĐỂ GIỐNG 100%
                                        camera_offset_x = 0.475
                                        forward_m, left_m = rel[0] - camera_offset_x, rel[1]
                                        
                                        msg = PointStamped()
                                        msg.header.stamp = rospy.Time(0)
                                        msg.header.frame_id = "base_link"
                                        msg.point.x = forward_m
                                        msg.point.y = left_m
                                        msg.point.z = 0.0
                                        
                                        try:
                                            self.tf_listener.waitForTransform("/map", "base_link", rospy.Time(0), rospy.Duration(2.0))
                                            pt = self.tf_listener.transformPoint("/map", msg)
                                            # Gọi target_locked_signal để TestPCApp gọi calculate_hybrid_safe_goal
                                            self.target_locked_signal.emit(pt.point.x, pt.point.y, None) # Không dùng obs_pt_map
                                            self.robot_state = "MOVING"
                                        except Exception as e:
                                            print(f"Lỗi TF: {e}")
                                            self.robot_state = "IDLE"
                                            self.locked_target_id = None

                    if track_id != -1 and track_id == self.locked_target_id:
                        if is_raising and has_fist:
                            self.fist_confirm_count += 1
                        else:
                            self.fist_confirm_count = 0
                            self.fist_hold_start = None

                        if self.fist_confirm_count > 3:
                            if self.fist_hold_start is None: self.fist_hold_start = curr_time
                            ho_time = curr_time - self.fist_hold_start
                            cv2.putText(annotated_frame, f"HUY LENH: {ho_time:.1f}s/2s", (int(x1), int(y1)-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            if ho_time >= 2.0:
                                self.status_update_signal.emit("CANCEL_ALL")
                                self.locked_target_id = None
                                self.robot_state = "IDLE"
                                
                    is_too_close = (0 < d_ngang_m < 1.0)
                    is_invalid = (d_ngang_m <= 0.0 or d_ngang_m > 5.0)
                    
                    if self.locked_target_id is not None:
                        if track_id == self.locked_target_id:
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 3)
                            cv2.putText(annotated_frame, "LOCKED TARGET", (int(x1), int(y1)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        if not is_invalid:
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            if self.locked_target_id is not None:
                detected_ids = [int(box.id[0].item()) for r in results_pose if r.boxes and r.boxes.id is not None for box in r.boxes]
                if self.locked_target_id not in detected_ids:
                    if not hasattr(self, 'target_lost_time'):
                        self.target_lost_time = curr_time
                    elif curr_time - self.target_lost_time > 3.0:
                        if self.robot_state != "MOVING":
                            self.status_update_signal.emit("CANCEL_ALL")
                            self.locked_target_id = None
                            self.robot_state = "IDLE"
                        if hasattr(self, 'target_lost_time'):
                            del self.target_lost_time
                else:
                    if hasattr(self, 'target_lost_time'):
                        del self.target_lost_time

            cv2.putText(annotated_frame, f"STATE: {self.robot_state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) if self.robot_state != "IDLE" else (0,255,0), 2)
            self.change_pixmap_signal.emit(annotated_frame)
            
        self.pipeline.stop()

    def stop(self):
        self._run_flag = False
        self.wait()

# ================= Main App =================
class TestPCApp(QMainWindow):
    map_signal = pyqtSignal(object)
    pose_signal = pyqtSignal(float, float, float)

    def __init__(self):
        super().__init__()
        self.current_goal = None
        self.is_moving = False
        self.last_pose_print_time = 0  # Thêm biến đếm thời gian in tọa độ
        self.setWindowTitle("TEST SMART DOCKING (Dynamic Radius & Target Exclusion)")
        self.resize(1600, 900)
        
        self.central_widget = QWidget()
        self.layout = QHBoxLayout(self.central_widget)
        
        self.left_panel = QVBoxLayout()
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.left_panel.addWidget(self.camera_label, 1)
        
        self.btn_scan = QPushButton("BẮT ĐẦU SCAN NGƯỜI")
        self.btn_scan.setMinimumHeight(50)
        self.btn_scan.clicked.connect(self.start_scanning)
        self.left_panel.addWidget(self.btn_scan)
        
        self.layout.addLayout(self.left_panel, 1)
        
        self.map_label = MapLabel()
        self.map_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.map_label, 1)
        
        self.setCentralWidget(self.central_widget)

        rospy.init_node('test_smart_raycast', anonymous=True)

        self.robot = nav.ws_connect()
        self.mir_headers = nav.api_login()

        self.map_signal.connect(self.map_label.set_map)
        self.pose_signal.connect(self.map_label.set_robot_pose)
        
        # Kết nối tín hiệu tính toán
        self.map_label.clicked_signal.connect(self.calculate_hybrid_safe_goal)

        self.video_thread = VideoThread()
        self.video_thread.change_pixmap_signal.connect(self.update_camera_image)
        self.video_thread.target_locked_signal.connect(self.calculate_hybrid_safe_goal)
        self.video_thread.status_update_signal.connect(self.handle_status_update)
        self.video_thread.start()

        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        rospy.Subscriber('/robot_pose', Pose, self.pose_callback)

    def start_scanning(self):
        print("[TEST] Đang mở quét tay...")
        pass # self.video_thread.is_scanning = True is no longer used
        
    def handle_status_update(self, status):
        if status == "CANCEL_ALL":
            self.cancel_all()
        else:
            print(f"[STATUS] {status}")
            
    def cancel_all(self):
        print("[NAV] Hủy toàn bộ lệnh di chuyển!")
        self.is_moving = False
        self.current_goal = None
        try:
            import requests
            if hasattr(self, 'mir_headers') and self.mir_headers:
                requests.delete(f"http://192.168.0.177/api/v2.0.0/mission_queue", headers=self.mir_headers, timeout=2)
        except Exception as e:
            print(f"Lỗi cancel_all: {e}")

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
        
        # IN TỌA ĐỘ HIỆN TẠI MỖI 3 GIÂY ĐỂ USER BIẾT VỊ TRÍ KHI LÁI TAY
        curr_time = time.time()
        if curr_time - self.last_pose_print_time > 3.0:
            print(f"[ROBOT POSE] Xe đang đứng tại: X = {msg.position.x:.2f}, Y = {msg.position.y:.2f}, Góc = {math.degrees(yaw):.1f}°")
            self.last_pose_print_time = curr_time
        
        if self.is_moving and self.current_goal:
            gx, gy = self.current_goal
            dist = math.hypot(msg.position.x - gx, msg.position.y - gy)
            if dist < 0.2: # Ngưỡng để xác định là đã tới (dưới 20cm)
                print("\n====================================")
                print("====== 🏆 TỚI ĐÍCH RỒI !!! ========")
                print("====================================\n")
                self.is_moving = False
                self.current_goal = None

    # ---------------- THUẬT TOÁN LAI HYBRID MỚI (Dynamic Radius + Target Exclusion) ----------------
    def calculate_hybrid_safe_goal(self, target_x, target_y, obs_pt_map=None):
        if not self.map_label.map_info or self.map_label.robot_px is None:
            return
            
        res = self.map_label.map_info.resolution
        ox = self.map_label.map_info.origin.position.x
        oy = self.map_label.map_info.origin.position.y
        w = self.map_label.map_info.width
        h = self.map_label.map_info.height
        
        px_t = int((target_x - ox) / res)
        py_t = int((target_y - oy) / res)
        
        px_r = self.map_label.robot_px[0]
        py_r = h - self.map_label.robot_px[1] - 1
        
        if not (0 <= px_t < w and 0 <= py_t < h):
            rospy.logwarn("[GEOM] Điểm đích vượt giới hạn bản đồ!")
            return

        # HIỂN THỊ NGAY LẬP TỨC ĐIỂM CUSTOMER LÊN GIAO DIỆN VÀ TERMINAL
        print(f"\n[TARGET LOCKED] Đã lấy được tọa độ Customer: X = {target_x:.2f}, Y = {target_y:.2f}")
        self.map_label.target_px = (int(px_t), h - int(py_t) - 1)
        self.map_label.update_view()

        # TẠO LƯỚI TỔNG HỢP GLOBAL TỪ BẢN ĐỒ 2D
        # QUAN TRỌNG: Phải coi vùng Unknown (-1) là vật cản (255) để tia quét không xuyên qua lòng bàn!
        obs_mask = np.where((self.map_label.map_data != 0), 255, 0).astype(np.uint8)
        combined_obs = obs_mask.copy()
        
        # 2. THÊM VẬT CẢN 3D LƠ LỬNG (NẾU CÓ)
        if obs_pt_map:
            obs_px_x = int((obs_pt_map[0] - ox) / res)
            obs_px_y = int((obs_pt_map[1] - oy) / res)
            self.map_label.obs3d_px = (obs_px_x, h - obs_px_y - 1)
            
            if 0 <= obs_px_x < w and 0 <= obs_px_y < h:
                radius_px = int(0.15 / res) # Vật cản 3D lơ lửng bán kính 15cm
                cv2.circle(combined_obs, (obs_px_x, obs_px_y), radius_px, 255, -1)
        else:
            self.map_label.obs3d_px = None

        # 3. BƠM PHỒNG VẬT CẢN (THÂN XE 0.4m + 0.05m LỀ = 0.45m)
        safe_radius_px = int(0.45 / res)
        kernel_safe = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (safe_radius_px*2+1, safe_radius_px*2+1))
        inflated_obs = cv2.dilate(combined_obs, kernel_safe)
        
        # === THUẬT TOÁN RAYCAST ĐỖ CHÉO (PLAN C) ===
        min_dist_normal = float('inf')
        theta_normal = 0.0
        max_ray_len = int(3.0 / res)
        
        self.map_label.cone_pixels = []
        
        # Bước 1: Quét 360 độ tìm Hướng Trực Diện (Pháp tuyến - Normal)
        # SỬ DỤNG THUẬT TOÁN HYBRID (Kết hợp cũ và mới)
        ray_distances = []
        
        # Kiểm tra xem khách hàng đang đứng sát bàn (trong vùng inflated_obs) hay đứng ở vùng trống
        is_inside_obs = False
        if 0 <= px_t < w and 0 <= py_t < h:
            if inflated_obs[int(py_t), int(px_t)] > 0:
                is_inside_obs = True
                
        if is_inside_obs:
            # PHƯƠNG PHÁP CŨ: Đi tìm lối thoát ra không gian trống gần nhất
            for angle in range(0, 360, 5):
                rad = math.radians(angle)
                dist = float('inf')
                for step in range(1, max_ray_len):
                    cx = int(px_t + step * math.cos(rad))
                    cy = int(py_t + step * math.sin(rad))
                    if not (0 <= cx < w and 0 <= cy < h): break
                    self.map_label.cone_pixels.append((cx, h - cy - 1))
                    if inflated_obs[cy, cx] == 0: # Tìm khoảng trống
                        dist = step
                        break
                if dist != float('inf'): ray_distances.append((rad, dist))
                
            if ray_distances:
                min_dist_normal = min(d for r, d in ray_distances)
                valid_angles = [r for r, d in ray_distances if d <= min_dist_normal + 2]
                sum_x = sum(math.cos(r) for r in valid_angles)
                sum_y = sum(math.sin(r) for r in valid_angles)
                theta_normal = math.atan2(sum_y, sum_x)
            else:
                theta_normal = 0.0
                
        else:
            # PHƯƠNG PHÁP MỚI: Khách ở vùng trống, đi tìm vật cản gần nhất để làm mốc
            for angle in range(0, 360, 5):
                rad = math.radians(angle)
                dist = float('inf')
                for step in range(1, max_ray_len):
                    cx = int(px_t + step * math.cos(rad))
                    cy = int(py_t + step * math.sin(rad))
                    if not (0 <= cx < w and 0 <= cy < h): break
                    self.map_label.cone_pixels.append((cx, h - cy - 1))
                    if combined_obs[cy, cx] > 0: # Tìm vật cản (lõi bàn)
                        dist = step
                        break
                if dist != float('inf'): ray_distances.append((rad, dist))
                
            if ray_distances:
                min_dist_obs = min(d for r, d in ray_distances)
                valid_angles = [r for r, d in ray_distances if d <= min_dist_obs + 2]
                sum_x = sum(math.cos(r) for r in valid_angles)
                sum_y = sum(math.sin(r) for r in valid_angles)
                theta_obs = math.atan2(sum_y, sum_x)
                theta_normal = theta_obs + math.pi # Lật ngược hướng bàn thành hướng trống
            else:
                theta_normal = 0.0
            
        # Bước 2 & 3: Quét không gian để tìm bên nào thoáng hơn (Dựa vào lõi bàn)
        # Quét góc phần tư phía sau bên Trái và phía sau bên Phải để đếm mật độ bàn
        obs_left = 0
        obs_right = 0
        for step in range(1, int(1.5 / res)): # Quét xa 1.5m
            for offset_deg in range(90, 180, 5):
                # Bên Trái
                rad_l = theta_normal + math.radians(offset_deg)
                cx_l = int(px_t + step * math.cos(rad_l))
                cy_l = int(py_t + step * math.sin(rad_l))
                if 0 <= cx_l < w and 0 <= cy_l < h and combined_obs[cy_l, cx_l] > 0:
                    obs_left += 1
                
                # Bên Phải
                rad_r = theta_normal - math.radians(offset_deg)
                cx_r = int(px_t + step * math.cos(rad_r))
                cy_r = int(py_t + step * math.sin(rad_r))
                if 0 <= cx_r < w and 0 <= cy_r < h and combined_obs[cy_r, cx_r] > 0:
                    obs_right += 1
                    
        if obs_left > obs_right:
            # Bàn nằm nhiều ở bên Trái -> Bên Phải thoáng hơn -> Đỗ ra khoảng không bên Phải
            theta_dock = theta_normal - math.radians(45)
            print(f"[SMART NAV] ↪️ Không gian PHẢI thoáng hơn (L={obs_left}, R={obs_right}). Đỗ chéo ra PHẢI (góc 45 độ).")
        else:
            # Bàn nằm nhiều ở bên Phải -> Bên Trái thoáng hơn -> Đỗ ra khoảng không bên Trái
            theta_dock = theta_normal + math.radians(45)
            print(f"[SMART NAV] ↪️ Không gian TRÁI thoáng hơn (L={obs_left}, R={obs_right}). Đỗ chéo ra TRÁI (góc 45 độ).")
            
        # Bước 4: Tính toán target_dist_m động dựa trên khoảng trống (lidar map)
        free_start_step = None
        free_end_step = None
        self.map_label.ray_pixels = []
        
        for step in range(1, max_ray_len): # Quét tối đa 3.0m
            cx = int(px_t + step * math.cos(theta_dock))
            cy = int(py_t + step * math.sin(theta_dock))
            
            if not (0 <= cx < w and 0 <= cy < h):
                if free_start_step is not None and free_end_step is None:
                    free_end_step = step
                break
                
            self.map_label.ray_pixels.append((cx, h - cy - 1))
            
            if inflated_obs[cy, cx] == 0: # Không có vật cản
                if free_start_step is None:
                    free_start_step = step
            else: # Gặp vật cản khác phía sau
                if free_start_step is not None:
                    free_end_step = step
                    break
                    
        best_pose_px = None
        goal_yaw = 0.0
        
        if free_start_step is not None:
            if free_end_step is None:
                free_end_step = max_ray_len
                
            # Tính toán tâm của khoảng trống để robot đỗ cách xa cả khách và vật cản sau lưng
            target_step = int((free_start_step + free_end_step) / 2)
            target_dist_m = target_step * res
            
            # SỬA LỖI ĐỖ XA: Giữ nguyên khoảng cách 0.50m - 0.65m tính từ ĐIỂM CUSTOMER
            if target_dist_m < 0.50:
                target_step = int(0.50 / res)
            elif target_dist_m > 0.65:
                target_step = int(0.65 / res)
                
            # Đảm bảo target_step không rơi vào vật cản
            if target_step >= free_end_step:
                target_step = max(free_start_step, free_end_step - 1)
            if target_step < free_start_step:
                target_step = free_start_step
                
            best_pose_px = (
                int(px_t + target_step * math.cos(theta_dock)),
                int(py_t + target_step * math.sin(theta_dock))
            )
            
            goal_yaw = math.atan2(py_t - best_pose_px[1], px_t - best_pose_px[0])
            actual_dist = math.hypot(px_t - best_pose_px[0], py_t - best_pose_px[1])
            print(f"[SMART NAV] ✅ Đã chốt điểm đỗ chéo ở vùng trống an toàn, cách khách {actual_dist * res:.2f}m")
        else:
            print("[SMART NAV] ❌ THẤT BẠI: Bị kẹt trên tia chéo, không thể tìm thấy chỗ lách vào an toàn!")
            self.map_label.target_px = (int(px_t), h - int(py_t) - 1)
            self.map_label.goal_px = None
            self.map_label.update_view()
            return
                
        # 5. GỬI LỆNH ĐIỀU HƯỚNG
            
        final_px_x, final_px_y = best_pose_px
        goal_w_x = ox + final_px_x * res
        goal_w_y = oy + final_px_y * res
        
        # Cập nhật GUI (Chuyển ngược tọa độ Y lên GUI)
        self.map_label.target_px = (int(px_t), h - int(py_t) - 1)
        self.map_label.goal_yaw = goal_yaw
        self.map_label.goal_px = (int(final_px_x), h - int(final_px_y) - 1)
        self.map_label.update_view()
        
        print(f"[SMART NAV] 🎯 Tọa độ đỗ cuối cùng (X,Y) = ({goal_w_x:.2f}, {goal_w_y:.2f}), Hướng Yaw = {math.degrees(goal_yaw):.1f}°")
        
        q = tf.transformations.quaternion_from_euler(0, 0, goal_yaw)
        diem_dong = {"x": goal_w_x, "y": goal_w_y, "qz": q[2], "qw": q[3], "arrive_dist": 0.15}
        
        print(f"🚀 [NAV] Bắn lệnh tới MiR Fleet / MoveBase!")
        
        # Bắt đầu theo dõi hành trình để thông báo khi tới nơi
        self.current_goal = (goal_w_x, goal_w_y)
        self.is_moving = True
        
        rest_ok = False
        if hasattr(self, 'mir_headers') and self.mir_headers:
            rest_ok = nav.api_navigate(self.mir_headers, diem_dong, "diem_dong")
        if not rest_ok and self.robot:
            nav.ws_send_goal(self.robot, diem_dong)

    def closeEvent(self, event):
        print("[INFO] Đang đóng luồng Camera an toàn...")
        self.video_thread.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = TestPCApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
