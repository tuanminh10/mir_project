#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

# ==============================================================================
# HACK HỖ TRỢ GPU RTX 5060 (Blackwell - sm_120) TRÊN ROS NOETIC (PYTHON 3.8)
# TRÁNH LỖI "no kernel image is available for execution on the device"
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
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import time

import math
import pyrealsense2 as rs

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
from geometry_msgs.msg import PointStamped

def get_depth_distance_m(depth_frame, box, frame_w, frame_h):
    x1, y1, x2, y2 = map(int, box)
    
    # CHUYÊN TRỊ ĐỐI TƯỢNG NGỒI GHẾ, XOAY NGƯỜI, NGHIÊNG NGƯỜI
    # Thay vì lấy 1 điểm tâm, ta cắt một vùng không gian (ROI) tập trung vào phần thân trên (Đầu, Vai, Ngực)
    width = x2 - x1
    height = y2 - y1
    
    # Bỏ 15% hai bên rìa để tránh lẹm vào lưng ghế hoặc tường
    roi_x1 = int(x1 + width * 0.15)
    roi_x2 = int(x2 - width * 0.15)
    
    # Lấy từ 10% (tránh đỉnh tóc/nhiễu viền) đến 60% (vùng bụng/ngực, tránh đùi hoặc mặt bàn)
    roi_y1 = int(y1 + height * 0.10)
    roi_y2 = int(y1 + height * 0.60)
    
    distances = []
    
    # Quét mạng lưới 10x10 điểm ảnh trong vùng thân trên này
    step_x = max(1, (roi_x2 - roi_x1) // 10)
    step_y = max(1, (roi_y2 - roi_y1) // 10)
    
    for px in range(roi_x1, roi_x2 + 1, step_x):
        for py in range(roi_y1, roi_y2 + 1, step_y):
            if 0 <= px < frame_w and 0 <= py < frame_h:
                d = depth_frame.get_distance(px, py)
                # Lọc bỏ các giá trị nhiễu hoặc quá xa/quá gần
                if 0.3 < d < 6.0: 
                    distances.append(d)
                
    if not distances:
        return -1.0
        
    # Phân vị 30% (30th percentile): Thuật toán vàng cho nhận diện người ngồi!
    # - Bỏ qua 30% điểm cực gần (tay huơ ra trước, mép bàn, nhiễu bụi)
    # - Bắt trúng khối cơ thể người (vì cơ thể là vật thể to thứ hai từ gần tới xa)
    # - Loại bỏ hoàn toàn 50-70% khoảng cách xa (lưng ghế, bức tường)
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

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        
        # CHỐNG GIẬT: Đổi xuống mô hình YOLO11n (Nano) siêu nhẹ, cho tốc độ 60+ FPS
        # Việc tăng khung hình sẽ giúp Tracker (ByteTrack) chạy chính xác hơn, không bị nhảy loạn ID
        print("[INFO] Đang tải mô hình cảnh báo người YOLO11n (Siêu Tốc Độ)...")
        self.model = YOLO("/home/tuanminh/mir_project/yolo11n.pt") 
        print("[INFO] Tải YOLO thành công!")

        # Khởi tạo MediaPipe Hand để nhận diện tay
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7, # Tăng độ tin cậy để chống nhận diện loạn tay ảo
            min_tracking_confidence=0.5
        )
        print("[INFO] Khởi tạo MediaPipe Hands thành công!")

        # Trạng thái Lock Target
        self.locked_track_id = None
        self.lost_target_start_time = 0  # Đếm thời gian mục tiêu bị mất dấu
        self.dist_history = []  # Bộ lọc chống nhiễu khoảng cách
        
        # Biến đếm thời gian cho tay
        self.target_candidate_id = None
        self.open_hand_start_time = 0
        
        self.fist_start_time = 0

    def is_hand_open(self, hand_landmarks):
        """Kiểm tra tay xoè (các ngón duỗi thẳng)"""
        # Đếm số ngón tay đang mở. 
        # Tip (đầu ngón) nằm cao hơn (y nhỏ hơn) Pip (khớp giữa)
        open_fingers = 0
        # Index, Middle, Ring, Pinky
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                open_fingers += 1
        
        # Thumb (Ngón cái) tính x thay vì y cho đơn giản
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x: # Có thể sai tuỳ tay trái phải, nhưng thường thumb duỗi là xè
            open_fingers += 1
            
        return open_fingers >= 4

    def is_hand_fist(self, hand_landmarks):
        """Kiểm tra tay nắm đấm (các ngón gập lại)"""
        closed_fingers = 0
        # Index, Middle, Ring, Pinky
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y:
                closed_fingers += 1
        return closed_fingers >= 4

        # Hệ thống Tracking DeepSORT/ByteTrack kết hợp NMS
        # model.predict hỗ trợ tích hợp tracker để gỡ rối các bbox bị chồng chéo (Occlusion)

    def run(self):
        # KHỞI TẠO REALSENSE THAY CHO WEBCAM
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

        # Vòng lặp đọc và phân tích khung hình (Chạy trên luồng phụ để chống giật)
        while self._run_flag:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            except Exception:
                continue
                
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())

            # Chạy inference kèm TRACKER (ByteTrack)
            # Dùng conf=0.45 để loại rác nền (do ảnh nhỏ đi nên mô hình nhạy hơn)
            results = self.model.track(
                frame, 
                classes=[0], 
                conf=0.45, 
                iou=0.6, 
                persist=True, 
                tracker="bytetrack.yaml", 
                verbose=False
            )

            # Phân tích MediaPipe Hands trên toàn khung hình
            # Cần chuyển sang RGB cho MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = self.hands.process(rgb_frame)

            annotated_frame = frame.copy()
            current_time = time.time()

            # Nếu có target được lock, gom tất cả người lại
            # Chỉ tìm người khớp track_id
            
            # Cấu trúc lưu trạng thái người (box, xyxy)
            people = []
            
            if results[0].boxes and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    people.append({
                        "id": track_id,
                        "box": (x1, y1, x2, y2)
                    })

            # Map xem bàn tay thuộc về người nào (bbox người chứa tay)
            hand_owner_id = None
            hand_state = None  # "open" hoặc "fist"

            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    # Lấy toạ độ điểm trung tâm tay (ví dụ cổ tay hoặc đốt ngón)
                    wrist = hand_landmarks.landmark[0]
                    h, w, _ = frame.shape
                    hx, hy = int(wrist.x * w), int(wrist.y * h)

                    # Kiểm tra trạng thái
                    if self.is_hand_open(hand_landmarks):
                        hand_state = "open"
                    elif self.is_hand_fist(hand_landmarks):
                        hand_state = "fist"

                    # Tìm người có chứa bàn tay này và thoả mãn điều kiện đang GIƠ TAY CAO LÊN ĐẦU
                    for person in people:
                        x1, y1, x2, y2 = person["box"]
                        # Tay nằm trong khoảng chiều rộng của người (x1 -> x2)
                        # VÀ tay phải giơ cao: ở quanh hoặc vượt mức đỉnh đầu (hy có thể cao hơn y1 một chút, hoặc nằm trong 25% phía trên của cơ thể)
                        if (x1 - 30) <= hx <= (x2 + 30) and hy < y1 + (y2 - y1) * 0.25:
                            hand_owner_id = person["id"]
                            break
                    
                    if hand_owner_id is not None:
                        # (Đã ẩn) Nhận diện tay chạy ngầm, không vẽ khung xương tay hay hiển thị state lên GUI dể nhìn gọn gàng hơn
                        break # Xử lý 1 tay trước

            # Logic Lock / Unlock
            if hand_owner_id is not None:
                # NẾU chưa lock ai, kiểm tra gesture OPEN (xoè tay)
                if self.locked_track_id is None:
                    if hand_state == "open":
                        if self.target_candidate_id != hand_owner_id:
                            # Đổi candidate
                            self.target_candidate_id = hand_owner_id
                            self.open_hand_start_time = current_time
                        else:
                            # Cùng ứng viên, ktra thời gian 3s
                            if current_time - self.open_hand_start_time >= 3.0:
                                self.locked_track_id = hand_owner_id
                                self.dist_history.clear()
                                print(f"[TARGET] ĐÃ KHÓA MỤC TIÊU ID {self.locked_track_id}")
                                # reset
                                self.target_candidate_id = None
                    else:
                        # Reset nếu không duy trì xoè tay
                        self.target_candidate_id = None
                        self.open_hand_start_time = current_time

                # NẾU đã lock chính người này, kiểm tra gesture FIST (nắm đấm)
                elif self.locked_track_id == hand_owner_id:
                    if hand_state == "fist":
                        # Cảm nhận nắm đấm
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
                # Không thấy tay ai hoặc đổi trạng thái, reset biến hold
                self.target_candidate_id = None
                self.fist_start_time = 0
            
            # Reset fist_start_time nếu lock ID mất hand_owner, ngưng giơ nắm đấm
            if self.locked_track_id is not None and (hand_owner_id != self.locked_track_id or hand_state != "fist"):
                self.fist_start_time = 0

            # --- LOGIC CHUYỂN ĐỔI ID KHI MẤT DẤU (Chống đổi ID) ---
            # Tìm xem ID mục tiêu hiện tại có trong khung hình không
            is_target_in_frame = False
            if self.locked_track_id is not None:
                for p in people:
                    if p["id"] == self.locked_track_id:
                        is_target_in_frame = True
                        self.locked_bbox = p["box"]
                        break
                
                # Nếu ID cũ biến mất, nhưng có người khác xuất hiện ở gần vị trí đó -> cập nhật ID mới
                if not is_target_in_frame and hasattr(self, 'locked_bbox') and self.locked_bbox is not None and len(people) > 0:
                    lx1, ly1, lx2, ly2 = self.locked_bbox
                    lcx, lcy = (lx1 + lx2) / 2, (ly1 + ly2) / 2
                    
                    best_match = None
                    min_dist = float('inf')
                    
                    for p in people:
                        px1, py1, px2, py2 = p["box"]
                        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
                        dist = ((pcx - lcx)**2 + (pcy - lcy)**2)**0.5
                        
                        # Khoảng cách tối đa 150 pixel để được coi là cùng 1 người
                        if dist < 150 and dist < min_dist:
                            min_dist = dist
                            best_match = p
                            
                    if best_match is not None:
                        print(f"[TARGET] Đã cập nhật ID mục tiêu: {self.locked_track_id} -> {best_match['id']} (Do nhầm lẫn ID)")
                        self.locked_track_id = best_match["id"]
                        self.locked_bbox = best_match["box"]
                        # Không clear dist_history vì vẫn là người đó, giữ mượt mà
                        is_target_in_frame = True
                        self.lost_target_start_time = 0

            # Logic Auto-Unlock nếu mất dấu Target quá lâu (Glitch / Occlusion dài)
            if self.locked_track_id is not None:
                if is_target_in_frame:
                    self.lost_target_start_time = 0 # Reset bộ đếm mất dấu
                else:
                    if self.lost_target_start_time == 0:
                        self.lost_target_start_time = current_time
                    else:
                        lost_duration = current_time - self.lost_target_start_time
                        if lost_duration > 1.5:  # Nếu mất dấu quá 1.5 giây thì tự động mở khoá
                            print(f"[TARGET] Tự động huỷ khoá mục tiêu do mất dấu! ID: {self.locked_track_id}")
                            self.locked_track_id = None
                            self.dist_history.clear()
                            self.lost_target_start_time = 0


            # --- VẼ HÌNH THỊ THUẬT QUANH NGƯỜI ---
            for person in people:
                track_id = person["id"]
                x1, y1, x2, y2 = person["box"]
                
                # NGUYÊN TẮC VẼ: 
                # 1. Nếu đang có Locked ID: CHỈ vẽ người bị lock (màu xanh lá)
                # 2. Nếu đang không có: Vẽ tất cả người (màu đỏ)
                
                if self.locked_track_id is not None:
                    if track_id != self.locked_track_id:
                        continue # Bỏ qua người không bị lock
                    
                    # Vẽ Locked
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 4) # Xanh lá dày
                    cv2.putText(annotated_frame, f"LOCKED TARGET #{track_id}", (x1, max(20, y1-10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # --- TÍNH TOÁN KHOẢNG CÁCH VÀ TỌA ĐỘ MAP ---
                    d_m_raw = get_depth_distance_m(depth_frame, (x1, y1, x2, y2), 640, 480)
                    if d_m_raw > 0:
                        self.dist_history.append(d_m_raw)
                        if len(self.dist_history) > 7:
                            self.dist_history.pop(0)
                            
                        # Dùng trung vị lịch sử 7 frame để khử hoàn toàn nhiễu
                        d_m = float(np.median(self.dist_history))
                        
                        delta_h = 1.2
                        d_ngang_m = math.sqrt(d_m**2 - delta_h**2) if d_m > delta_h else d_m
                        
                        dist_text = f"Dist: {d_ngang_m:.2f}m"
                        cv2.putText(annotated_frame, dist_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        # FIX: Dùng khoảng cách ngang (d_ngang_m) để chiếu lên bản đồ thay vì cạnh huyền d_m
                        rel = get_person_relative_position_m((x1, y1, x2, y2), 640, self.depth_intrinsics, d_ngang_m)
                        if rel is not None:
                            forward_m, left_m = rel
                            forward_m -= 0.475 # Offset camera sau base_link
                            
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
                                
                                # IN RA TERMINAL MỖI 1 GIÂY
                                if current_time - getattr(self, 'last_coord_print_time', 0) >= 1.0:
                                    print(f"🎯 [TARGET INFO] Khoảng cách: {d_ngang_m:.2f}m | Tọa độ bản đồ: X={map_x:.2f}, Y={map_y:.2f}")
                                    self.last_coord_print_time = current_time
                                    
                            except Exception as e:
                                cv2.putText(annotated_frame, "Map: N/A (TF Error)", (x1, y2 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                if current_time - getattr(self, 'last_coord_print_time', 0) >= 1.0:
                                    print(f"⚠️ [TF ERROR] Không thể quy đổi toạ độ /map: {e}")
                                    self.last_coord_print_time = current_time
                    
                    # Cảnh báo huỷ nếu đang giơ nắm đấm
                    if self.fist_start_time > 0 and hand_owner_id == track_id and hand_state == "fist":
                        progress = current_time - self.fist_start_time
                        # Hiển thị số đếm timer 0 -> 3s thật TO RÕ ở giữa khu vực người
                        text = f"UNLOCKING: {progress:.1f}s"
                        cv2.putText(annotated_frame, text, (x1, int(y1 + (y2-y1)/2)), 
                                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)
                
                else: # Chưa lock
                    # Vẽ bình thường (Đỏ)
                    color = (0, 0, 255)
                    thickness = 2
                    label = f"Person #{track_id}"
                    
                    # Đổi màu/ghi chú nếu đang trong quá trình locking (Xoè tay)
                    if self.target_candidate_id == track_id and hand_state == "open" and hand_owner_id == track_id:
                        progress = current_time - self.open_hand_start_time
                        color = (0, 200, 255) # Vàng cam
                        thickness = 3
                        
                        # Hiển thị số đếm timer 0 -> 3s thật TO RÕ ở giữa khu vực người
                        text = f"LOCKING: {progress:.1f}s"
                        cv2.putText(annotated_frame, text, (x1, int(y1 + (y2-y1)/2)), 
                                cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 3)
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                    cv2.putText(annotated_frame, label, (x1, max(20, y1-10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Render UI Text trạng thái chung
            if self.locked_track_id:
                if self.lost_target_start_time > 0:
                    lost_prog = current_time - self.lost_target_start_time
                    mode_text = f"TARGET LOST! AUTO UNLOCK IN: {1.5 - lost_prog:.1f}s"
                    mode_color = (0, 165, 255) # Màu Cam
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

class HumanDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trực tiếp Camera - Phát hiện người (YOLO11m Cấu Hình Cao)")
        
        # Setup giao diện
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.label)

        # Khởi tạo Video Thread chống giật
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def update_image(self, cv_img):
        # Convert qua QImage để hiển thị trên PyQt label
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        
        qImg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        self.label.setPixmap(QPixmap.fromImage(qImg))
        # Không khoá resize tĩnh để đỡ chớp cửa sổ, chỉ scale lên khi cần
        if self.width() < w or self.height() < h:
            self.resize(w, h)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

def main():
    rospy.init_node('xlahu_node', anonymous=True, disable_signals=True)
    app = QApplication(sys.argv)
    window = HumanDetectorApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
