import cv2
import time
import math
import threading
import numpy as np
import rospy

try:
    import pyrealsense2 as rs
    import mediapipe as mp
    from ultralytics import YOLO
except ImportError:
    print("❌ Vẫn thiếu thư viện AI, vui lòng kiểm tra môi trường")

import config
from gui import signal_bus

def get_depth_distance_m(depth_frame, box, frame_w, frame_h):
    """Tính toán khoảng cách trung bình từ camera đến vùng tâm ngực của mục tiêu (Theo mét)"""
    x1, y1, x2, y2 = map(int, box)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    roi_size = int(max(8, min(x2 - x1, y2 - y1) // 4)) # Lấy một vùng hình vuông kích thước động ở giữa tâm
    distances = []
    for dx in range(-roi_size, roi_size + 1, 8):
        for dy in range(-roi_size, roi_size + 1, 8):
            px, py = center_x + dx, center_y + dy
            orig_px = frame_w - 1 - px # Phục hồi tạo độ lật ngược
            if 0 <= orig_px < frame_w and 0 <= py < frame_h:
                d = depth_frame.get_distance(orig_px, py)
                if 0.3 < d < 6.0: distances.append(d) # Lọc nhiễu >6m hoặc <0.3m
    return float(np.median(distances)) if distances else -1.0

def get_person_relative_position_m(depth_frame, box, frame_w, frame_h, depth_intrinsics, distance_m):
    """Quy đổi tọa độ Pixel màn hình sang tọa độ thực tế (Tiến bao nhiêu mét, lệch xang trái bao nhiêu mét)"""
    x1, y1, x2, y2 = map(int, box)
    center_x = (x1 + x2) // 2
    orig_px = frame_w - 1 - center_x
    if distance_m <= 0: return None
    if depth_intrinsics is None:
        # Giả lập FOV 69 độ nếu không lấy được tham số thật từ Realsense
        hfov_rad = math.radians(69.0)
        angle = ((orig_px - frame_w / 2.0) / frame_w) * hfov_rad
        x_cam = distance_m * math.tan(angle)
    else:
        x_cam = (orig_px - depth_intrinsics.ppx) / depth_intrinsics.fx * distance_m
    return (distance_m, -x_cam) # (forward_m, left_m)

class VisionTracker:
    def __init__(self, on_lock_callback=None, on_unlock_callback=None):
        self.model = YOLO(config.YOLO_MODEL_PATH)
        
        # 1. Khởi động Camera RealSense
        self.pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        self.camera_ready = False
        self.use_webcam_fallback = False
        self.webcam = None
        try:
            self.pipeline.start(rs_config)
            self.camera_ready = True
            # Sử dụng align để ép điểm ảnh Độ sâu vừa vặn với Màu sắc
            self.align = rs.align(rs.stream.color) 
            
            profile = self.pipeline.get_active_profile()
            depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            self.depth_intrinsics = depth_profile.get_intrinsics()
        
        except RuntimeError as e:
            rospy.logerr(f"❌ KHÔNG TÌM THẤY CAMERA REALSENSE. Chuyển sang dùng WEBCAM Laptop!")
            self.depth_intrinsics = None
            self.use_webcam_fallback = True
            self.webcam = cv2.VideoCapture(0) # 0, 1 hoặc 2 (tuỳ webcam) có thể đổi nếu lap có ảo
            if self.webcam.isOpened():
                self.camera_ready = True
            else:
                rospy.logerr("❌ KHÔNG MỞ ĐƯỢC WEBCAM LUÔN!")


        # 2. Khởi tạo thuật toán nhận diện gân/xương bàn tay
        mp_hands = mp.solutions.hands
        self.hands_detector = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.locked_target_id = None
        self.hand_raise_start = {}
        self.open5_confirm_count = {}
        self.fist_hold_start = None
        self.fist_confirm_count = 0

        # Sự kiện Callback khi AI dò ra cử chỉ
        self.on_lock_callback = on_lock_callback
        self.on_unlock_callback = on_unlock_callback

        # Đẩy quá trình xử lý ảnh ra luồng (Thread) phụ để không chết GUI
        self.worker_thread = threading.Thread(target=self.run, daemon=True)
        self.worker_thread.start()

    def run(self):
        if not self.camera_ready:
            signal_bus.status_update.emit("Lỗi: Không kết nối Camera 3D! Vẫn chờ /map")
        
        while not rospy.is_shutdown():
            if not self.camera_ready:
                rospy.sleep(1)
                continue
            
            # --- Lấy Ảnh ---
            try:
                if hasattr(self, "use_webcam_fallback") and self.use_webcam_fallback:
                    ret, frame = self.webcam.read()
                    if not ret: continue
                    depth_frame = None
                else:
                    frames = self.pipeline.wait_for_frames()
                    aligned = self.align.process(frames)
                    depth_frame = aligned.get_depth_frame()
                    color_frame = aligned.get_color_frame()
                    if not depth_frame or not color_frame: continue
                    frame = np.asanyarray(color_frame.get_data())
            except Exception:
                continue
            frame = cv2.flip(frame, 1) # Lật như gương soi
            frame_h, frame_w = frame.shape[:2]

            # --- A. MEDIAPIPE: Tìm & Xác định Cử Chỉ Tay ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = self.hands_detector.process(rgb_frame)
            detected_hands = []

            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    def get_dist(i1, i2):
                        p1, p2 = hand_landmarks.landmark[i1], hand_landmarks.landmark[i2]
                        return math.hypot(p1.x - p2.x, p1.y - p2.y)
                    
                    # Logic so sánh độ dài khớp ngón (pip) với ngón tay (tip)
                    tip_ids = [4, 8, 12, 16, 20]; pip_ids = [2, 6, 10, 14, 18]
                    tip_dists = [get_dist(t, 0) for t in tip_ids]
                    pip_dists = [get_dist(p, 0) for p in pip_ids]
                    
                    fingers = sum(1 for td, pd in zip(tip_dists, pip_dists) if td > pd)
                    all_ext = all(td >= 1.3 * pd for td, pd in zip(tip_dists, pip_dists)) # Duỗi thẳng
                    thumb_sp = get_dist(4, 8) > 0.45 * max(1e-6, get_dist(5, 17)) # Ngón cái tách khỏi cụm
                    open5_strict = (fingers == 5) and all_ext and thumb_sp

                    wrist = hand_landmarks.landmark[0]
                    hx, hy = int(wrist.x * frame_w), int(wrist.y * frame_h)
                    detected_hands.append((hx, hy, fingers, open5_strict))

            # --- B. YOLO-POSE: Nhận diện người, Khóa ID và Cử chỉ cơ thể ---
            results = self.model.track(frame, conf=0.45, persist=True, tracker="bytetrack.yaml", verbose=False)
            curr_time = time.time()
            
            for result in results:
                if result.boxes is None: continue
                boxes = result.boxes
                keypoints = getattr(result, "keypoints", None)

                for i, box in enumerate(boxes):
                    cls = int(box.cls[0].cpu().item())
                    if cls != 0: continue # Chỉ quan tâm (class 0: Gương mặt/Người)
                    
                    track_id = int(box.id[0].item()) if box.id is not None else -1
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    if hasattr(self, "use_webcam_fallback") and self.use_webcam_fallback:
                        d_m = 1.5
                    else:
                        d_m = get_depth_distance_m(depth_frame, (x1, y1, x2, y2), frame_w, frame_h)
                    
                    delta_h = 1.8 - 0.6
                    d_ngang_m = math.sqrt(d_m**2 - delta_h**2) if d_m > delta_h else d_m

                    # Phân tích Keypoints để xem tay có vượt qua bả vai chưa
                    is_raising = False
                    if keypoints and keypoints.data is not None and i < len(keypoints.data):
                        kpts = keypoints.data[i].cpu().numpy()
                        if len(kpts) >= 11:
                            ls, rs, lw, rw = kpts[5], kpts[6], kpts[9], kpts[10] # Bả vai, Cổ Tay
                            def v_kpt(k): return k[2]>0.4 if len(k)>=3 else (k[0]>0 and k[1]>0)
                            if v_kpt(ls) and v_kpt(lw) and lw[1] < ls[1]: is_raising = True
                            if v_kpt(rs) and v_kpt(rw) and rw[1] < rs[1]: is_raising = True

                    has_open_five = False
                    has_fist = False
                    open5_flags = []
                    fing_for_unlock = []
                    
                    # Ghép đôi Bàn Tay (MediaPipe) vào trong Hộp Người (YOLO)
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
                    elif is_raising: has_fist = True

                    # --- LOGIC GẮN KHÓA (LOCK TARGET) ---
                    # Nếu chưa khóa ai, và thấy có người giơ tay chĩa 5 ngón bám rịt 5 giây
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
                                signal_bus.status_update.emit(f"ĐÃ KHÓA ! Đang tải dữ liệu không gian...")
                                
                                rel_coords = get_person_relative_position_m(
                                    depth_frame, (x1, y1, x2, y2), frame_w, frame_h, self.depth_intrinsics, d_m)
                                if rel_coords and self.on_lock_callback:
                                    # KÍCH HOẠT SỰ KIỆN QUA MAIN.PY
                                    self.on_lock_callback(rel_coords[0], rel_coords[1])

                    # --- LOGIC MỞ KHÓA (UNLOCK TARGET - Bằng Năm đấm) ---
                    # Nếu đã khóa mục tiêu, và mục tiêu giơ nắm đấm suốt 5 giây
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
                                self.locked_target_id = None
                                if self.on_unlock_callback:
                                    self.on_unlock_callback()
                                signal_bus.status_update.emit(f"Trạng thái: Đang theo dõi người dùng")

                    # --- VẼ HÌNH OSD VÀO ẢNH CAMERA ---
                    is_too_close = d_ngang_m < 1.0 # Nguy hiểm nếu nhỏ hơn 1m
                    if self.locked_target_id is not None:
                        if track_id == self.locked_target_id:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 3) # Vàng - Đang Khóa
                            cv2.putText(frame, "LOCKED TARGET", (int(x1), int(y1)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            dist_str = f"Khoang cach: {d_ngang_m:.2f}m"
                            txt_color = (0, 0, 255) if is_too_close else (0, 255, 255)
                            if is_too_close: dist_str += " (QUA GAN - KHONG DI)"
                            cv2.putText(frame, dist_str, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 2)
                        else:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (100, 100, 100), 1) # Xám - Bơ đi
                            cv2.putText(frame, f"{d_ngang_m:.2f}m", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
                    else:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2) # Xanh lá - Idle chưa bám ID nào
                        dist_str = f"{d_ngang_m:.2f}m"
                        txt_color = (0, 0, 255) if is_too_close else (0, 255, 0)
                        if is_too_close: dist_str += " (Qua gan)"
                        cv2.putText(frame, dist_str, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 2)

            # --- KIỂM TRA MẤT DẤU NGƯỜI DÙNG QUÁ LÂU ---
            if self.locked_target_id is not None:
                detected_ids = [int(box.id[0].item()) for r in results if r.boxes and r.boxes.id is not None for box in r.boxes]
                if self.locked_target_id not in detected_ids:
                    if not hasattr(self, 'target_lost_time'):
                        self.target_lost_time = curr_time
                    elif curr_time - self.target_lost_time > 3.0: # Quá 3 giây không thấy bộ xương đâu
                        self.locked_target_id = None
                        if self.on_unlock_callback:
                            self.on_unlock_callback()
                        signal_bus.status_update.emit("Mất dấu mục tiêu > 3s! Đã tự mở khóa.")
                        del self.target_lost_time
                else:
                    if hasattr(self, 'target_lost_time'):
                        del self.target_lost_time

            # Đẩy khung hình đã vẽ vời xong ra GUI
            signal_bus.frame_update.emit(frame)
