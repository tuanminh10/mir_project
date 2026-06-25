#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math
import time
import threading
import os
import pyrealsense2 as rs
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
import tf
import rospy
from geometry_msgs.msg import PointStamped
import config

def get_depth_distance_m(depth_frame, box, frame_w, frame_h, center_pt=None):
    x1, y1, x2, y2 = map(int, box)
    width = x2 - x1
    height = y2 - y1
    
    # 1. Bắt chước kien.py: Cắt ROI chỉ lấy ngực/đầu (né bàn ở dưới và tường ở 2 bên)
    roi_x1 = max(0, int(x1 + width * 0.20))
    roi_x2 = min(frame_w, int(x2 - width * 0.20))
    roi_y1 = max(0, int(y1 + height * 0.05))
    roi_y2 = min(frame_h, int(y1 + height * 0.45))
    
    if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
        return -1.0
        
    distances = []
    step_x = max(1, (roi_x2 - roi_x1) // 20)
    step_y = max(1, (roi_y2 - roi_y1) // 20)
    
    for px in range(roi_x1, roi_x2, step_x):
        for py in range(roi_y1, roi_y2, step_y):
            orig_px = frame_w - 1 - px # Lật trục X vì camera lật
            if 0 <= orig_px < frame_w and 0 <= py < frame_h:
                d = depth_frame.get_distance(orig_px, py)
                # Bỏ qua nhiễu quá gần (<0.3m) để tránh dính ngón tay chĩa sát camera
                if 0.3 < d < 6.0: distances.append(d)
    
    if not distances:
        return -1.0
        
    # 2. Giải thuật lọc Foreground của kien.py (Tinh chỉnh lại)
    d_arr = np.array(distances)
    
    # KHI NGỒI XA: Người chiếm diện tích rất nhỏ trong ROI.
    # Giải pháp: Hạ xuống 5% để đảm bảo bắt trúng thân người, không bao giờ bắt nhầm tường (nền sau).
    p5_Z = np.percentile(d_arr, 5)
    
    # Lấy các điểm dao động xung quanh ngực (cho sai số lùi 0.2m để bao trọn nếp áo)
    mask = (d_arr >= p5_Z - 0.2) & (d_arr <= p5_Z + 0.5)
    person_pts = d_arr[mask]
    
    if len(person_pts) < 5:
        person_pts = d_arr # Fallback an toàn nếu mask quá gắt
        
    return float(np.median(person_pts))

# Tích hợp Segmentation & 2D Radius Filter (Morphological Erosion)
def get_depth_distance_m_seg(depth_frame, binary_mask, frame_w, frame_h):
    # 1. Cắt gọt hình thái học (Morphological Erosion) - Tương đương Radius Outlier Removal 3D
    kernel = np.ones((7, 7), np.uint8) # Gọt mạnh viền (7x7) để tránh hoàn toàn nhiễu từ nền phía sau
    eroded_mask = cv2.erode(binary_mask, kernel, iterations=1)
    
    ys, xs = np.where(eroded_mask == 1)
    if len(ys) == 0:
        return -1.0
        
    distances = []
    # Lấy ngẫu nhiên tối đa 400 điểm để tối ưu tốc độ tính toán (Đạt > 60 FPS)
    step = max(1, len(ys) // 400)
    
    for i in range(0, len(ys), step):
        py = int(ys[i])
        px = int(xs[i])
        orig_px = frame_w - 1 - px # Lật trục X vì RGB đã bị cv2.flip(1)
        
        if 0 <= orig_px < frame_w and 0 <= py < frame_h:
            d = depth_frame.get_distance(orig_px, py)
            if 0.3 < d < 6.0: distances.append(d)
            
    if not distances: return -1.0
        
    d_arr = np.array(distances)
    p5_Z = np.percentile(d_arr, 5)
    
    mask = (d_arr >= p5_Z - 0.2) & (d_arr <= p5_Z + 0.5)
    person_pts = d_arr[mask]
    if len(person_pts) < 5: person_pts = d_arr
        
    return float(np.median(person_pts))
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

    pitch_rad = math.radians(config.CAMERA_PITCH_DEG)
    # CÔNG THỨC CHUẨN ĐÃ ĐƯỢC GIÁO SƯ CHỨNG MINH:
    forward_m_camera = z_opt * math.cos(pitch_rad) - y_opt * math.sin(pitch_rad)
    
    # BÙ TRỪ TRỤC QUANG HỌC VÀ VỊ TRÍ LẮP ĐẶT (Camera Calibration):
    lateral_offset_m = 0.0 # Giáo sư yêu cầu bỏ bù trái/phải
    forward_offset_m = 0.05 # Bù khung motor nhô ra trước 10cm (Từ đuôi lên đầu)
    
    forward_m = forward_m_camera + forward_offset_m
    left_m = -x_opt + lateral_offset_m
    
    # CÔNG THỨC CHUẨN Z:
    down_m = z_opt * math.sin(pitch_rad) + y_opt * math.cos(pitch_rad)
    camera_height_m = config.CAMERA_HEIGHT_M
    z_m = camera_height_m - down_m
    
    return forward_m, left_m, z_m


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    target_locked_signal = pyqtSignal(float, float, object)
    status_update_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.pause_emit = False
        self.device = 0 if os.path.exists('/opt/ai_venv/bin/python') else 'cpu'
        self.model_pose = YOLO('yolo11s-pose.pt')
        self.model_seg = YOLO('yolo11s-seg.pt')
        if self.device == 0:
            self.model_pose.to('cuda')
            self.model_seg.to('cuda')
            
        self.locked_target_id = None
        self.robot_state = "IDLE"
        self.locked_bbox = None
        
        self.hand_raise_start = {}
        self.open5_confirm_count = {}
        self.target_history = {} # Lưu đệm tọa độ trong 2s
        
        self.fist_confirm_count = 0
        self.fist_hold_start = None

        self.latest_rgb_frame = None
        self.frame_lock = threading.Lock()

    def run(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        try:
            profile = self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)
            self.pc = rs.pointcloud()
            
            # --- CẤU HÌNH CẢM BIẾN COLOR ĐỂ GIẢM LÓA & MỜ ---
            color_sensor = profile.get_device().first_color_sensor()
            if color_sensor:
                # Trả lại Auto-Exposure để không bị lóa khi phòng có đèn sáng chói
                color_sensor.set_option(rs.option.enable_auto_exposure, 1)
                
                # BÍ QUYẾT: Tắt tính năng "Auto-Exposure Priority" (Ưu tiên phơi sáng).
                # Khi tắt, camera BẮT BUỘC phải giữ framerate 30fps bằng mọi giá, 
                # nên nó KHÔNG THỂ tăng thời gian phơi sáng quá dài -> Cắt đứt hoàn toàn hiện tượng Motion Blur (nhòe ảnh)
                if color_sensor.supports(rs.option.auto_exposure_priority):
                    color_sensor.set_option(rs.option.auto_exposure_priority, 0)
                    
                # Vẫn giữ Sharpness ở mức cao để nét hình
                if color_sensor.supports(rs.option.sharpness):
                    color_sensor.set_option(rs.option.sharpness, 100)
            # ------------------------------------------------------------------
            
            # --- CẤU HÌNH CẢM BIẾN DEPTH ĐỂ TĂNG ĐỘ CHÍNH XÁC KHI ĐỨNG XA ---
            depth_sensor = profile.get_device().first_depth_sensor()
            if depth_sensor:
                # Tăng công suất phát tia hồng ngoại (Laser Power) lên tối đa (Thường là 360mW)
                # Tia hồng ngoại mạnh hơn sẽ dội lại rõ hơn khi khách hàng ngồi xa > 5m
                if depth_sensor.supports(rs.option.laser_power):
                    depth_sensor.set_option(rs.option.laser_power, 360)
                
                # Bật chế độ "High Accuracy" (Preset 3) trực tiếp trên chip xử lý của camera
                # Giúp camera tự động gọt bỏ các pixel bị nhiễu viền ở khoảng cách xa
                if depth_sensor.supports(rs.option.visual_preset):
                    try:
                        depth_sensor.set_option(rs.option.visual_preset, 3) # 3 là High Accuracy
                    except Exception as e:
                        print(f"[CẢNH BÁO] Không thể set High Accuracy Preset: {e}")
            # ------------------------------------------------------------------

            # QUAN TRỌNG: Lấy intrinsics của COLOR stream vì depth đã được align sang color
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            self.depth_intrinsics = color_profile.get_intrinsics()
            
            print("[INFO] Đã KẾT NỐI RealSense (Đã đồng bộ Intrinsics Color) và Cấu hình Sensor!")
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
            
            with self.frame_lock:
                self.latest_rgb_frame = frame.copy()

            results_pose = []
            results_seg = []
            
            # TỐI ƯU HÓA HIỆU NĂNG: Chỉ chạy YOLO Pose và Seg khi đang ở trạng thái TÌM KHÁCH
            # Tránh việc Camera và GPU chạy 100% công suất 24/7 gây lag/quá nhiệt Jetson
            if getattr(self, 'is_scanning_for_hand', False):
                if self.robot_state in ["IDLE", "STANDBY (SAVING GPU)"]: 
                    self.robot_state = "SCANNING"
                results_pose = self.model_pose.track(frame, conf=0.45, persist=True, tracker="bytetrack.yaml", verbose=False, half=(self.device==0), device=self.device)
                results_seg = self.model_seg.predict(frame, conf=0.45, verbose=False, half=(self.device==0), device=self.device)
            else:
                self.robot_state = "STANDBY (SAVING GPU)"
                self.locked_target_id = None
            
            curr_time = time.time()
            annotated_frame = frame.copy()

            for result_pose in results_pose:
                boxes = result_pose.boxes
                if boxes is None or boxes.id is None: continue
                
                keypoints = result_pose.keypoints.data if result_pose.keypoints is not None else None
                seg_result = results_seg[0] if len(results_seg) > 0 else None
                masks = seg_result.masks.xy if (seg_result and seg_result.masks is not None) else None

                # LỌC NGƯỜI GIƠ TAY BẰNG GIẢI PHẪU (ANATOMICAL LOGIC) QUA YOLO POSE
                raising_hands_ids = set()
                if keypoints is not None:
                    kpts_xy = result_pose.keypoints.xy.cpu().numpy()
                    kpts_conf = result_pose.keypoints.conf.cpu().numpy() if result_pose.keypoints.conf is not None else None
                    for j, box_j in enumerate(boxes.xyxy.cpu().numpy()):
                        if kpts_conf is None: continue
                        kp = kpts_xy[j]
                        cf = kpts_conf[j]
                        box_h = box_j[3] - box_j[1]
                        t_id = int(boxes.id[j].item())
                        
                        def get_valid_arm(kp_wrist, kp_elbow, side):
                            if cf[kp_wrist] > 0.25 and cf[kp_elbow] > 0.25:
                                wx, wy = kp[kp_wrist]
                                ex, ey = kp[kp_elbow]
                                forearm_len = math.hypot(wx - ex, wy - ey)
                                
                                is_pointing_up = (ey - wy) > (forearm_len * 0.85)
                                is_long_enough = forearm_len > max(30, box_h * 0.1)
                                
                                shoulder_idx = 5 if side == 'L' else 6
                                if cf[shoulder_idx] > 0.3:
                                    is_elbow_raised = ey < kp[shoulder_idx][1]
                                elif cf[0] > 0.3: # NOSE
                                    is_elbow_raised = ey < (kp[0][1] + forearm_len * 0.5)
                                else:
                                    is_elbow_raised = ey < (box_j[1] + box_h * 0.3)
                                    
                                if cf[0] > 0.3:
                                    is_high_enough = wy < (kp[0][1] - forearm_len * 0.4)
                                else:
                                    is_high_enough = wy < box_j[1]
                                    
                                if is_pointing_up and is_long_enough and is_elbow_raised and is_high_enough:
                                    return True
                            return False
                            
                        # KP_L_WRIST=9, KP_L_ELBOW=7, KP_R_WRIST=10, KP_R_ELBOW=8
                        if get_valid_arm(9, 7, 'L') or get_valid_arm(10, 8, 'R'):
                            raising_hands_ids.add(t_id)

                for i, box in enumerate(boxes):
                    track_id = int(box.id[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    binary_mask = None
                    if seg_result and seg_result.masks is not None:
                        # THUẬT TOÁN CENTER DISTANCE: Tìm đúng cái mask thuộc về người này (Tránh lấy nhầm người ngồi cạnh)
                        best_dist = float('inf')
                        best_mask_idx = -1
                        
                        target_cx = (x1 + x2) / 2
                        target_cy = (y1 + y2) / 2
                        
                        for m_idx, seg_box in enumerate(seg_result.boxes.xyxy.cpu().numpy()):
                            if int(seg_result.boxes.cls[m_idx].item()) != 0: continue # Chỉ lấy class 0 (Person)
                            
                            sx1, sy1, sx2, sy2 = seg_box
                            mcx = (sx1 + sx2) / 2
                            mcy = (sy1 + sy2) / 2
                            
                            dist = math.hypot(target_cx - mcx, target_cy - mcy)
                            if dist < best_dist and dist < 100: # Lệch tâm tối đa 100px
                                best_dist = dist
                                best_mask_idx = m_idx
                                
                        if best_mask_idx != -1:
                            mask_raw = seg_result.masks.data[best_mask_idx].cpu().numpy()
                            binary_mask = cv2.resize(mask_raw, (frame_w, frame_h))
                            binary_mask = (binary_mask > 0.5).astype(np.uint8)
                        
                        # --- TRỰC QUAN HÓA THUẬT TOÁN GỌT VIỀN LÊN GIAO DIỆN CHỨNG MINH CHO ĐỒ ÁN ---
                        try:
                            # 1. Vẽ đường viền gốc (Chưa gọt, còn nhiễu) -> Màu Đỏ nét mỏng
                            contours_raw, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(annotated_frame, contours_raw, -1, (0, 0, 255), 1)
                            
                            # 2. Gọt viền và vẽ đường viền ĐÃ LỌC (Sạch tuyệt đối) -> Màu Xanh Lá nét dày
                            kernel_vis = np.ones((7, 7), np.uint8)
                            eroded_mask_vis = cv2.erode(binary_mask, kernel_vis, iterations=1)
                            contours_eroded, _ = cv2.findContours(eroded_mask_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(annotated_frame, contours_eroded, -1, (0, 255, 0), 2)
                        except Exception as e:
                            pass # Bỏ qua nếu lỗi thư viện đồ họa để tránh sập luồng chính
                    
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
                
                    # MULTI-LAYER LIDAR FALLBACK (3 lớp bảo vệ chống xuyên tường)
                    body_distances = []
                    depth_layer_used = "none"
                    if keypoints is not None and i < len(keypoints):
                        kp = keypoints[i]
                        
                        # LỚP 1: Quét KHUÔN MẶT (Keypoints 0-4: Mũi, Mắt, Tai)
                        # Khuôn mặt phản xạ IR tốt nhất, nằm trọn trong cơ thể.
                        for k_idx in range(5): 
                            if len(kp) > k_idx and len(kp[k_idx]) >= 3 and kp[k_idx][2].item() > 0.4:
                                kx, ky = int(kp[k_idx][0].item()), int(kp[k_idx][1].item())
                                for dx in range(-7, 8, 3):
                                    for dy in range(-7, 8, 3):
                                        px, py = kx + dx, ky + dy
                                        if 0 <= px < frame_w and 0 <= py < frame_h:
                                            orig_px = frame_w - 1 - px
                                            d = depth_frame.get_distance(orig_px, py)
                                            if 0.2 < d < 6.0: body_distances.append(d)
                        
                        if body_distances:
                            depth_layer_used = "face"
                        else:
                            # LỚP 2: Quét VAI + HÔNG (Keypoints 5,6,11,12)
                            # Khi mặt bị khuất nhưng thân vẫn thấy rõ.
                            for k_idx in [5, 6, 11, 12]:
                                if len(kp) > k_idx and len(kp[k_idx]) >= 3 and kp[k_idx][2].item() > 0.4:
                                    kx, ky = int(kp[k_idx][0].item()), int(kp[k_idx][1].item())
                                    for dx in range(-10, 11, 4):
                                        for dy in range(-10, 11, 4):
                                            px, py = kx + dx, ky + dy
                                            if 0 <= px < frame_w and 0 <= py < frame_h:
                                                orig_px = frame_w - 1 - px
                                                d = depth_frame.get_distance(orig_px, py)
                                                if 0.2 < d < 6.0: body_distances.append(d)
                            if body_distances:
                                depth_layer_used = "body"
                                        
                    if body_distances:
                        body_distances.sort()
                        if depth_layer_used == "face":
                            d_m = float(body_distances[int(len(body_distances) * 0.3)])
                        else:
                            # Vai/Hông có thể lem ra ngoài -> dùng 15% an toàn hơn
                            d_m = float(body_distances[int(len(body_distances) * 0.15)])
                    else:
                        # LỚP 3: CẮT GỌT THEO SEGMENTATION VÀ 2D RADIUS FILTER
                        d_m = -1.0
                        if binary_mask is not None:
                            d_m = get_depth_distance_m_seg(depth_frame, binary_mask, frame_w, frame_h)
                        
                        # LỚP 4 (Fallback cuối cùng): Quét BBox cũ nếu Segmentation thất bại
                        if binary_mask is None or d_m <= 0:
                            d_m = get_depth_distance_m(depth_frame, (x1, y1, x2, y2), frame_w, frame_h, (person_center_x, person_center_y))
                    d_ngang_m = d_m
                
                    is_raising = False
                    if getattr(self, "is_scanning_for_hand", False) and track_id in raising_hands_ids:
                        is_raising = True

                    if track_id != -1 and self.locked_target_id is None:
                        # Tăng khoảng cách nhận diện từ 5.0m lên 8.0m vì YOLO nhận diện được rất xa
                        if is_raising and d_ngang_m <= 6.5:
                            self.open5_confirm_count[track_id] = self.open5_confirm_count.get(track_id, 0) + 1
                            if track_id not in self.hand_raise_start:
                                self.hand_raise_start[track_id] = curr_time
                                self.target_history[track_id] = []
                        else:
                            count = self.open5_confirm_count.get(track_id, 0)
                            if count > 0: self.open5_confirm_count[track_id] = count - 1
                            else: 
                                self.hand_raise_start.pop(track_id, None)
                                self.target_history.pop(track_id, None)
                    
                        if self.open5_confirm_count.get(track_id, 0) >= 2:
                            if track_id in self.hand_raise_start:
                                # Tích lũy (buffer) tọa độ tâm và khoảng cách nếu d_ngang_m hợp lệ
                                if d_ngang_m > 0 and d_ngang_m < 6.0:
                                    self.target_history[track_id].append((person_center_x, person_center_y, d_ngang_m))

                                hold_time = curr_time - self.hand_raise_start[track_id]
                                cv2.putText(annotated_frame, f"DANG KHOA TARGET: {hold_time:.1f}s/2s", (int(x1), int(y1)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                if hold_time >= 2.0:
                                    self.hand_raise_start.pop(track_id, None)
                                    self.open5_confirm_count[track_id] = 0
                                
                                    # 3. CHỐT TỌA ĐỘ BẰNG CÁCH LẤY TRUNG BÌNH (MEDIAN) CỦA POINTCLOUD TRONG VÙNG NGỰC
                                    hist = self.target_history.pop(track_id, [])
                                    
                                    # TÍNH HỆ QUY CHIẾU (ANCHOR) TỪ 60 KHUNG HÌNH (2 GIÂY VỪA QUA) ĐỂ KIỂM TRA CHÉO
                                    avg_cx, avg_cy, avg_dm = person_center_x, person_center_y, d_m
                                    if len(hist) > 0:
                                        avg_cx = float(np.median([h[0] for h in hist]))
                                        avg_cy = float(np.median([h[1] for h in hist]))
                                        avg_dm = float(np.median([h[2] for h in hist]))
                                        
                                    # Tạo PointCloud 3D
                                    self.pc.map_to(color_frame)
                                    points = self.pc.calculate(depth_frame)
                                    vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(frame_h, frame_w, 3)
                                    
                                    # Tích hợp Segmentation Mask vào PointCloud để độ chính xác tuyệt đối
                                    pc_success = False
                                    if binary_mask is not None:
                                        # binary_mask là ảnh đã flip. Cần unflip để khớp với vertices (ảnh gốc từ Lidar)
                                        unflipped_mask = cv2.flip(binary_mask, 1)
                                        
                                        # Lấy điểm PointCloud nằm trong Mask (Và có cự ly hợp lệ)
                                        valid_mask = (unflipped_mask > 0.7) & (vertices[:, :, 2] > 0.3) & (vertices[:, :, 2] < 6.0)
                                        valid_pts = vertices[valid_mask]
                                        
                                        if len(valid_pts) > 10:
                                            # Lọc nhiễu Foreground (những vật lồi lõm trước mặt)
                                            Z_values = valid_pts[:, 2]
                                            # Dùng p5_Z (5%) thay vì 15% vì ở BBox Fallback, người ở xa chiếm diện tích rất nhỏ.
                                            # Nếu dùng 15%, code sẽ bắt nhầm bức tường phía sau (5m) gây ra lỗi tọa độ văng ra ngoài Map!
                                            p5_Z = np.percentile(Z_values, 5)
                                            person_mask = (Z_values >= p5_Z - 0.1) & (Z_values <= p5_Z + 0.5)
                                            person_pts = valid_pts[person_mask]
                                            if len(person_pts) < 5: person_pts = valid_pts
                                            
                                            median_pt = np.median(person_pts, axis=0)
                                            x_opt, y_opt, z_opt = float(median_pt[0]), float(median_pt[1]), float(median_pt[2])
                                            
                                            # KIỂM TRA CHÉO (CROSS-VALIDATION) VỚI LỊCH SỬ 60 KHUNG HÌNH (ANCHOR)
                                            if avg_dm > 0 and abs(z_opt - avg_dm) > 0.4:
                                                print(f"[SMART NAV] ❌ CẢNH BÁO: PointCloud bị nhiễu (Z={z_opt:.2f}m khác xa Anchor={avg_dm:.2f}m). Từ chối PointCloud!")
                                                pc_success = False
                                            else:
                                                print(f"[SMART NAV] Ổn định tọa độ bằng Segmentation + PointCloud (Z={z_opt:.3f}m, X={-x_opt:.3f}m)")
                                                
                                                pitch_rad = math.radians(config.CAMERA_PITCH_DEG)
                                                forward_m_camera = z_opt * math.cos(pitch_rad) - y_opt * math.sin(pitch_rad)
                                                forward_m = forward_m_camera + 0.00 - 0.475 
                                                left_m = -x_opt
                                                pc_success = True
                                            
                                    if not pc_success:
                                        # Nếu không có Segmentation, Fallback về cắt BBox (Hình chữ nhật)
                                        roi_w = x2 - x1
                                        roi_h = y2 - y1
                                        roi_x1 = max(0, int(x1 + roi_w * 0.20))
                                        roi_x2 = min(frame_w, int(x2 - roi_w * 0.20))
                                        roi_y1 = max(0, int(y1 + roi_h * 0.05))
                                        roi_y2 = min(frame_h, int(y1 + roi_h * 0.45))
                                        
                                        if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                                            unflip_x1 = frame_w - roi_x2
                                            unflip_x2 = frame_w - roi_x1
                                            roi_pts = vertices[roi_y1:roi_y2, unflip_x1:unflip_x2]
                                            valid_mask = (roi_pts[:, :, 2] > 0.3) & (roi_pts[:, :, 2] < 6.0)
                                            valid_pts = roi_pts[valid_mask]
                                            
                                            if len(valid_pts) > 10:
                                                Z_values = valid_pts[:, 2]
                                                p5_Z = np.percentile(Z_values, 5)
                                                person_mask = (Z_values >= p5_Z - 0.1) & (Z_values <= p5_Z + 0.5)
                                                person_pts = valid_pts[person_mask]
                                                if len(person_pts) < 5: person_pts = valid_pts
                                                
                                                median_pt = np.median(person_pts, axis=0)
                                                x_opt, y_opt, z_opt = float(median_pt[0]), float(median_pt[1]), float(median_pt[2])
                                                
                                                # KIỂM TRA CHÉO (CROSS-VALIDATION) VỚI LỊCH SỬ 60 KHUNG HÌNH (ANCHOR)
                                                if avg_dm > 0 and abs(z_opt - avg_dm) > 0.4:
                                                    print(f"[SMART NAV] ❌ CẢNH BÁO: PointCloud bị nhiễu (Z={z_opt:.2f}m khác xa Anchor={avg_dm:.2f}m). Từ chối PointCloud!")
                                                    pc_success = False
                                                else:
                                                    print(f"[SMART NAV] Ổn định tọa độ bằng BBox + PointCloud (Z={z_opt:.3f}m, X={-x_opt:.3f}m)")
                                                    
                                                    pitch_rad = math.radians(config.CAMERA_PITCH_DEG)
                                                    forward_m_camera = z_opt * math.cos(pitch_rad) - y_opt * math.sin(pitch_rad)
                                                    forward_m = forward_m_camera + 0.00 - 0.475
                                                    left_m = -x_opt
                                                    pc_success = True
                                                
                                    if not pc_success:
                                        # Fallback cuối cùng nếu PointCloud hoàn toàn mù hoặc bị từ chối
                                        print(f"[SMART NAV] Ổn định tọa độ bằng Lidar Fallback 4 Lớp (Z={avg_dm:.3f}m)")
                                        rel = get_person_relative_position_m(depth_frame, (avg_cx, avg_cy), frame_w, frame_h, self.depth_intrinsics, avg_dm)
                                        if rel is not None:
                                            forward_m = rel[0] - 0.475; left_m = rel[1]
                                        else:
                                            forward_m = left_m = 0
                                        
                                    self.locked_target_id = track_id
                                    self.locked_bbox = (x1, y1, x2, y2)
                                    self.robot_state = "COLLECTING"
                                    self.status_update_signal.emit(f"ĐÃ KHÓA ! Đang tải dữ liệu không gian...")
                                
                                    print(f"\n{'='*60}")
                                    print(f"[DEBUG VISION] PC base_link = fwd:{forward_m:.3f}, left:{left_m:.3f}")
                                    print(f"{'='*60}")
                                    
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

                    # Đã loại bỏ logic has_fist (Nắm tay hủy lệnh) vì file này không dùng Mediapipe nữa
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
            if getattr(self, 'pause_emit', False) == False:
                self.change_pixmap_signal.emit(annotated_frame)
            
        self.pipeline.stop()

    def get_latest_frame(self):
        with self.frame_lock:
            if self.latest_rgb_frame is not None:
                return self.latest_rgb_frame.copy()
            return None

    def stop(self):
        self._run_flag = False
        self.wait()


# ================= Main App =================
