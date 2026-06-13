import sys

with open("test.py", "r") as f:
    content = f.read()

# 1. Replace geometry functions
geom_start_idx = content.find("def extract_3d_coordinates_from_pc(")
geom_end_idx = content.find("# ================= GUI Map =================")

if geom_start_idx != -1 and geom_end_idx != -1:
    geom_new = """def get_depth_distance_m_seg(depth_frame, poly_pts, frame_w, frame_h):
    import numpy as np
    import cv2
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(poly_pts, dtype=np.int32)], 1)
    ys, xs = np.where(mask == 1)
    distances = []
    for idx in range(0, len(xs), 10):
        px, py = xs[idx], ys[idx]
        orig_px = frame_w - 1 - px
        if 0 <= orig_px < frame_w and 0 <= py < frame_h:
            d = depth_frame.get_distance(orig_px, py)
            if 0.3 < d < 6.0: distances.append(d)
    return float(np.median(distances)) if distances else -1.0

def get_person_relative_position_m(depth_frame, center_pt, frame_w, frame_h, depth_intrinsics, distance_m):
    import math
    import pyrealsense2 as rs
    if len(center_pt) == 4:
        x1, y1, x2, y2 = map(int, center_pt)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
    else:
        center_x, center_y = map(int, center_pt)
        
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

"""
    content = content[:geom_start_idx] + geom_new + content[geom_end_idx:]

# 2. Replace VideoThread
thread_start_idx = content.find("class VideoThread(QThread):")
thread_end_idx = content.find("# ================= Main App =================")

if thread_start_idx != -1 and thread_end_idx != -1:
    thread_new = """class VideoThread(QThread):
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
            annotated_frame = results_pose[0].plot() if len(results_pose) > 0 else frame.copy()

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
                    
                    poly_pts = masks[i] if (masks and i < len(masks)) else None
                    if poly_pts is not None and len(poly_pts) > 10:
                        d_m = get_depth_distance_m_seg(depth_frame, poly_pts, frame_w, frame_h)
                    else:
                        d_m = get_depth_distance_m(depth_frame, (x1, y1, x2, y2), frame_w, frame_h)
                        
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

                    if track_id != -1 and (self.locked_target_id is None or self.locked_target_id == track_id):
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
                                    self.locked_target_id = track_id
                                    self.locked_bbox = (x1, y1, x2, y2)
                                    self.robot_state = "COLLECTING"
                                    self.status_update_signal.emit(f"ĐÃ KHÓA ! Đang tải dữ liệu không gian...")
                                    
                                    # Chuyển đổi tọa độ ngay lập tức và emit signal
                                    rel = get_person_relative_position_m(depth_frame, (person_center_x, person_center_y), frame_w, frame_h, self.depth_intrinsics, d_m)
                                    if rel is not None:
                                        camera_offset_x = 0.475
                                        forward_m, left_m = rel[0] - camera_offset_x, rel[1]
                                        
                                        msg = PointStamped()
                                        msg.header.stamp = rospy.Time(0)
                                        msg.header.frame_id = "base_link"
                                        msg.point.x = forward_m
                                        msg.point.y = left_m
                                        msg.point.z = 0.0
                                        
                                        try:
                                            self.tf_listener.waitForTransform("/map", "base_link", rospy.Time(0), rospy.Duration(1.0))
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

"""
    content = content[:thread_start_idx] + thread_new + content[thread_end_idx:]

with open("test.py", "w") as f:
    f.write(content)
