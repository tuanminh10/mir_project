#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os

# --- ROS FIX FOR VENV ---
_ros_sys_path = '/opt/ros/noetic/lib/python3/dist-packages'
if os.path.isdir(_ros_sys_path) and _ros_sys_path not in sys.path:
    sys.path.insert(1, _ros_sys_path)
    for mod_name in list(sys.modules.keys()):
        if any(mod_name.startswith(p) for p in ['geometry_msgs', 'nav_msgs', 'sensor_msgs',
                                                  'std_msgs', 'actionlib_msgs', 'tf2_msgs', 'move_base_msgs']):
            del sys.modules[mod_name]

import rospy
import numpy as np
import cv2
import math
import tf.transformations
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal, QTimer

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped

class MapLabelTest(QLabel):
    # Signal truyền tọa độ thực tế (X, Y) được click
    clicked_signal = pyqtSignal(float, float)

    def __init__(self):
        super().__init__()
        self.map_img = None
        self.map_info = None
        
        self.robot_px = None
        self.robot_yaw = 0.0
        
        self.original_target_px = None # Điểm click ban đầu (màu đỏ)
        self.retreated_goal_px = None  # Điểm đích lùi lại (màu xanh lá)
        self.goal_yaw = 0.0            # Hướng nhìn (Yaw)
        self.table_box_px = []         # Viền bàn nhận diện được

    def set_robot_pose(self, wx, wy, yaw=0.0):
        if not self.map_info:
            return
        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        h = self.map_info.height
        
        px = int((wx - ox) / res)
        py = h - int((wy - oy) / res) - 1
        
        if 0 <= px < self.map_info.width and 0 <= py < h:
            self.robot_px = (px, py)
            self.robot_yaw = yaw
            self.update_view()

    def set_map(self, occ_grid):
        self.map_info = occ_grid.info
        w = self.map_info.width
        h = self.map_info.height
        data = np.array(occ_grid.data, dtype=np.int8).reshape((h, w))
        self.map_data = data
        
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # Sửa lỗi hiển thị: Bắt chính xác màu theo giá trị chuẩn của ROS
        img[data == -1] = [220, 220, 220] # Xám nhạt cho vùng Unknown
        img[data == 0] = [255, 255, 255]  # Trắng cho vùng Free
        img[data > 0] = [0, 0, 0]         # Đen cho MỌI vật cản (kể cả xác suất thấp)
        
        self.map_img = cv2.flip(img, 0)
        self.update_view()

    def world_to_pixel(self, wx, wy):
        if not self.map_info: return None
        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        h = self.map_info.height
        px = int((wx - ox) / res)
        py = h - int((wy - oy) / res) - 1
        return (px, py)

    def update_view(self):
        if self.map_img is None:
            return
        display_img = self.map_img.copy()

        # 1. Vẽ Robot
        if self.robot_px and self.map_info:
            res = self.map_info.resolution
            rob_len_px = (0.89 / res) / 2
            rob_wid_px = (0.58 / res) / 2
            pts = []
            for dx, dy in [(-rob_len_px, -rob_wid_px), (rob_len_px, -rob_wid_px), 
                           (rob_len_px, rob_wid_px), (-rob_len_px, rob_wid_px)]:
                rx = dx * math.cos(-self.robot_yaw) - dy * math.sin(-self.robot_yaw)
                ry = dx * math.sin(-self.robot_yaw) + dy * math.cos(-self.robot_yaw)
                pts.append([int(self.robot_px[0] + rx), int(self.robot_px[1] + ry)])

            pts = np.array(pts, np.int32).reshape((-1, 1, 2))
            overlay = display_img.copy()
            cv2.fillPoly(overlay, [pts], (255, 165, 0)) # Cam mờ
            cv2.addWeighted(overlay, 0.6, display_img, 0.4, 0, display_img)
            cv2.polylines(display_img, [pts], True, (0, 0, 0), 2)
            
            end_x = int(self.robot_px[0] + rob_len_px * 1.5 * math.cos(-self.robot_yaw))
            end_y = int(self.robot_px[1] + rob_len_px * 1.5 * math.sin(-self.robot_yaw))
            cv2.arrowedLine(display_img, self.robot_px, (end_x, end_y), (0, 0, 255), 2, tipLength=0.3)

        # 2. Vẽ viền bàn nhận diện được (Xanh dương)
        if len(self.table_box_px) == 4:
            pts = np.array(self.table_box_px, np.int32).reshape((-1, 1, 2))
            cv2.polylines(display_img, [pts], True, (255, 0, 0), 2)

        # 3. Vẽ điểm gốc click (Target của AI nhắm vào người) - Đỏ (X)
        if self.original_target_px:
            tx, ty = self.original_target_px
            cv2.drawMarker(display_img, (tx, ty), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
            cv2.circle(display_img, (tx, ty), 6, (0, 0, 255), 1)

        # 4. Vẽ điểm lùi (Safe Goal thực tế) - Xanh lá
        if self.retreated_goal_px and self.original_target_px:
            gx, gy = self.retreated_goal_px
            
            # Đoạn lùi 1 mét vẽ vạch liền màu vàng
            cv2.line(display_img, self.original_target_px, self.retreated_goal_px, (0, 255, 255), 2, cv2.LINE_AA)
            
            # Điểm dừng cuối cùng (Đích thật của robot)
            cv2.circle(display_img, (gx, gy), 8, (0, 255, 0), -1)
            cv2.putText(display_img, "SAFE GOAL", (gx+10, gy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
            
            # Vẽ mũi tên thể hiện góc nhìn (Yaw) 45 độ
            ar_len = 35
            # Yaw trong GUI (Pixel) cần xoay ngược do trục Y hướng xuống
            gui_yaw = -self.goal_yaw
            end_x = int(gx + ar_len * math.cos(gui_yaw))
            end_y = int(gy + ar_len * math.sin(gui_yaw))
            cv2.arrowedLine(display_img, (gx, gy), (end_x, end_y), (0, 255, 0), 3, tipLength=0.3)

        self.display_img = display_img
        h, w, ch = self.display_img.shape
        bytesPerLine = ch * w
        qImg = QImage(self.display_img.data, w, h, bytesPerLine, QImage.Format_RGB888).copy()
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
        
        self.clicked_signal.emit(wx, wy)

class TestApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TEST THUẬT TOÁN LÙI KHOẢNG CÁCH (1.0 MÉT)")
        self.resize(800, 800)
        
        self.map_label = MapLabelTest()
        self.map_label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.map_label)
        
        self.map_label.clicked_signal.connect(self.simulate_safe_distance)

        # --- Khởi tạo ROS ---
        rospy.init_node('test_safe_distance_gui', anonymous=True, disable_signals=True)
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        rospy.Subscriber('/robot_pose', Pose, self.pose_callback)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.amcl_pose_callback)
        rospy.Subscriber('/mir_pose_simple', Pose, self.pose_callback)
        
        # Robot tọa độ lưu trữ
        self.robot_wx = 0.0
        self.robot_wy = 0.0
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.ros_spin)
        self.timer.start(100)

        print("\n[INFO] ĐÃ SẴN SÀNG.")
        print("💡 Hướng dẫn: Nhấp vào bất kỳ đâu trên bản đồ (mô phỏng vị trí con người/cái bàn).")
        print("Hệ thống sẽ vẽ:\n - Dấu X đỏ: Điểm bạn vừa nhấp\n - Chấm Xanh: Vị trí robot thực tế sẽ di chuyển đến (lùi lại 1 mét).")

    def simulate_safe_distance(self, target_x, target_y):
        if not self.map_label.map_info: return
        
        print(f"\n[TEST] Điểm Click: ({target_x:.2f}, {target_y:.2f})")
        
        res = self.map_label.map_info.resolution
        ox = self.map_label.map_info.origin.position.x
        oy = self.map_label.map_info.origin.position.y
        w = self.map_label.map_info.width
        h = self.map_label.map_info.height
        
        px_t = int((target_x - ox) / res)
        py_t = int((target_y - oy) / res)
        
        px_r = int((self.robot_wx - ox) / res)
        py_r = int((self.robot_wy - oy) / res)
        
        if not (0 <= px_t < w and 0 <= py_t < h):
            print("[TEST] Điểm Click nằm ngoài bản đồ.")
            return

        # 1. Tạo Mask Vật Cản (Bàn)
        # Sửa lỗi chí mạng: Lấy TẤT CẢ các điểm không phải là Không gian trống (0) và không phải là Chưa biết (-1).
        # Điều này bao gồm mọi giá trị dị thường (ví dụ số âm, số xác suất thấp) có thể có trong map của bạn.
        obs_mask = np.where((self.map_label.map_data != 0) & (self.map_label.map_data != -1), 255, 0).astype(np.uint8)
        
        # THUẬT TOÁN MỚI: Trích xuất cục bộ (Local Window Crop)
        # Thay vì quét cả bản đồ (dễ bị dính bàn vào tường thành 1 khối khổng lồ),
        # ta chỉ cắt 1 ô vuông 6m x 6m xung quanh điểm click để phân tích hình học.
        win_m = 6.0
        win_px = int(win_m / res)
        half_win = win_px // 2
        
        x1 = max(0, px_t - half_win)
        x2 = min(w, px_t + half_win)
        y1 = max(0, py_t - half_win)
        y2 = min(h, py_t + half_win)
        
        local_mask = obs_mask[y1:y2, x1:x2].copy()
        
        contours, _ = cv2.findContours(local_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Dịch chuyển toạ độ contours từ Local Window về lại Global Map
        global_contours = []
        for cnt in contours:
            global_cnt = cnt + np.array([[[x1, y1]]])
            global_contours.append(global_cnt)
            
        print(f"[DEBUG] Tìm thấy {len(global_contours)} khối vật cản trong bán kính {win_m/2}m.")
        
        # 2. Tìm Contour (Cái bàn) gần điểm Click nhất
        best_contour = None
        min_dist = float('inf')
        pt = (px_t, py_t)
        
        for cnt in global_contours:
            # Chỉ lọc bỏ nhiễu liti (cỡ 1-2 pixel). 
            # Dù cái bàn có là một nét đứt gãy không khép kín (Area = 0), thuật toán vẫn bắt được!
            if cv2.contourArea(cnt) < 2 and len(cnt) < 5:
                continue
            
            dist = cv2.pointPolygonTest(cnt, pt, True)
            
            if dist >= 0:
                # Điểm click nằm hoàn toàn bên trong cái bàn này -> Lấy luôn!
                best_contour = cnt
                break
            else:
                abs_dist = abs(dist)
                if abs_dist < min_dist:
                    min_dist = abs_dist
                    best_contour = cnt
                
        if best_contour is None:
            print("[TEST] Không tìm thấy cái bàn nào gần đây! (Xung quanh 3m toàn khoảng trống)")
            return
            
        # 3. Phân tích Hình học của Cái Bàn (minAreaRect)
        rect = cv2.minAreaRect(best_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        edges = []
        for i in range(4):
            p1 = box[i]
            p2 = box[(i+1)%4]
            length = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
            center = ((p1[0]+p2[0])/2.0, (p1[1]+p2[1])/2.0)
            edges.append({'p1': p1, 'p2': p2, 'len': length, 'center': center})
            
        edges.sort(key=lambda e: e['len'], reverse=True)
        long_edges = edges[0:2]
        short_edges = edges[2:4]
        
        # 4. Tìm cạnh dài gần người ngồi nhất
        best_long_edge = None
        min_ed = float('inf')
        for edge in long_edges:
            d = math.hypot(edge['center'][0] - px_t, edge['center'][1] - py_t)
            if d < min_ed:
                min_ed = d
                best_long_edge = edge
                
        # Chiếu điểm Click lên Cạnh dài
        p1 = np.array(best_long_edge['p1'], dtype=float)
        p2 = np.array(best_long_edge['p2'], dtype=float)
        
        vec_edge = p2 - p1
        length_edge = best_long_edge['len']
        vec_edge_unit = vec_edge / length_edge if length_edge > 0 else np.array([1, 0])
        
        vec_pt = np.array([px_t - p1[0], py_t - p1[1]], dtype=float)
        proj_length = np.dot(vec_pt, vec_edge_unit)
        
        # Tỷ lệ vị trí người ngồi trên bàn t (0 -> 1)
        t = proj_length / length_edge if length_edge > 0 else 0.5
        t = max(0.0, min(1.0, t))
        
        proj_pt = p1 + vec_edge_unit * proj_length
        
        # Xác định pháp tuyến của cạnh dài hướng ra ngoài bàn
        rect_center = np.array(rect[0])
        vec_center_to_edge = np.array(best_long_edge['center']) - rect_center
        normal_long = np.array([-vec_edge_unit[1], vec_edge_unit[0]]) # Vuông góc
        if np.dot(vec_center_to_edge, normal_long) < 0:
            normal_long = -normal_long # Đảm bảo hướng ra ngoài
            
        goal_px_x, goal_px_y = None, None
        goal_yaw = 0.0
        
        case_type = ""
        if t < 0.2 or t > 0.8:
            # === KỊCH BẢN A: Người Đầu Bàn ===
            case_type = "A (Đầu Bàn/Góc Bàn)"
            best_short_edge = None
            min_sd = float('inf')
            for edge in short_edges:
                d = math.hypot(edge['center'][0] - proj_pt[0], edge['center'][1] - proj_pt[1])
                if d < min_sd:
                    min_sd = d
                    best_short_edge = edge
                    
            safe_dist_px = int(0.7 / res) # Lùi 0.7m
            
            sp1 = np.array(best_short_edge['p1'], dtype=float)
            sp2 = np.array(best_short_edge['p2'], dtype=float)
            vec_se = sp2 - sp1
            vec_se_unit = vec_se / best_short_edge['len'] if best_short_edge['len'] > 0 else np.array([1,0])
            normal_short = np.array([-vec_se_unit[1], vec_se_unit[0]])
            if np.dot(np.array(best_short_edge['center']) - rect_center, normal_short) < 0:
                normal_short = -normal_short
                
            goal_px_x = best_short_edge['center'][0] + normal_short[0] * safe_dist_px
            goal_px_y = best_short_edge['center'][1] + normal_short[1] * safe_dist_px
            
            # Góc nhìn 45 độ từ giữa cạnh ngắn đâm chéo vào góc người ngồi
            dir_yaw = np.array([proj_pt[0] - goal_px_x, proj_pt[1] - goal_px_y])
            goal_yaw = math.atan2(dir_yaw[1], dir_yaw[0])
            
        else:
            # === KỊCH BẢN B: Người Giữa Bàn ===
            case_type = "B (Giữa Dãy Bàn)"
            # Lùi lệch chéo 45 độ.
            offset_px = int(0.7 / res) # 0.7m ra xa, 0.7m lệch sang trái/phải
            
            # Goal hướng về P2 (Cuối cạnh dài)
            goal_p2 = proj_pt + normal_long * offset_px + vec_edge_unit * offset_px
            # Goal hướng về P1 (Đầu cạnh dài)
            goal_p1 = proj_pt + normal_long * offset_px - vec_edge_unit * offset_px
            
            # --- TÙY CHỌN 1 (ĐANG DÙNG): Tối ưu HRI (Tương tác con người) ---
            # Ưu tiên chọn hướng đỗ nghiêng về phía góc bàn GẦN NGƯỜI NGỒI NHẤT
            # Điều này giúp robot không chen vào giữa bàn (tránh cản đường người khác)
            if t < 0.5:
                preferred_goal = goal_p1
                fallback_goal = goal_p2
            else:
                preferred_goal = goal_p2
                fallback_goal = goal_p1
                
            # --- TÙY CHỌN 2 (ĐANG TẮT): Tối ưu Di chuyển cho Robot ---
            # Chọn hướng đỗ sao cho điểm đến GẦN VỚI VỊ TRÍ HIỆN TẠI CỦA ROBOT NHẤT.
            # Ưu điểm: Robot di chuyển nhanh nhất. Nhược điểm: Có thể chen ngang vào giữa dãy bàn.
            # Để dùng cách này, hãy XÓA DẤU # ở đoạn code dưới đây và THÊM DẤU # vào đoạn TÙY CHỌN 1:
            # d1 = math.hypot(goal_p1[0] - px_r, goal_p1[1] - py_r)
            # d2 = math.hypot(goal_p2[0] - px_r, goal_p2[1] - py_r)
            # if d1 < d2:
            #     preferred_goal = goal_p1
            #     fallback_goal = goal_p2
            # else:
            #     preferred_goal = goal_p2
            #     fallback_goal = goal_p1
            
            # Kiểm tra an toàn
            free_space = (self.map_label.map_data == 0).astype(np.uint8)
            safe_radius_px = int(0.4 / res)
            kernel_safe = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (safe_radius_px*2+1, safe_radius_px*2+1))
            safe_mask = cv2.erode(free_space, kernel_safe)
            
            def is_safe(g):
                gx, gy = int(g[0]), int(g[1])
                if 0 <= gx < w and 0 <= gy < h:
                    return safe_mask[gy, gx] == 1
                return False
                
            if is_safe(preferred_goal):
                chosen = preferred_goal
            elif is_safe(fallback_goal):
                chosen = fallback_goal
            else:
                chosen = proj_pt + normal_long * offset_px # Fallback thẳng góc
                
            goal_px_x = chosen[0]
            goal_px_y = chosen[1]
            
            dir_yaw = np.array([proj_pt[0] - goal_px_x, proj_pt[1] - goal_px_y])
            goal_yaw = math.atan2(dir_yaw[1], dir_yaw[0])

        safe_x = ox + goal_px_x * res
        safe_y = oy + goal_px_y * res
        
        print(f"[TEST] 🎯 Áp dụng Kịch bản {case_type}")
        print(f"[TEST] Điểm đích an toàn: ({safe_x:.2f}, {safe_y:.2f})")
        
        px_target_gui = self.map_label.world_to_pixel(target_x, target_y)
        px_safe_gui = self.map_label.world_to_pixel(safe_x, safe_y)
        
        if px_target_gui and px_safe_gui:
            self.map_label.original_target_px = px_target_gui
            self.map_label.retreated_goal_px = px_safe_gui
            self.map_label.goal_yaw = goal_yaw # Update GUI arrow yaw
            # Cần vẽ thêm yaw
            
            # Gửi tín hiệu để vẽ cái bàn
            self.map_label.table_box_px = []
            for b in box:
                b_gui = self.map_label.world_to_pixel(ox + b[0]*res, oy + b[1]*res)
                if b_gui: self.map_label.table_box_px.append(b_gui)
                
            self.map_label.update_view()

    # --- ROS Callbacks ---
    def map_callback(self, msg):
        self.map_label.set_map(msg)
        self.resize(msg.info.width, msg.info.height)

    def pose_callback(self, msg):
        yaw = tf.transformations.euler_from_quaternion([
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])[2]
        self.robot_wx = msg.position.x
        self.robot_wy = msg.position.y
        self.map_label.set_robot_pose(self.robot_wx, self.robot_wy, yaw)

    def amcl_pose_callback(self, msg):
        yaw = tf.transformations.euler_from_quaternion([
            msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, 
            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2]
        self.robot_wx = msg.pose.pose.position.x
        self.robot_wy = msg.pose.pose.position.y
        self.map_label.set_robot_pose(self.robot_wx, self.robot_wy, yaw)

    def ros_spin(self):
        if rospy.is_shutdown(): self.close()

def main():
    app = QApplication(sys.argv)
    window = TestApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
