#!/usr/bin/env python3
import sys
import rospy
import numpy as np
import cv2
import actionlib
import requests
import time
import math
import tf.transformations
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal, QTimer

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Pose, PoseWithCovarianceStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

MIR_IP = "192.168.0.177"
MIR_API_URL = f"http://{MIR_IP}/api/v2.0.0"
MIR_AUTH = "Basic YWRtaW46OGM2OTc2ZTViNTQxMDQxNWJkZTkwOGJkNGRlZTE1ZGZiMTY3YTljODczZmM0YmI4YTgxZjZmMmFiNDQ4YTkxOA=="

class MapLabel(QLabel):
    # Signal truyền tọa độ (x, y, yaw) thế giới thực ra ngoài khi click và kéo
    clicked_signal = pyqtSignal(float, float, float)

    def __init__(self):
        super().__init__()
        self.map_img = None
        self.map_info = None
        
        self.goal_px = None
        self.goal_yaw = 0.0
        self.path_px = []
        self.robot_px = None
        self.robot_yaw = 0.0
        
        self.drag_start_px = None
        self.drag_current_px = None
        self.is_dragging = False

    def set_robot_pose(self, wx, wy, yaw=0.0):
        """Cập nhật vị trí và góc quay robot"""
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
        """Chuyển đổi dữ liệu OccupancyGrid 1D của ROS thành hình ảnh OpenCV 2D"""
        self.map_info = occ_grid.info
        w = self.map_info.width
        h = self.map_info.height
        
        # Reshape data
        data = np.array(occ_grid.data, dtype=np.int8).reshape((h, w))
        
        # Tô màu map: -1 là chưa rõ (xám), 0 là trống (trắng), 100 là vật cản (đen)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[data == -1] = [127, 127, 127]
        img[data == 0] = [255, 255, 255]
        img[data == 100] = [0, 0, 0]
        
        # Cần flip (lật) lại ảnh do origin của hệ tọa độ ROS /map ở góc dưới bên trái.
        # Trong khi OpenCV/PyQt thì tọa độ (0,0) nằm ở góc trên bên trái.
        self.map_img = cv2.flip(img, 0)
        self.update_view()

    def set_path(self, path_msg):
        """Lưu đường đi do Navfn hoặc GlobalPlanner trả về và chuyển sang pixel"""
        if not self.map_info:
            return
        
        self.path_px = []
        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        h = self.map_info.height
        
        # Đổi từng pose trong path sang dạng pixel
        for pose in path_msg.poses:
            wx = pose.pose.position.x
            wy = pose.pose.position.y
            
            px = int((wx - ox) / res)
            py = h - int((wy - oy) / res) - 1
            if 0 <= px < self.map_info.width and 0 <= py < h:
                self.path_px.append((px, py))
                
        self.update_view()

    def update_view(self):
        """Vẽ đường, vẽ điểm click và hiển thị lên QLabel"""
        if self.map_img is None:
            return
            
        display_img = self.map_img.copy()
        
        # Đã đổi màu vẽ đường đi sang Đỏ (Trong OpenCV thì BGR là (0, 0, 255))
        # Khi đẩy ra QImage Format_RGB888, ảnh BGR sẽ đổi kênh nhưng ta cần 
        # (255, 0, 0) trong RGB. Thực chất OpenCV lưu (B,G,R), còn QImage yêu cầu (R,G,B).
        # Vì đoạn lệnh gọi QPixmap không xài cvtColor, chúng ta sẽ điền tuple ngược.
        # Muốn in Đỏ trong RGB -> Cần phần tử thứ nhất của Tuple = 255 (R). Nên tuple viết là (255, 0, 0)
        if len(self.path_px) > 1:
            for i in range(len(self.path_px)-1):
                cv2.line(display_img, self.path_px[i], self.path_px[i+1], (255, 0, 0), 2)
                
        # Vẽ đích đến click - chấm Xanh lá -> tuple (0, 255, 0) và hình mũi tên chỉ hướng
        if self.goal_px:
            cv2.circle(display_img, self.goal_px, 6, (0, 255, 0), -1)
            # Vẽ hướng (mũi tên) cho Goal đã chốt (màu xanh lá)
            gx, gy = self.goal_px
            ar_len = 30
            end_x = int(gx + ar_len * math.cos(-self.goal_yaw))
            end_y = int(gy + ar_len * math.sin(-self.goal_yaw))
            cv2.arrowedLine(display_img, (gx, gy), (end_x, end_y), (0, 255, 0), 2, tipLength=0.3)

        # Vẽ đường đang kéo (mũi tên xem trước hướng) - màu tím -> tuple (255, 0, 255)
        if self.is_dragging and self.drag_start_px and self.drag_current_px:
            # Chỉ vẽ nếu kéo đủ rộng
            dist = math.hypot(self.drag_current_px[0] - self.drag_start_px[0], self.drag_current_px[1] - self.drag_start_px[1])
            if dist > 5:
                cv2.arrowedLine(display_img, self.drag_start_px, self.drag_current_px, (255, 0, 255), 2, tipLength=0.3)

        # Vẽ vị trí thực của robot dạng Footprint (Hình chữ nhật tượng trưng cho MiR)
        if self.robot_px and self.map_info:
            res = self.map_info.resolution
            # Kích thước ước tính của MiR từ config: Dài 0.89m, Rộng 0.58m
            rob_len_px = (0.89 / res) / 2
            rob_wid_px = (0.58 / res) / 2

            # Tính toán 4 góc của footprint
            pts = []
            for dx, dy in [(-rob_len_px, -rob_wid_px), (rob_len_px, -rob_wid_px), 
                           (rob_len_px, rob_wid_px), (-rob_len_px, rob_wid_px)]:
                # Xoay theo góc yaw
                # Chú ý: Trục Y pixel lộn ngược so với Y thực tế, nên ta dùng -self.robot_yaw để bù
                rx = dx * math.cos(-self.robot_yaw) - dy * math.sin(-self.robot_yaw)
                ry = dx * math.sin(-self.robot_yaw) + dy * math.cos(-self.robot_yaw)
                pts.append([int(self.robot_px[0] + rx), int(self.robot_px[1] + ry)])

            pts = np.array(pts, np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Vẽ hình chữ nhật mô phỏng kích thước URDF (Màu Cam mờ, viền Đen)
            overlay = display_img.copy()
            cv2.fillPoly(overlay, [pts], (0, 165, 255))
            cv2.addWeighted(overlay, 0.6, display_img, 0.4, 0, display_img)
            cv2.polylines(display_img, [pts], True, (0, 0, 0), 2)
            
            # Vẽ mũi tên chỉ hướng của Robot
            end_x = int(self.robot_px[0] + rob_len_px * 1.5 * math.cos(-self.robot_yaw))
            end_y = int(self.robot_px[1] + rob_len_px * 1.5 * math.sin(-self.robot_yaw))
            cv2.arrowedLine(display_img, self.robot_px, (end_x, end_y), (0, 0, 255), 2, tipLength=0.3)

        # Đẩy màn chập ra QPixmap. Lưu self.display_img để giữ bộ nhớ không bị giải phóng
        self.display_img = display_img
        h, w, ch = self.display_img.shape
        bytesPerLine = ch * w
        qImg = QImage(self.display_img.data, w, h, bytesPerLine, QImage.Format_RGB888).copy()
        self.setPixmap(QPixmap.fromImage(qImg))

    def mousePressEvent(self, event):
        """Bắt sự kiện BẮT ĐẦU click chuột để lấy vị trí gốc"""
        if self.map_info is None:
            return
            
        self.drag_start_px = (event.x(), event.y())
        self.drag_current_px = self.drag_start_px
        self.is_dragging = True
        self.update_view()

    def mouseMoveEvent(self, event):
        """Sự kiện KÉO chuột để cập nhật hướng chỉ"""
        if not self.is_dragging:
            return
        self.drag_current_px = (event.x(), event.y())
        self.update_view()

    def mouseReleaseEvent(self, event):
        """Sự kiện THẢ chuột để chốt Goal và tính toán hướng (Yaw)"""
        if not self.is_dragging or self.map_info is None:
            return
        
        self.is_dragging = False
        self.drag_current_px = (event.x(), event.y())
        
        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        h = self.map_info.height
        
        # 1. Tọa độ Goal (Chính là Tọa độ Start lúc click nhấn)
        spx, spy = self.drag_start_px
        wx = ox + spx * res
        wy = oy + (h - spy - 1) * res
        
        # 2. Tọa độ hướng chỉ (Để tính Yaw)
        epx, epy = self.drag_current_px
        ewx = ox + epx * res
        ewy = oy + (h - epy - 1) * res
        
        # Tính góc lệnh Yaw (Radians)
        yaw = math.atan2(ewy - wy, ewx - wx)
        
        # Nếu nháp trúng hoặc kéo quá ngắn (Distance < 0.1m) => cho Yaw về hướng mặc định = 0 để chống lỗi
        if math.hypot(ewx - wx, ewy - wy) < 0.1:
            yaw = 0.0
            
        self.goal_px = self.drag_start_px
        self.goal_yaw = yaw
        
        # Phát signal (wx, wy, yaw)
        self.clicked_signal.emit(wx, wy, yaw)
        self.update_view()


class MapNavApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive MiR Navigation Map")
        
        # Giao diện chính
        self.map_label = MapLabel()
        self.map_label.clicked_signal.connect(self.send_goal)
        self.setCentralWidget(self.map_label)
        
        # --- Khởi tạo ROS ---
        rospy.init_node('interactive_map_gui', anonymous=True, disable_signals=True)
        
        # 1. Topic Bản đồ
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        
        # 2. Topic Kế hoạch đường đi 
        # Đã thêm các cấu hình topic phổ biến của MiR để đường (Path) màu đỏ hiện lên
        rospy.Subscriber("/move_base_node/GlobalPlanner/plan", Path, self.path_callback)
        rospy.Subscriber("/move_base/GlobalPlanner/plan", Path, self.path_callback)
        rospy.Subscriber('/move_base_node/SBPLLatticePlanner/plan', Path, self.path_callback)
        rospy.Subscriber('/move_base_node/mir_global_planner/plan', Path, self.path_callback)
        rospy.Subscriber('/move_base/NavfnROS/plan', Path, self.path_callback)
        rospy.Subscriber('/mir_planner/global_path', Path, self.path_callback)
        
        # 3. Topic Robot Pose (Vị trí hiện tại)
        # MiR thường publish ở một trong các topic dưới đây
        rospy.Subscriber('/robot_pose', Pose, self.pose_callback)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.amcl_pose_callback)
        rospy.Subscriber('/mir_pose_simple', Pose, self.pose_callback)
        
        # Action Client: Gửi Goal tới move_base
        self.move_base = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        
        # Thêm cách chuẩn của RVIZ (2D Nav Goal) để gửi qua topic trực tiếp cho chắc chắn
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        
        print("Sẵn sàng! Hãy click lên bản đồ để điều khiển robot.")
        
        # --- Timer vòng lặp (Spin) cho ROS ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.ros_spin)
        self.timer.start(100) # Cập nhật mỗi 100ms

    def map_callback(self, msg):
        self.map_label.set_map(msg)
        # Tự động thay đổi kích thước cửa sổ vừa với bản đồ
        self.resize(msg.info.width, msg.info.height)

    def path_callback(self, msg):
        self.map_label.set_path(msg)

    def pose_callback(self, msg):
        yaw = tf.transformations.euler_from_quaternion([
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        ])[2]
        self.map_label.set_robot_pose(msg.position.x, msg.position.y, yaw)

    def amcl_pose_callback(self, msg):
        yaw = tf.transformations.euler_from_quaternion([
            msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, 
            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
        ])[2]
        self.map_label.set_robot_pose(msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)

    def send_goal(self, wx, wy, wyaw):
        print(f"[!] Gửi mục tiêu đến tọa độ bản đồ: X={wx:.2f}, Y={wy:.2f}, Góc (Yaw)={wyaw:.2f} rad")

        # -------------------------------------------------------------
        # Gọi REST API để xóa lỗi và ép MiR vào trạng thái Ready (State 3)
        # Nếu Robot bị kẹt ở trạng thái pause/error (vàng/đỏ) thì ROS sẽ bị khước từ.
        # -------------------------------------------------------------
        headers = {"Content-Type": "application/json", "Authorization": MIR_AUTH}
        try:
            requests.delete(f"{MIR_API_URL}/status", headers=headers, timeout=1)
            time.sleep(0.1)
            requests.put(f"{MIR_API_URL}/status", headers=headers, json={"state_id": 3}, timeout=1)
            print("Đã cấp quyền Ready (Màu xanh) cho Web Dashboard của MiR!")
        except Exception as e:
            print(f"Cảnh báo: Không thể ép API Dashboard sang trạng thái Ready, Lỗi: {e}")
            return
            
        # Lấy trạng thái hiện tại để biết map_id
        try:
            st = requests.get(f"{MIR_API_URL}/status", headers=headers, timeout=2).json()
            map_id = st.get("map_id", "")
            if not map_id:
                print("Lỗi: Không tìm thấy map_id của robot!")
                return
        except Exception as e:
            print(f"Lỗi khi lấy map_id: {e}")
            return

        # Tìm mã ID của Mission "Move"
        move_guid = None
        param_name = "Position"
        try:
            ms = requests.get(f"{MIR_API_URL}/missions", headers=headers, timeout=2).json()
            for m in ms:
                name = m.get("name", "").lower()
                if name in ("move", "go to", "di chuyen", "goto"):
                    move_guid = m.get("guid")
                    break
        except Exception:
            pass

        if not move_guid:
            print("Lỗi: Không tìm thấy mission 'Move' trên MiR Dashboard!")
            return

        # 1. Tạo vị trí tạm thời
        pos_name = f"click_nav_{int(time.time())}"
        try:
            r = requests.post(f"{MIR_API_URL}/positions", headers=headers, json={
                "name": pos_name,
                "pos_x": wx,
                "pos_y": wy,
                "orientation": math.degrees(wyaw), # MiR REST API yêu cầu nhận Yaw bằng độ (Degrees)
                "type_id": 0,
                "map_id": map_id
            }, timeout=2)
            if r.status_code not in (200, 201):
                print(f"Lỗi tạo Position: {r.text}")
                return
            pos_guid = r.json().get("guid", "")
            print(f"Đã tạo điểm đích tạm thời (GUID: {pos_guid[:8]}...)")
        except Exception as e:
            print(f"Lỗi gọi API tạo position: {e}")
            return

        # 2. Xóa Queue cũ
        try:
            requests.delete(f"{MIR_API_URL}/mission_queue", headers=headers, timeout=2)
            time.sleep(0.1)
        except:
            pass

        # 3. Add Mission Move vào Queue
        try:
            r = requests.post(f"{MIR_API_URL}/mission_queue", headers=headers, json={
                "mission_id": move_guid,
                "parameters": [{"input_name": param_name, "value": pos_guid}]
            }, timeout=2)
            if r.status_code in (200, 201):
                print("🎯 Lệnh di chuyển đã được đẩy vào MiR Queue thành công!")
            else:
                print(f"Lỗi đưa lệnh vào Queue: {r.text}")
        except Exception as e:
            print(f"Lỗi add mission: {e}")

        # Đồng thời vẫn báo cho ROS Topic để Rviz hoặc node khác có thể bắt kịp
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = "map"
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.pose.position.x = wx
        goal_msg.pose.position.y = wy
        
        q = tf.transformations.quaternion_from_euler(0, 0, wyaw)
        goal_msg.pose.orientation.x = q[0]
        goal_msg.pose.orientation.y = q[1]
        goal_msg.pose.orientation.z = q[2]
        goal_msg.pose.orientation.w = q[3]
        
        self.goal_pub.publish(goal_msg)

    def ros_spin(self):
        """Giả lập hàm rospy.spin() không block giao diện PyQt"""
        if rospy.is_shutdown():
            self.close()

def main():
    app = QApplication(sys.argv)
    gui = MapNavApp()
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
