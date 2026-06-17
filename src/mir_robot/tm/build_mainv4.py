import re

with open('/home/tuanminh/mir_project/src/mir_robot/tm/fix54.py', 'r') as f:
    fix54 = f.read()

with open('/home/tuanminh/mir_project/src/mir_robot/tm/mainv3.py', 'r') as f:
    mainv3 = f.read()

imports_and_hacks = fix54.split('# ================= Utils =================')[0]

imports_and_hacks += "import dongco\nimport mir_tts\nimport requests\nimport queue\nimport json\nimport threading\nfrom std_msgs.msg import String\n\n# ================= Utils =================\n"

utils = fix54.split('# ================= Utils =================')[1].split('# ================= GUI Map =================')[0]
maplabel = fix54.split('# ================= GUI Map =================')[1].split('# ================= Camera Thread =================')[0]
videothread = fix54.split('# ================= Camera Thread =================')[1].split('# ================= Main App =================')[0]

hybrid_algo = fix54.split('def calculate_hybrid_safe_goal(self, target_x, target_y, obs_pt_map=None):')[1].split('def closeEvent(self, event):')[0]
hybrid_algo = '    def calculate_hybrid_safe_goal(self, target_x, target_y, obs_pt_map=None):\n' + hybrid_algo

main_app = """
# ================= Main App =================
class MainApp(QMainWindow):
    map_signal = pyqtSignal(object)
    pose_signal = pyqtSignal(float, float, float)

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
            self.laptop_yolo = YOLO('/home/tuanminh/mir_project/src/mir_robot/tm/best/best.pt')
        except:
            rospy.logwarn("Không tìm thấy model laptop best.pt")

        self.task_queue = queue.Queue()
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
            self.servo.set_angle(90)
        except Exception as e:
            print("Lỗi Servo:", e)
            self.servo = None

        self.robot = nav.ws_connect()
        self.mir_headers = nav.api_login()
        if self.mir_headers:
            nav.api_ensure_ready(self.mir_headers)

        self.map_signal.connect(self.map_label.set_map)
        self.pose_signal.connect(self.map_label.set_robot_pose)

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

    def on_shutdown_hook(self):
        rospy.logwarn("ĐANG DỪNG KHẨN CẤP...")
        self.video_thread.stop()
        if self.servo: self.servo.cleanup()
        if self.mir_headers: nav.api_set_state(self.mir_headers, 4)

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
            self.task_queue.put({"type": "GUEST_CALL", "target": ban})
            self.charging_cancel_event.set()
        except: pass

    def on_web_order(self, msg):
        try:
            data = json.loads(msg.data)
            ban = str(data.get("ban", "")).strip()
            if ban.isdigit(): ban = f"ban {ban}"
            self.active_orders[ban] = {"coca": int(data.get("coca", 0)), "lavie": int(data.get("lavie", 0))}
            self.wait_event.set()
        except: pass

    def on_kitchen_cmd(self, msg):
        try:
            data = json.loads(msg.data)
            action = data.get("action")
            if action == "call_robot":
                self.task_queue.put({"type": "KITCHEN_CALL", "target": "bep"})
                self.charging_cancel_event.set()
            elif action == "deliver":
                ban = str(data.get("table", "")).strip()
                if ban.isdigit(): ban = f"ban {ban}"
                if self.current_location != "bep":
                    self.task_queue.put({"type": "KITCHEN_CALL", "target": "bep"})
                self.task_queue.put({"type": "DELIVER", "target": ban})
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

    def verify_tray(self, exp_coca, exp_lavie, check_empty=False):
        if not self.laptop_yolo: return True
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): return True
        start = rospy.Time.now()
        success = 0
        while (rospy.Time.now() - start).to_sec() < 30.0:
            ret, frame = cap.read()
            if not ret: break
            res = self.laptop_yolo.track(frame, persist=True, stream=True, conf=0.40, verbose=False)
            coca, lavie = 0, 0
            for r in res:
                if r.boxes:
                    for b in r.boxes:
                        if int(b.cls[0]) == 0: coca += 1
                        else: lavie += 1
            if check_empty:
                if coca == 0 and lavie == 0: success += 1
                else: success = 0
            else:
                ec, el = max(0, exp_coca), max(0, exp_lavie)
                if ec==0 and el==0: el=1
                if coca >= ec and lavie >= el: success += 1
                else: success = 0
            if success >= 5:
                cap.release()
                return True
        cap.release()
        return False

    def worker_loop(self):
        while not rospy.is_shutdown():
            try:
                task = self.task_queue.get(timeout=2.0)
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
                    self.task_queue.put({"type": "RETURN_HOME", "target": "sac"})

    def execute_task(self, task, cancel_event=None):
        ttype, target = task["type"], task["target"]
        
        if ttype == "GUEST_CALL":
            mir_tts.speak_on_mir(f"Đã nhận lệnh gọi từ {target}, xe đang di chuyển tới.")
            if self.servo: self.servo.set_angle(90)
            ok = self.move_to_static_goal(target, cancel_event=cancel_event)
            if not ok: return
            
            if self.servo: self.servo.set_angle(20)
            mir_tts.speak_on_mir("Chào quý khách, khách nào order thì giơ tay lên.")
            
            self.target_locked_coords = None
            self.scanning_event.clear()
            self.video_thread.is_scanning_for_hand = True
            
            if self.scanning_event.wait(timeout=20.0):
                self.video_thread.is_scanning_for_hand = False
                tx, ty, obj = self.target_locked_coords
                self.saved_locations[target] = (tx, ty)
                
                self.nav_arrived_event.clear()
                self.calculate_hybrid_safe_goal(tx, ty, obj)
                self.nav_arrived_event.wait()
                
                self.current_location = "specific_" + target
                mir_tts.speak_on_mir("Mời khách order.")
                self.wait_event.clear()
                self.pub_arrived.publish(json.dumps({"action": "popup_menu", "ban": target}))
                
                start_wait = time.time()
                ordered = False
                while time.time() - start_wait < 45.0:
                    if target in self.active_orders: ordered = True; break
                    self.wait_event.wait(timeout=1.0)
                    self.wait_event.clear()
                
                if ordered: mir_tts.speak_on_mir("Đã nhận order, vui lòng đợi món.")
                else: mir_tts.speak_on_mir("Hết thời gian order, robot xin phép rời đi.")
            else:
                self.video_thread.is_scanning_for_hand = False
                mir_tts.speak_on_mir("Không thấy ai giơ tay, robot xin phép quay về.")

        elif ttype == "RETURN_HOME":
            mir_tts.speak_on_mir("Robot đang quay về vị trí sạc.")
            if self.servo: self.servo.set_angle(90)
            self.move_to_static_goal("sac", cancel_event=cancel_event)

        elif ttype == "KITCHEN_CALL":
            mir_tts.speak_on_mir("Đang quay về bếp để nhận món.")
            if self.servo: self.servo.set_angle(90)
            ok = self.move_to_static_goal("bep", cancel_event=cancel_event)
            if ok: mir_tts.speak_on_mir("Mời bếp đặt đồ lên xe và bấm nút xác nhận.")

        elif ttype == "DELIVER":
            if self.servo: self.servo.set_angle(60)
            exp_coca = self.active_orders.get(target, {}).get("coca", 0)
            exp_lavie = self.active_orders.get(target, {}).get("lavie", 0)
            
            mir_tts.speak_on_mir("Đang kiểm tra đồ uống trên khay.")
            has_items = self.verify_tray(exp_coca, exp_lavie)
            if not has_items:
                mir_tts.speak_on_mir("Hình như đặt thiếu đồ, xin bếp kiểm tra lại.")
                return 
            else:
                mir_tts.speak_on_mir("Đồ uống đã đủ, robot bắt đầu đi giao.")
                
            if self.servo: self.servo.set_angle(90)
            
            if target in self.saved_locations:
                tx, ty = self.saved_locations[target]
                self.nav_arrived_event.clear()
                self.calculate_hybrid_safe_goal(tx, ty, None)
                self.nav_arrived_event.wait()
                self.current_location = "specific_" + target
            else:
                ok = self.move_to_static_goal(target, cancel_event=cancel_event)
                if not ok: return
                
            mir_tts.speak_on_mir(f"Đã tới nơi. Mời khách lấy đồ uống.")
            time.sleep(10)
            mir_tts.speak_on_mir("Chúc quý khách ngon miệng.")
            self.active_orders.pop(target, None)

REPLACE_ME_HYBRID
"""

main_app = main_app.replace('REPLACE_ME_HYBRID', hybrid_algo)

videothread = videothread.replace('if track_id in raising_hands_ids:', 'if getattr(self, "is_scanning_for_hand", False) and track_id in raising_hands_ids:')

main_app = main_app.replace('final_success = True\n                        break', 'final_success = True\n                        self.nav_arrived_event.set()\n                        break')
main_app = main_app.replace('nav.ws_send_goal(self.robot, diem_dong)', 'nav.ws_send_goal(self.robot, diem_dong)\n                self.nav_arrived_event.set()')

run_block = """
def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
"""

full_code = imports_and_hacks + utils + maplabel + videothread + main_app + run_block

with open('/home/tuanminh/mir_project/src/mir_robot/tm/mainv4.py', 'w') as f:
    f.write(full_code)

print("Đã tạo mainv4.py thành công!")
