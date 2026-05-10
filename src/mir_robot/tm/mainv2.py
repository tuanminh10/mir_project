#!/usr/bin/env python3
import rospy
import json
import threading
import queue
import time
import os
import cv2
from ultralytics import YOLO
from std_msgs.msg import String

# ==============================================================================
# Bật chế độ YOLO Offline
os.environ['YOLO_OFFLINE'] = 'True' 
# ==============================================================================

# Import trực tiếp các hàm từ navigationcacdiem
import navigationcacdiem as nav
import mir_tts  # Import bộ tổng hợp giọng nói TTS

class MainControlMinimal:
    def __init__(self):
        rospy.init_node('main_control_v1', anonymous=True)
        rospy.loginfo("[Main_v1] KHỞI ĐỘNG HỆ THỐNG ACTIONLIB (ROS THUẦN)...")
        
        # --- TẢI MODEL YOLO SẴN NHƯNG ÉP CHẠY BẰNG CHIP CPU BÊN TRONG DOCKER ---
        # Kiểm tra cả 2 đường dẫn (tuyệt đối của máy thực và tương đối trong Docker)
        abs_path = '/home/tuanminh/mir_project/src/mir_robot/tm/best/best.pt'
        rel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best', 'best.pt')
        
        model_path = abs_path if os.path.exists(abs_path) else rel_path
        
        if os.path.exists(model_path):
            rospy.loginfo(f"⏳ Đang tải AI Model (YOLO) vào bộ nhớ tại {model_path}...")
            self.yolo_model = YOLO(model_path)
        else:
            self.yolo_model = None
            rospy.logwarn(f"❌ Không tìm thấy model YOLO, tính năng AI sẽ bị bỏ qua!")
            
        # --- LƯU TRỮ ĐƠN HÀNG ĐỂ KIỂM TRA ---
        self.active_orders = {} # dict lưu: "ban 3" -> {"coca": x, "lavie": y}
        
        # YÊU CẦU: PURE ROS ACTIONLIB, KHÔNG WEBSOCKET
        self.robot = None
        self.headers = nav.api_login()
        
        import actionlib
        from move_base_msgs.msg import MoveBaseAction
        rospy.loginfo("⏳ Đang khởi tạo kết nối SimpleActionClient move_base...")
        self.action_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        
        if self.headers:
            import requests
            requests.delete(f"{nav.API_URL}/mission_queue", headers=self.headers, timeout=2)
            nav.api_set_state(self.headers, 3)
            rospy.loginfo("🔓 Đã mở phanh Dashboard (State 3) để ActionLib hoạt động!")
        
        # --- HỆ THỐNG CƠ CHẾ HÀNG ĐỢI (QUEUE) MỚI ---
        self.task_queue = queue.Queue()
        # Biến để theo dõi vị trí hiện tại tránh đi lại 1 chỗ
        self.current_location = "sac" 
        
        # Sự kiện dùng để có thể ngắt thời gian chờ sớm (Ví dụ: khách bấm "Xong" sẽ bỏ qua thời gian Timeout)
        self.wait_event = threading.Event()
        
        # Sự kiện dùng để hủy lệnh di chuyển về sạc ngay lập tức khi có lệnh khác
        self.charging_cancel_event = threading.Event()
        
        # Subscribe tín hiệu nút bấm từ ESP32 (Khách gọi)
        self.sub_button = rospy.Subscriber('/table_call_buttons', String, self.on_button_pressed)
        
        # Subscribe đơn hàng từ Web khách (Chỉ log, không trigger chạy)
        self.sub_order = rospy.Subscriber('/robot_orders', String, self.on_web_order_received)
        
        # Lắng nghe tín hiệu trả robot từ Web (Để ngắt thời gian chờ sớm - Điều kiện A)
        self.sub_release = rospy.Subscriber('/robot_release', String, self.on_robot_release)
        
        # Subscribe LỆNH TỪ NHÀ BẾP (Web Bếp gọi xe hoặc giao hàng)
        self.sub_kitchen = rospy.Subscriber('/kitchen_commands', String, self.on_kitchen_command)
        
        self.pub_arrived = rospy.Publisher('/robot_arrived_table', String, queue_size=10)
        
        # Khởi chạy Worker xử lý hàng đợi dưới nền
        self.worker_thread = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker_thread.start()
        
        # Bắt sự kiện khi dùng Ctrl+C tắt chương trình
        rospy.on_shutdown(self.on_shutdown_hook)
        
        rospy.loginfo("[Main_v1] HOÀN TẤT KHỞI ĐỘNG CÙNG LOGIC: WEB BẾP + HÀNG ĐỢI + TIMEOUT + ĐIỂM CHỜ.")

    def on_shutdown_hook(self):
        rospy.logwarn("\n[!!!] BẠN VỪA NHẤN CTRL+C. ĐANG DỪNG KHẨN CẤP ROBOT VÀ XÓA LỆNH...")
        try:
            # 1. Trút bỏ toàn bộ hàng đợi của luồng Python
            while not self.task_queue.empty():
                try: self.task_queue.get_nowait()
                except: pass

            # 2. Gửi tín hiệu cancel cho actionlib
            if hasattr(self, 'action_client'):
                self.action_client.cancel_all_goals()
                
            rospy.loginfo("[Main_v1] Đã xóa hàng đợi. Robot dừng an toàn. Tạm biệt!")
            import os
            os._exit(0) # Ngắt cứng 100% tất cả các luồng đang chạy ngầm
        except Exception as e:
            rospy.logerr(f"Lỗi khi dừng khẩn cấp: {e}")
            import os
            os._exit(1)

    # ==========================================
    # 1. NHẬN LỆNH TỪ KHÁCH BẤM NÚT (Hoặc Web gọi)
    # ==========================================
    def on_button_pressed(self, msg):
        try:
            data = json.loads(msg.data)
            raw_ban = data.get("ban")
            if not raw_ban: return

            target_table = str(raw_ban).strip()
            if target_table.isdigit(): 
                target_table = f"ban {target_table}"
                
            rospy.loginfo(f"[Queue] Nhận lệnh KHÁCH GỌI từ '{target_table}'. Đẩy vào hàng đợi.")
            self.task_queue.put({"type": "GUEST_CALL", "target": target_table})
            self.charging_cancel_event.set() # Hủy về sạc nếu đang chạy
        except Exception as e:
            rospy.logerr(f"Lỗi nút bấm: {e}")

    # ==========================================
    # 2. KHÁCH ĐẶT MÓN TRÊN WEB (CHỈ LOG VÀ LƯU DATA ĐỂ AI KIỂM TRA)
    # ==========================================
    def on_web_order_received(self, msg):
        try:
            order_data = json.loads(msg.data)
            ban_dat_do = str(order_data.get("ban", "?")).strip()
            if ban_dat_do.isdigit(): 
                ban_dat_do = f"ban {ban_dat_do}"
            
            # Đọc số lượng Coca và Lavie từ Json (mặc định 0)
            coca_qty = int(order_data.get("coca", 0))
            lavie_qty = int(order_data.get("lavie", 0))
            
            # Lưu vào bộ nhớ để check hàng tại bếp
            self.active_orders[ban_dat_do] = {"coca": coca_qty, "lavie": lavie_qty}
            
            # Khách order xong (trên Web gửi xuống) chỉ việc in log báo cho Bếp biết
            rospy.loginfo(f"[Main_v1] 🛎️ ĐƠN ẢO TỪ WEB: '{ban_dat_do}' vừa đặt món! (Coca: {coca_qty}, Lavie: {lavie_qty}). Chờ Bếp thao tác.")
            
            # Robot thông báo xác nhận đơn hàng
            mir_tts.speak_on_mir("Đơn hàng của quý khách đã được xác nhận.")
            
            # Kích hoạt sự kiện để Robot có thể NGAY LẬP TỨC rời đi (nếu đang đứng ở bàn đó)
            self.wait_event.set()
        except Exception as e:
            rospy.logerr(f"Lỗi nhận đơn: {e}")

    # Tín hiệu khi khách bấm nút "Trả Robot" (Cho robot đi)
    def on_robot_release(self, msg):
        rospy.loginfo("[Main_v1] Nhận tín hiệu TRẢ ROBOT. Sẽ rời đi ngay lập tức.")
        self.wait_event.set()

    # ==========================================
    # 3. NHÀ BẾP PHÁT LỆNH QUA WEB BẾP
    # ==========================================
    def on_kitchen_command(self, msg):
        try:
            # JSON format: {"action": "call_robot"} hoặc {"action": "deliver", "table": "ban 3"}
            data = json.loads(msg.data)
            action = data.get("action")
            
            if action == "call_robot":
                rospy.loginfo("[Queue] Web Bếp nhấn GỌI ROBOT. Đẩy vào hàng đợi.")
                self.task_queue.put({"type": "KITCHEN_CALL", "target": "bep"})
                self.charging_cancel_event.set() # Hủy về sạc nếu đang chạy
                
            elif action == "deliver":
                target_table = str(data.get("table", "")).strip()
                if target_table.isdigit(): 
                    target_table = f"ban {target_table}"
                
                rospy.loginfo(f"[Queue] Web Bếp báo NẤU XONG, YÊU CẦU GIAO ĐẾN '{target_table}'.")
                # Đẩy 2 bước vào hàng đợi: 1 là tới bếp lấy (nếu chưa ở bếp), 2 là đi giao
                if self.current_location != "bep":
                    self.task_queue.put({"type": "KITCHEN_CALL", "target": "bep"})
                self.task_queue.put({"type": "DELIVER", "target": target_table})
                self.charging_cancel_event.set() # Hủy về sạc nếu đang chạy
                
        except Exception as e:
            rospy.logerr(f"Lỗi lệnh bếp: {e}")

    # ==========================================
    # 4. VỊ TRÍ CHỜ MẶC ĐỊNH & THỜI GIAN THIẾT LẬP
    # ==========================================
    FALLBACK_HOME = "sac"  # Đổi thành "sac" để robot tự về sạc khi rảnh rỗi
    WAIT_TIME_GUEST = 45   # Giây đứng đợi ở bàn khách khi khách gọi
    WAIT_TIME_DELIVER = 10 # Giây đứng đợi ở bàn khách khi giao đồ (Đổi thành 10s theo yêu cầu)
    WAIT_TIME_KITCHEN_LOAD = 10 # Thời gian trễ để bếp chất đồ lên xe sau khi gọi

    # ==========================================
    # 5. VÒNG LẶP XỬ LÝ TASK (WORKER THREAD)
    # ==========================================
    def worker_loop(self):
        while not rospy.is_shutdown():
            try:
                # Cố gắng lấy task trong 2 giây
                task = self.task_queue.get(timeout=2.0)
                if task["type"] == "RETURN_HOME":
                    self.charging_cancel_event.clear()
                    self.execute_task(task, cancel_event=self.charging_cancel_event)
                else:
                    self.execute_task(task)
                self.task_queue.task_done()
                
            except queue.Empty:
                # QUEUE RỖNG: Không có task nào trong hàng đợi
                # Khi rảnh rỗi, chỉ tự về sạc nếu đang ở Bàn Khách. Nếu đang đậu ở 'bep' thì phải tiếp tục đỗ chờ lệnh giao hàng!
                if self.current_location not in [self.FALLBACK_HOME, "moving_to_" + self.FALLBACK_HOME, "bep"]:
                    rospy.loginfo(f"[System] Rảnh rỗi. Khách đã xong dọn mâm. Tự động lùi về cắm Dock sạc: '{self.FALLBACK_HOME}'.")
                    # Gán một trạng thái tạm để ngăn việc liên tục đẩy thêm lệnh sạc vào queue khi robot còn đang đi cất
                    self.current_location = "moving_to_" + self.FALLBACK_HOME
                    self.task_queue.put({"type": "RETURN_HOME", "target": self.FALLBACK_HOME})
                    
            except Exception as e:
                rospy.logerr(f"[Worker] Lỗi queue: {e}")

    # KHỐI LOGIC THÊM: HÀM AI NHẬN DIỆN VÀ KIỂM ĐỒ TRƯỚC KHI ĐI GIAO (CHẠY THẲNG TRONG DOCKER BẰNG CPU)
    def verify_items_with_ai(self, expected_coca, expected_lavie):
        if not self.yolo_model:
            rospy.logwarn("⚠️ AI không khả dụng (model bị lỗi hoặc thiếu file). Chạy thẳng cho an toàn.")
            return True
            
        if expected_coca <= 0 and expected_lavie <= 0:
            rospy.logwarn("⚠️ Đơn trên Web bị 0 món. Robot tự chuyển sang chế độ Demo: Bắt bù phải có MỘT CHAI (Coca/Lavie) MỚI CHẠY!")
            expected_coca = 0
            expected_lavie = 1 
            
        rospy.loginfo(f"📸 [AI] Đang mở Camera chạy 30s... Yêu cầu: Coca={expected_coca}, Lavie={expected_lavie}")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            rospy.logerr("❌ AI Không thể mở Camera!")
            return False
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        start_time = rospy.Time.now()
        timeout = rospy.Duration(30.0) 
        success_frames = 0
        
        while (rospy.Time.now() - start_time) < timeout:
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            
            # Đã bỏ cờ device='cpu'. Hệ thống sẽ tự động bắt GPU (CUDA) siêu mạnh của bạn!
            results = self.yolo_model.track(frame, persist=True, stream=True, conf=0.40, iou=0.45, imgsz=640, verbose=False)
            
            count_coca = 0
            count_lavie = 0
            
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        if (x2 - x1) * (y2 - y1) < 1000: continue
                        cls = int(box.cls[0])
                        if cls == 0: count_coca += 1
                        elif cls == 1: count_lavie += 1

            # NẾU ĐƠN TRỐNG Đòi ít nhất 1 đồ bất kì
            if (expected_coca <= 0 and expected_lavie <= 0) and (count_coca + count_lavie >= 1):
                success_frames += 1
            # NẾU ĐƠN CÓ DATA THẬT
            elif (expected_coca > 0 or expected_lavie > 0) and (count_coca >= expected_coca and count_lavie >= expected_lavie):
                success_frames += 1
            else:
                success_frames = 0
                
            if success_frames >= 5: 
                rospy.loginfo("✅ [AI] Kiểm tra: ĐÃ ĐỦ ĐỒ RỒI! Bắt đầu hành trình chạy.")
                cap.release()
                return True

        cap.release()
        rospy.logwarn("❌ [AI] Hết 30 giây kiểm tra, VẪN THIẾU ĐỒ! Từ chối rời bếp!")
        return False
        
    def execute_task(self, task, cancel_event=None):
        task_type = task["type"]
        target = task["target"]
        
        rospy.loginfo(f">>> BẮT ĐẦU TASK: {task_type} -> Mục tiêu: {target}")
        
        # --- BƯỚC CHECK ĐỒ TRƯỚC KHI ĐI GIAO ---
        if task_type == "DELIVER" and self.current_location in ["bep", "moving_to_bep"]:
            # Lấy order hiện tại của bàn này (có thể Web vẫn 0 0, ta xử lý ở hàm verification_with_ai)
            order = self.active_orders.get(target, {"coca": 0, "lavie": 0})
            
            # GỌI VOICE & CHỜ
            mir_tts.speak_on_mir(f"Hệ thống đang kiểm tra tự động xem đã đủ số lượng món ăn hay chưa.")
            rospy.sleep(3) # đợi nói xong
                
            # Đẩy qua AI Check ngay trên Main Thread, camera sẽ bật 
            is_enough = self.verify_items_with_ai(order["coca"], order["lavie"])
            if not is_enough:
                mir_tts.speak_on_mir("Phát hiện số lượng món ăn chưa đủ, yêu cầu nhà bếp cho thêm hoặc chỉnh lại, robot tạm thời từ chối rời bếp.")
                rospy.loginfo(f"[Worker] Hủy chuyến giao đến {target} do AI ĐÃ TỪ CHỐI! -----------------")
                return # KHÔNG đi giao nữa
            else:
                mir_tts.speak_on_mir("Số lượng món ăn đã đủ. Robot bắt đầu đi giao.")
                rospy.sleep(3) # đợi nói xong đi

        # 1. DI CHUYỂN (PURE ACTIONLIB EXACTLY LIKE hung.py)
        if target != self.current_location:
            if target not in nav.DIEM:
                rospy.logwarn(f"⚠️ Điểm đến '{target}' không tồn tại trong map!")
                return
                
            diem = nav.DIEM[target]
            rospy.loginfo(f"🚀 [ROS MoveBase ActionLib]: Gửi mục tiêu {target} ({diem['x']}, {diem['y']}) tới action server...")
            
            # --- START ACTIONLIB LOGIC ---
            import actionlib
            from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
            from actionlib_msgs.msg import GoalStatus
            
            if getattr(self, 'action_client', None) is None:
                self.action_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
                
            self.action_client.wait_for_server()
            
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "map"
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.pose.position.x = diem["x"]
            goal.target_pose.pose.position.y = diem["y"]
            goal.target_pose.pose.orientation.z = diem["qz"]
            goal.target_pose.pose.orientation.w = diem["qw"]
            
            self.action_client.send_goal(goal)
            
            is_moving_to_goal = True
            target_reached = False
            
            while is_moving_to_goal and not rospy.is_shutdown():
                if cancel_event and cancel_event.is_set():
                    self.action_client.cancel_goal()
                    rospy.loginfo(f"⚠️ Đã hủy di chuyển bằng ActionLib tới '{target}'")
                    self.current_location = "interrupted"
                    return
                
                # Check status
                finished = self.action_client.wait_for_result(rospy.Duration(0.5))
                if finished:
                    state = self.action_client.get_state()
                    if state == GoalStatus.SUCCEEDED:
                        rospy.loginfo(f"✅ Robot reached the goal {target} using ActionLib")
                        target_reached = True
                        break
                    else:
                        rospy.logwarn(f"❌ Failed to reach {target}, state: {state}")
                        break

            # --- END ACTIONLIB LOGIC ---
            
            if target_reached:
                self.current_location = target
            else:
                return # Abort the rest of the task if we failed to reach
        else:
            rospy.loginfo(f"[Worker] Khỏi cần đi, Robot đang ở sẵn '{target}'.")

        # 2. HÀNH ĐỘNG TẠI ĐIỂM ĐẾN
        self.wait_event.clear() # Đặt lại sự kiện ngắt

        if task_type == "GUEST_CALL":
            # Gửi tín hiệu để bật Menu Web cho khách
            arrived_msg = json.dumps({"action": "popup_menu", "ban": target})
            self.pub_arrived.publish(arrived_msg)
            
            rospy.loginfo(f"[Worker] Đã mở Menu. Đứng chờ khách Order tối đa {self.WAIT_TIME_GUEST} giây...")
            mir_tts.speak_on_mir("Xin chào quý khách, xin mời quý khách order món ăn qua menu.")
            
            # Đứng đợi, nhưng nếu tự order qua web hoặc nút xong thì sẽ break sớm
            is_released_early = self.wait_event.wait(timeout=self.WAIT_TIME_GUEST)
            
            if is_released_early:
                rospy.loginfo("[Worker] Khách ĐÃ ORDER XONG sớm (hoặc bấm trả robot)! Lập tức kết thúc lệnh.")
            else:
                rospy.loginfo(f"[Worker] Hết {self.WAIT_TIME_GUEST}s Timeout. Kết thúc lệnh, dọn dẹp để phục vụ task khác.")

        elif task_type == "KITCHEN_CALL":
            rospy.loginfo(f"[Worker] Đã tới Bếp! Mở thời gian chờ {self.WAIT_TIME_KITCHEN_LOAD} giây cho đầu bếp chất đồ...")
            mir_tts.speak_on_mir("Đã tới bếp, yêu cầu đặt món lên khay.")
            rospy.sleep(self.WAIT_TIME_KITCHEN_LOAD) # Sleep cứng cho bếp chuẩn bị đồ
            rospy.loginfo("[Worker] Chuẩn bị khởi hành nếu có lệnh Deliver tiếp theo...")

        elif task_type == "DELIVER":
            rospy.loginfo(f"[Worker] Đã giao xong đồ cho '{target}'. Chờ khách lấy đồ xuống khoảng {self.WAIT_TIME_DELIVER} giây...")
            mir_tts.speak_on_mir("Đây là món ăn quý khách đã order, chúc quý khách có bữa ăn ngon miệng.")
            # Khách có thể lấy đồ xong và bấm "Trả" hoặc chờ hết giờ.
            self.wait_event.wait(timeout=self.WAIT_TIME_DELIVER)
            rospy.loginfo("[Worker] Hoàn tất chuyến giao hàng.")

        elif task_type == "RETURN_HOME":
            rospy.loginfo(f"[Worker] Đã về tới điểm chờ chuẩn bị ('{target}'). Đang Standby...")
            # Nằm yên, kết thúc task

if __name__ == '__main__':
    try:
        MainControlMinimal()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
