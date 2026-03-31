#!/usr/bin/env python3
import rospy
import json
import threading
import queue
from std_msgs.msg import String

# Import trực tiếp các hàm từ navigationcacdiem
import navigationcacdiem as nav

class MainControlMinimal:
    def __init__(self):
        rospy.init_node('main_control_v1', anonymous=True)
        rospy.loginfo("[Main_v1] Đang kết nối duy trì sẵn với Robot (ROSBridge & REST)...")
        
        # Kết nối 1 lần duy nhất lúc bật
        self.robot = nav.ws_connect()
        self.headers = nav.api_login()
        if self.headers:
            nav.api_ensure_ready(self.headers)
            rospy.loginfo("[Main_v1] REST API & ROSBridge đã sẵn sàng!")
        else:
            rospy.logwarn("[Main_v1] REST API không khả dụng! Chỉ dùng ROSBridge.")
        
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

            # 2. Gửi tín hiệu xoá toàn bộ hàng đợi của MiR bằng REST API
            if self.headers:
                import requests
                requests.delete(f"{nav.API_URL}/mission_queue", headers=self.headers, timeout=3)
                
                # 3. Chuyển state về Pause (4) để robot dừng hẳn ngay lập tức, xoá đà đang đi
                nav.api_set_state(self.headers, 4)
                
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
    # 2. KHÁCH ĐẶT MÓN TRÊN WEB (CHỈ LOG)
    # ==========================================
    def on_web_order_received(self, msg):
        try:
            order_data = json.loads(msg.data)
            ban_dat_do = str(order_data.get("ban", "?")).strip()
            if ban_dat_do.isdigit(): 
                ban_dat_do = f"ban {ban_dat_do}"
            
            # Khách order xong (trên Web gửi xuống) chỉ việc in log báo cho Bếp biết
            rospy.loginfo(f"[Main_v1] 🛎️ ĐƠN ẢO TỪ WEB: '{ban_dat_do}' vừa đặt món! Chờ Bếp thao tác trên Web Bếp.")
            
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

    def execute_task(self, task, cancel_event=None):
        task_type = task["type"]
        target = task["target"]
        
        rospy.loginfo(f">>> BẮT ĐẦU TASK: {task_type} -> Mục tiêu: {target}")
        
        # 1. DI CHUYỂN
        if target != self.current_location:
            nav.handle_command(target, self.robot, self.headers, non_interactive=True, cancel_event=cancel_event)
            if cancel_event and cancel_event.is_set():
                rospy.loginfo(f"[Worker] Tạm hủy di chuyển đến '{target}' để ưu tiên lệnh mới.")
                self.current_location = "interrupted"
                return # Thoát thực thi ngay lập tức
            self.current_location = target
        else:
            rospy.loginfo(f"[Worker] Khỏi cần đi, Robot đang ở sẵn '{target}'.")

        # 2. HÀNH ĐỘNG TẠI ĐIỂM ĐẾN
        self.wait_event.clear() # Đặt lại sự kiện ngắt

        if task_type == "GUEST_CALL":
            # Gửi tín hiệu để bật Menu Web cho khách
            arrived_msg = json.dumps({"action": "popup_menu", "ban": target})
            self.pub_arrived.publish(arrived_msg)
            
            rospy.loginfo(f"[Worker] Đã mở Menu. Đứng chờ khách Order tối đa {self.WAIT_TIME_GUEST} giây...")
            # Đứng đợi, nhưng nếu tự order qua web hoặc nút xong thì sẽ break sớm
            is_released_early = self.wait_event.wait(timeout=self.WAIT_TIME_GUEST)
            
            if is_released_early:
                rospy.loginfo("[Worker] Khách ĐÃ ORDER XONG sớm (hoặc bấm trả robot)! Lập tức kết thúc lệnh.")
            else:
                rospy.loginfo(f"[Worker] Hết {self.WAIT_TIME_GUEST}s Timeout. Kết thúc lệnh, dọn dẹp để phục vụ task khác.")

        elif task_type == "KITCHEN_CALL":
            rospy.loginfo(f"[Worker] Đã tới Bếp! Mở thời gian chờ {self.WAIT_TIME_KITCHEN_LOAD} giây cho đầu bếp chất đồ...")
            rospy.sleep(self.WAIT_TIME_KITCHEN_LOAD) # Sleep cứng cho bếp chuẩn bị đồ
            rospy.loginfo("[Worker] Chuẩn bị khởi hành nếu có lệnh Deliver tiếp theo...")

        elif task_type == "DELIVER":
            rospy.loginfo(f"[Worker] Đã giao xong đồ cho '{target}'. Chờ khách lấy đồ xuống khoảng {self.WAIT_TIME_DELIVER} giây...")
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
