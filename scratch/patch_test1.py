import sys

with open("src/mir_robot/tm/test1.py", "r") as f:
    content = f.read()

start_marker = "        # ============================================================\n        # THUẬT TOÁN CANDIDATE RING SCORING (Thay thế hoàn toàn Raycast)\n        # ============================================================"
end_marker = "    def closeEvent"

if start_marker not in content:
    print("Cannot find start_marker!")
    sys.exit(1)

part1 = content.split(start_marker)[0]
part2 = "    def closeEvent" + content.split(end_marker)[1]

new_block = """        # === THUẬT TOÁN RAYCAST ĐỖ CHÉO ĐỘNG (Dựa theo test_map_click.py) ===
        min_dist_normal = float('inf')
        theta_normal = 0.0
        max_ray_len = int(3.0 / res)
        
        self.map_label.cone_pixels = []
        
        # Bước 1: Quét 360 độ tìm Hướng Trực Diện (Pháp tuyến - Normal) của Bàn
        ray_distances = []
        for angle in range(0, 360, 5):
            rad = math.radians(angle)
            dist = float('inf')
            for step in range(1, max_ray_len):
                cx = int(px_t + step * math.cos(rad))
                cy = int(py_t + step * math.sin(rad))
                if not (0 <= cx < w and 0 <= cy < h):
                    break
                self.map_label.cone_pixels.append((cx, h - cy - 1))
                if inflated_obs[cy, cx] == 0: # Tìm thấy điểm không bị chạm vùng bơm phồng
                    dist = step
                    break
            if dist != float('inf'):
                ray_distances.append((rad, dist))
                    
        if not ray_distances:
            print("[SMART NAV] ❌ THẤT BẠI: Kẹt hoàn toàn! Khách hàng bị bọc kín trong vùng cấm.")
            self.map_label.target_px = (int(px_t), h - int(py_t) - 1)
            self.map_label.goal_px = None
            self.map_label.update_view()
            return
            
        # Tìm tâm của vùng không gian trống để có vector pháp tuyến chuẩn nhất
        min_dist_normal = min(d for r, d in ray_distances)
        valid_angles = [r for r, d in ray_distances if d <= min_dist_normal + 2] # sai số 2 pixels
        sum_x = sum(math.cos(r) for r in valid_angles)
        sum_y = sum(math.sin(r) for r in valid_angles)
        theta_normal = math.atan2(sum_y, sum_x)
            
        # Bước 2 & 3: Quét không gian để tìm bên nào thoáng hơn (Dựa vào lõi bàn thực tế)
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
            theta_dock = theta_normal - math.radians(45)
            print(f"[SMART NAV] ↪️ Không gian PHẢI thoáng hơn (L={obs_left}, R={obs_right}). Đỗ chéo ra PHẢI (góc 45 độ).")
        else:
            theta_dock = theta_normal + math.radians(45)
            print(f"[SMART NAV] ↪️ Không gian TRÁI thoáng hơn (L={obs_left}, R={obs_right}). Đỗ chéo ra TRÁI (góc 45 độ).")
            
        # Bước 4: Tính toán target_dist_m ĐỘNG dọc theo tia đỗ
        free_start_step = None
        free_end_step = None
        self.map_label.ray_pixels = []
        
        for step in range(1, max_ray_len): # Quét tối đa 3.0m dọc tia đỗ
            cx = int(px_t + step * math.cos(theta_dock))
            cy = int(py_t + step * math.sin(theta_dock))
            
            if not (0 <= cx < w and 0 <= cy < h):
                if free_start_step is not None and free_end_step is None:
                    free_end_step = step
                break
                
            self.map_label.ray_pixels.append((cx, h - cy - 1))
            
            if inflated_obs[cy, cx] == 0: # Không có vật cản (an toàn tuyệt đối)
                if free_start_step is None:
                    free_start_step = step
            else: # Bị chặn bởi tường/vật cản khác
                if free_start_step is not None:
                    free_end_step = step
                    break
                    
        best_pose_px = None
        goal_yaw = 0.0
        
        if free_start_step is not None:
            if free_end_step is None:
                free_end_step = max_ray_len
                
            # Đặt xe vào ngay khoảng giữa vùng an toàn dọc theo tia
            target_step = int((free_start_step + free_end_step) / 2)
            target_dist_m = target_step * res
            
            # Ép cự ly nằm trong khoảng vàng (0.65m - 0.85m)
            if target_dist_m < 0.65:
                target_step = int(0.65 / res)
            elif target_dist_m > 0.85:
                target_step = int(0.85 / res)
                
            # Double-check không để target bị đè vào vùng cấm
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
            print(f"[SMART NAV] ✅ Chốt điểm đỗ cực mạnh: Cách khách {actual_dist * res:.2f}m")
        else:
            print("[SMART NAV] ❌ THẤT BẠI: Tia đỗ bị chặn hoàn toàn từ trong ra ngoài!")
            self.map_label.target_px = (int(px_t), h - int(py_t) - 1)
            self.map_label.goal_px = None
            self.map_label.update_view()
            return
                
        # 5. GỬI LỆNH ĐIỀU HƯỚNG
        final_px_x, final_px_y = best_pose_px
        goal_w_x = ox + final_px_x * res
        goal_w_y = oy + final_px_y * res
        
        self.map_label.target_px = (int(px_t), h - int(py_t) - 1)
        self.map_label.goal_yaw = goal_yaw
        self.map_label.goal_px = (int(final_px_x), h - int(final_px_y) - 1)
        self.map_label.update_view()
        
        print(f"[SMART NAV] 🎯 Tọa độ gửi MiR (X,Y) = ({goal_w_x:.2f}, {goal_w_y:.2f}), Yaw = {math.degrees(goal_yaw):.1f}°")
        
        q = tf.transformations.quaternion_from_euler(0, 0, goal_yaw)
        diem_dong = {"x": goal_w_x, "y": goal_w_y, "qz": q[2], "qw": q[3], "arrive_dist": 0.15}
        
        print(f"🚀 [NAV] Bắn lệnh tới MiR Fleet / MoveBase!")
        
        self.current_goal = (goal_w_x, goal_w_y)
        self.is_moving = True
        
        rest_ok = False
        if hasattr(self, 'mir_headers') and self.mir_headers:
            rest_ok = nav.api_navigate(self.mir_headers, [diem_dong], "diem_dong")
        if not rest_ok and self.robot:
            nav.ws_send_goal(self.robot, diem_dong)

"""

new_content = part1 + new_block + part2

with open("src/mir_robot/tm/test1.py", "w") as f:
    f.write(new_content)

with open("src/mir_robot/tm/test1v.py", "w") as f:
    f.write(new_content)

print("Patched successfully!")
