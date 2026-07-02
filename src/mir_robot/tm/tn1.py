#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from std_msgs.msg import String
import json
import time
import csv
import os

CSV_FILE = "tn1_5_chenhlech.csv"

virtual_queue = []
first_arrival_time = None
test_counter = 0

def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["STT", "Thoi_gian_chenh_lech_ms", "Ket_qua"])
        print(f"📁 Đã tạo file kết quả mới: {CSV_FILE}")
    else:
        print(f"📁 Sẽ tiếp tục ghi thêm vào file cũ: {CSV_FILE}")

def log_to_csv(stt, diff_ms, result):
    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([stt, round(diff_ms, 1), result])

def on_guest_call(msg):
    global virtual_queue, first_arrival_time, test_counter
    
    t_recv = time.time() * 1000  # Thời gian hiện tại (milli-giây)
    
    try:
        data = json.loads(msg.data)
        ban = data.get("ban", "unknown")
        
        # Nếu đã có 1 bàn trong hàng đợi, nhưng phải chờ quá 2 giây (2000ms) 
        # chứng tỏ lần bấm trước bị lỗi hoặc bạn không bấm cùng lúc. 
        # Ta sẽ xóa đi để làm lại, tránh bị cộng dồn thời gian.
        if first_arrival_time is not None and (t_recv - first_arrival_time) > 2000:
            print(f"⚠️ [CẢNH BÁO] Quá 2 giây không thấy bàn kia bấm! Hủy kết quả cũ, đo lại.")
            virtual_queue.clear()
            first_arrival_time = None
            
        # Ngăn chặn trường hợp 1 bàn bấm 2 lần liên tục bị tính là 2 bàn
        if ban in virtual_queue:
            return
            
        virtual_queue.append(ban)
        
        if len(virtual_queue) == 1:
            # Bàn đầu tiên vừa tới
            first_arrival_time = t_recv
            print(f"============================================================")
            print(f"⏳ Tín hiệu Bàn {ban} đã vào hàng đợi... (Chờ tín hiệu còn lại)")
            
        elif len(virtual_queue) == 2:
            # Bàn thứ hai vừa tới
            time_diff = t_recv - first_arrival_time
            test_counter += 1
            
            # Ghi vào CSV
            log_to_csv(test_counter, time_diff, "Đạt")
            
            print(f"✅ Tín hiệu Bàn {ban} đã vào hàng đợi.")
            print(f"   => Lần test thứ {test_counter}: KẾT QUẢ = ĐẠT!")
            print(f"   => Thời gian chênh lệch khi vào Server: {time_diff:.1f} ms")
            print(f"   => Đã lưu vào {CSV_FILE}")
            print(f"============================================================\n")
            
            # Xóa hàng đợi để sẵn sàng cho lần đếm "1 2 3" tiếp theo
            virtual_queue.clear()
            first_arrival_time = None
            
    except Exception as e:
        print(f"[LỖI] parse JSON: {e}")

if __name__ == "__main__":
    rospy.init_node("tn1_latency_tester", anonymous=True)
    
    print("🚀 [KB 1.5] CÔNG CỤ ĐO THỜI GIAN CHÊNH LỆCH KHI BẤM ĐỒNG THỜI")
    init_csv()
    
    rospy.Subscriber('/table_call_buttons', String, on_guest_call)
    
    print("Hãy đếm 1,2,3 và bấm 2 nút cùng lúc!")
    print("Đang chờ tín hiệu...")
    
    rospy.spin()
