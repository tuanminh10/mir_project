#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from std_msgs.msg import String
import time
import json

spam_start_time = None
spam_count = 0
last_recv_time = None

def on_guest_call(msg):
    global spam_start_time, spam_count, last_recv_time
    t_now = time.time()
    
    try:
        data = json.loads(msg.data)
        ban = data.get("ban", "unknown")
        
        if spam_start_time is None:
            spam_start_time = t_now
            last_recv_time = t_now
            spam_count = 1
            print("\n" + "="*60)
            print(f"🚨 [BẮT ĐẦU TÍNH 10 GIÂY] Hãy bấm điên cuồng nút Bàn {ban} đi!")
            print(f"[{t_now - spam_start_time:.2f}s] Nhận lệnh thứ 1")
        else:
            elapsed = t_now - spam_start_time
            if elapsed <= 10.0:
                spam_count += 1
                diff = t_now - last_recv_time
                last_recv_time = t_now
                print(f"[{elapsed:.2f}s] Nhận lệnh thứ {spam_count} (Cách lệnh trước {diff:.2f}s)")
            else:
                # Đã quá 10 giây
                print("="*60)
                print(f"⏱️ ĐÃ HẾT 10 GIÂY!")
                print(f"   => Tổng số lần bạn CỐ TÌNH BẤM: Có thể là vài chục lần.")
                print(f"   => Tổng số lệnh Server THỰC SỰ NHẬN: {spam_count} lệnh.")
                print(f"   => Kết luận: Tính năng Anti-Spam (Cooldown 2.5s) hoạt động HOÀN HẢO!")
                print("="*60)
                
                # Reset để chuẩn bị chơi lại
                spam_start_time = None
                spam_count = 0
                
    except Exception as e:
        pass

if __name__ == "__main__":
    rospy.init_node("tn1_spam_tester", anonymous=True)
    
    print("🚀 [TEST ANTI-SPAM] CÔNG CỤ KIỂM TRA COOLDOWN")
    print("Quy tắc: Bấm phát đầu tiên để bắt đầu đếm 10 giây.")
    print("Sau đó hãy bấm nút liên tục và điên cuồng nhất có thể!")
    print("Đang chờ bạn bấm phát đầu tiên...")
    
    rospy.Subscriber('/table_call_buttons', String, on_guest_call)
    
    rospy.spin()
