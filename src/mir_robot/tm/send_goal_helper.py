#!/usr/bin/env python3
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import Twist
import tf.transformations
import sys
import threading

current_goal = None
is_canceling = False

def read_input():
    global current_goal, is_canceling
    while not rospy.is_shutdown():
        line = sys.stdin.readline()
        if not line:
            rospy.signal_shutdown("EOF")
            break
        line = line.strip()
        if not line:
            continue
        
        if line == "CANCEL":
            current_goal = None
            is_canceling = True
            print("OK_CANCEL", flush=True)
        else:
            try:
                parts = line.split(',')
                x = float(parts[0])
                y = float(parts[1])
                yaw = float(parts[2]) if len(parts) > 2 else 0.0
                current_goal = (x, y, yaw)
                is_canceling = False
                print(f"ACK_GOAL_{x}_{y}", flush=True)
            except:
                pass

def main():
    global is_canceling, current_goal
    rospy.init_node('send_goal_daemon', anonymous=True)
    
    cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    threading.Thread(target=read_input, daemon=True).start()
    
    # 1. KHỞI TẠO CLIENT 1 LẦN DUY NHẤT NGOÀI VÒNG LẶP
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    print("Dang cho move_base server...", flush=True)
    client.wait_for_server()
    print("Da ket noi! San sang nhan lenh.", flush=True)
    
    active_goal = None

    while not rospy.is_shutdown():
        # XỬ LÝ HỦY LỆNH
        if is_canceling:
            if active_goal:
                client.cancel_goal()
            cmd_vel_pub.publish(Twist()) # Stop xe khẩn cấp
            is_canceling = False
            active_goal = None
            rospy.sleep(0.1)
            continue

        # XỬ LÝ GỬI LỆNH MỚI
        if current_goal and current_goal != active_goal:
            active_goal = current_goal
            
            x, y, yaw = active_goal
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "map"
            
            # 2. THỦ THUẬT HACK CLOCK: Cộng thêm 0.1s để chắc chắn Timestamp luôn "trẻ" hơn lệnh Cancel
            goal.target_pose.header.stamp = rospy.Time.now() + rospy.Duration(0.1)
            
            goal.target_pose.pose.position.x = x
            goal.target_pose.pose.position.y = y
            q = tf.transformations.quaternion_from_euler(0, 0, yaw)
            goal.target_pose.pose.orientation.x = q[0]
            goal.target_pose.pose.orientation.y = q[1]
            goal.target_pose.pose.orientation.z = q[2]
            goal.target_pose.pose.orientation.w = q[3]

            client.send_goal(goal)
            
            # Chờ kết quả hoặc bị ghi đè bởi tọa độ mới
            while not rospy.is_shutdown() and current_goal == active_goal and not is_canceling:
                if client.wait_for_result(rospy.Duration(0.1)):
                    break

            if current_goal != active_goal or is_canceling:
                continue
            
            # XỬ LÝ KẾT QUẢ TRẢ VỀ
            state = client.get_state()
            if state == GoalStatus.SUCCEEDED:
                print("STATE_3", flush=True)
                current_goal = None
                active_goal = None
            else:
                text = client.get_goal_status_text()
                print(f"ERROR_Bi Server tu choi Code {state} ({text}). DANG RETRY THEO CHUAN CUA HUNG.PY...", flush=True)
                rospy.sleep(0.5) # Giảm thời gian chờ Retry xuống cho mượt
                active_goal = None
        else:
            rospy.sleep(0.05)

if __name__ == '__main__':
    main()