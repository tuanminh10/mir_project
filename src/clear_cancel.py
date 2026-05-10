#!/usr/bin/env python3
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction

rospy.init_node("kill_cancel")
client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
client.wait_for_server()
# Gửi một Cancel với timestamp (0,0) - tức là hủy tất cả MỌI LÚC trước đó.
# Mặc dù ActionLib đã lưu time, nhưng nếu có 1 goal ID bị kẹt, ta đè nó. 
client.cancel_all_goals()
print("Done")
