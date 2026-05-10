#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
import time

rospy.init_node("test_cmd_vel")
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
time.sleep(1)

msg = Twist()
msg.linear.x = 0.2
for _ in range(10):
    pub.publish(msg)
    time.sleep(0.1)

msg.linear.x = 0.0
pub.publish(msg)
print("TEST DONE")
