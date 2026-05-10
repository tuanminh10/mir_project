#!/usr/bin/env python3
import rospy
from move_base_msgs.msg import MoveBaseActionGoal

def cb(msg):
    print("--- RAW SUB GOAL ---", flush=True)
    print(msg.goal.target_pose.pose.position, flush=True)

rospy.init_node("monitor")
rospy.Subscriber("/move_base/goal", MoveBaseActionGoal, cb)
rospy.spin()
