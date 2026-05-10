#!/usr/bin/env python3
import rospy
from move_base_msgs.msg import MoveBaseActionGoal
from actionlib_msgs.msg import GoalID

def cb_goal(msg):
    print("--- GOAL ---", flush=True)
    print(f"Goal Stamp: {msg.header.stamp.to_sec()}", flush=True)
    print(f"Goal ID Stamp: {msg.goal_id.stamp.to_sec()}", flush=True)
    print(f"Node Time.now(): {rospy.Time.now().to_sec()}", flush=True)

def cb_cancel(msg):
    print("--- CANCEL ---", flush=True)
    print(f"Cancel Stamp: {msg.stamp.to_sec()}", flush=True)

rospy.init_node("debug_time")
rospy.Subscriber("/move_base/goal", MoveBaseActionGoal, cb_goal)
rospy.Subscriber("/move_base/cancel", GoalID, cb_cancel)
rospy.spin()
