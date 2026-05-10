#!/usr/bin/env python3
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

rospy.init_node("test_goal")
client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
client.wait_for_server()
x,y,z,w = 16.1, 15.5, 0, 1

goal = MoveBaseGoal()
goal.target_pose.header.frame_id = "map"
goal.target_pose.header.stamp = rospy.Time.now()
goal.target_pose.pose.position.x = x
goal.target_pose.pose.position.y = y
goal.target_pose.pose.orientation.z = z
goal.target_pose.pose.orientation.w = w

def cb(*args):
    print(args)

client.send_goal(goal, done_cb=cb)
print("GOAL SENT")
rospy.spin()
