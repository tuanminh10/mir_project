import sys
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

def test():
    rospy.init_node('test_actionlib', anonymous=True)
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    rospy.loginfo("Waiting for server...")
    client.wait_for_server()
    rospy.loginfo("Server found. Sending goal...")
    g = MoveBaseGoal()
    g.target_pose.header.frame_id = "map"
    g.target_pose.header.stamp = rospy.Time.now()
    g.target_pose.pose.position.x = 10.5
    g.target_pose.pose.position.y = 20.3
    g.target_pose.pose.orientation.w = 1.0
    client.send_goal(g)
    client.wait_for_result(rospy.Duration(5.0))
    state = client.get_state()
    rospy.loginfo(f"State: {state}")

test()
