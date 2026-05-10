import rospy
from nav_msgs.msg import Odometry

rospy.init_node('test_monkey')

original_now = rospy.Time.now
time_offset = rospy.Duration(0)

def custom_now():
    return original_now() + time_offset

rospy.Time.now = custom_now

print("Original now:", original_now().to_sec())

msg = rospy.wait_for_message('/odom', Odometry, timeout=5.0)
time_offset = msg.header.stamp - original_now()

print("Offset:", time_offset.to_sec())
print("Custom now:", rospy.Time.now().to_sec())
print("Robot Odometry time:", msg.header.stamp.to_sec())
