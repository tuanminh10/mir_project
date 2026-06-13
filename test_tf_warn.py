import os
import sys

rosconsole_conf_path = "/tmp/suppress_tf_warning.conf"
with open(rosconsole_conf_path, "w") as f:
    f.write("log4j.logger.ros.tf2=ERROR\n")
    f.write("log4j.logger.ros.tf=ERROR\n")
    f.write("log4j.logger.ros.tf2_ros=ERROR\n")
os.environ["ROSCONSOLE_CONFIG_FILE"] = rosconsole_conf_path

import rospy
import tf

rospy.init_node('test_tf_warn', anonymous=True)
print("ROS_CONFIG:", os.environ.get("ROSCONSOLE_CONFIG_FILE"))
