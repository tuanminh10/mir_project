import re

with open("/home/tuanminh/mir_project/src/mir_robot/tm/xlanav.py", "r") as f:
    text = f.read()

sub_code = """
    # Lắng nghe nhiều topic Global Plan phổ biến để luôn vẽ được đường đi
    rospy.Subscriber('/move_base_node/SBPLLatticePlanner/plan', Path, core_logic.path_callback)
    rospy.Subscriber('/move_base_node/GlobalPlanner/plan', Path, core_logic.path_callback)
    rospy.Subscriber('/move_base/GlobalPlanner/plan', Path, core_logic.path_callback)
    rospy.Subscriber('/move_base_node/mir_global_planner/plan', Path, core_logic.path_callback)
    rospy.Subscriber('/move_base/NavfnROS/plan', Path, core_logic.path_callback)
    rospy.Subscriber('/mir_planner/global_path', Path, core_logic.path_callback)
    
    rospy.loginfo("[Map] Đã đăng ký subscriber /map & PATH")
"""
pattern = re.compile(r"# Lắng nghe Global Plan.*Sẵn sàng", re.DOTALL)
text = re.sub(r"# Lắng nghe Global Plan.*\n    rospy\.Subscriber\('/move_base_node/SBPLLatticePlanner/plan', Path, core_logic\.path_callback\)\n    \n    rospy\.loginfo\(\"\[Map\] Đã đăng ký subscriber /map và SBPLLatticePlanner/plan\"\)", sub_code.strip(), text)

with open("/home/tuanminh/mir_project/src/mir_robot/tm/xlanav.py", "w") as f:
    f.write(text)

