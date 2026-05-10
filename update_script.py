import re

with open('/home/tuanminh/mir_project/src/mir_robot/tm/mainv2.py', 'r') as f:
    text = f.read()

pattern1 = re.compile(r'    def verify_items_with_ai\(self, expected_coca, expected_lavie\):.*?return False\n', re.DOTALL)
new_func1 = '''    def verify_items_with_ai(self, expected_coca, expected_lavie):
        rospy.loginfo("⚠️ Đã bỏ qua bước nhận diện AI 30s tại bếp (do yếu tố camera nhiễu).")
        return True
'''
text = pattern1.sub(new_func1, text, count=1)

pattern2 = re.compile(r'    def wait_until_items_taken\(self\):.*?return False\n', re.DOTALL)
new_func2 = '''    def wait_until_items_taken(self):
        rospy.loginfo("⚠️ Đã bỏ qua bước AI nhận diện đồ 5s tại bàn khách. Đợi 5 giây để khách lấy đồ.")
        rospy.sleep(5.0)
        return True
'''
text = pattern2.sub(new_func2, text, count=1)

with open('/home/tuanminh/mir_project/src/mir_robot/tm/mainv2.py', 'w') as f:
    f.write(text)
print("Update applied")
