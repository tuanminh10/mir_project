import re
with open('/home/tuanminh/mir_project/src/mir_robot/tm/xlanav/vis.py', 'r') as f:
    content = f.read()

# Thêm biến cờ fallback
content = content.replace('self.camera_ready = False', 'self.camera_ready = False\n        self.use_webcam_fallback = False\n        self.webcam = None')

# Khởi tạo cv2 thay thế nếu lỗi
fallback_code = """
        except RuntimeError as e:
            rospy.logerr(f"❌ KHÔNG TÌM THẤY CAMERA REALSENSE. Chuyển sang dùng WEBCAM Laptop!")
            self.depth_intrinsics = None
            self.use_webcam_fallback = True
            self.webcam = cv2.VideoCapture(0) # 0, 1 hoặc 2 (tuỳ webcam) có thể đổi nếu lap có ảo
            if self.webcam.isOpened():
                self.camera_ready = True
            else:
                rospy.logerr("❌ KHÔNG MỞ ĐƯỢC WEBCAM LUÔN!")
"""
content = re.sub(r'except RuntimeError as e:\n.*rospy.logerr\(f"❌ KHÔNG TÌM THẤY CAMERA REALSENSE: \{e\}"\)\n.*self\.depth_intrinsics = None', fallback_code, content)

# Trong hàm run(), thêm đoạn xử lý cho fallback
run_match = re.search(r'    def run\(self\):.*?while not rospy\.is_shutdown\(\):', content, re.DOTALL)
if run_match:
    pass

with open('/home/tuanminh/mir_project/src/mir_robot/tm/xlanav/vis.py', 'w') as f:
    f.write(content)
