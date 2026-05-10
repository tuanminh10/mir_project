import re
with open('/home/tuanminh/mir_project/src/mir_robot/tm/xlanav/vis.py', 'r') as f:
    content = f.read()

# Sửa lại đoạn lặp lấy frame
new_loop = """            try:
                if getattr(self, "use_webcam_fallback", False):
                    ret, frame = self.webcam.read()
                    if not ret: continue
                    depth_frame = None
                else:
                    frames = self.pipeline.wait_for_frames()
                    aligned = self.align.process(frames)
                    depth_frame = aligned.get_depth_frame()
                    color_frame = aligned.get_color_frame()
                    if not depth_frame or not color_frame: continue
                    frame = np.asanyarray(color_frame.get_data())
            except Exception as e:
                rospy.logwarn(f"Lỗi frame camera: {e}")
                continue"""
content = re.sub(r'            try:.*?\n                if getattr.*?color_frame\.get_data\(\)\)[\n ]*', new_loop + '\n', content, flags=re.DOTALL)
with open('/home/tuanminh/mir_project/src/mir_robot/tm/xlanav/vis.py', 'w') as f:
    f.write(content)
