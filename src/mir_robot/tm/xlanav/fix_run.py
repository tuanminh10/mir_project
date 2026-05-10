import re
with open('/home/tuanminh/mir_project/src/mir_robot/tm/xlanav/vis.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'frames = self.pipeline.wait_for_frames()' in line:
        new_lines.append(line.replace('frames = self.pipeline.wait_for_frames()', '''if hasattr(self, "use_webcam_fallback") and self.use_webcam_fallback:
                    ret, frame = self.webcam.read()
                    if not ret: continue
                    depth_frame = None
                else:
                    frames = self.pipeline.wait_for_frames()'''))
    elif 'frame = np.asanyarray(color_frame.get_data())' in line:
        new_lines.append(line.replace('frame = np.asanyarray(color_frame.get_data())', '''if hasattr(self, "use_webcam_fallback") and self.use_webcam_fallback:
                  pass # "frame" từ webcam read bên trên
              else:
                  frame = np.asanyarray(color_frame.get_data())'''))
    # Disable depth logic
    elif 'd, left_m = get_depth_distance_m' in line:
        new_lines.append('''                        if self.depth_intrinsics:
                            d, left_m = get_depth_distance_m(depth_frame, box, frame_w, frame_h)
                        else:
                            d, left_m = 1.5, 0.0 # Giả lập cách 1.5m, lệch 0 khi dùng webcam
''')
    else:
        new_lines.append(line)

with open('/home/tuanminh/mir_project/src/mir_robot/tm/xlanav/vis.py', 'w') as f:
    f.writelines(new_lines)
