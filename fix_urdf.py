import re
with open('/home/tuanminh/mir_project/src/mir_robot/mir_description/urdf/include/realsense.urdf.xacro', 'r') as f:
    text = f.read()

# Fix thẻ XML bị sai lúc nãy
text = text.replace('<always_on>true</always_on>', '<always_on>true</always_on>')
text = text.replace('<update_rate>15.0</update_rate>', '<update_rate>30.0</update_rate>')

with open('/home/tuanminh/mir_project/src/mir_robot/mir_description/urdf/include/realsense.urdf.xacro', 'w') as f:
    f.write(text)
