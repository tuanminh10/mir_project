for filename in ['test4.py', 'fix4.py']:
    with open(filename, 'r') as f:
        content = f.read()
    
    # Đảo ngược toàn bộ logic hướng nhìn của camera
    content = content.replace("forward_m = -cam_z - 0.475", "forward_m = cam_z + 0.475")
    content = content.replace("left_m = -cam_x", "left_m = cam_x")
    
    with open(filename, 'w') as f:
        f.write(content)
