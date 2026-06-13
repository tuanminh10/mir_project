for filename in ['test4.py', 'fix4.py']:
    with open(filename, 'r') as f:
        content = f.read()
    
    # Sửa lỗi math.hypot
    content = content.replace("score = math.hypot(hx - (x1_j+x2_j)/2, hy - (y1_j+y2_j)/2)",
                              "score = math.hypot(float(hx - (x1_j+x2_j)/2), float(hy - (y1_j+y2_j)/2))")
    
    with open(filename, 'w') as f:
        f.write(content)
