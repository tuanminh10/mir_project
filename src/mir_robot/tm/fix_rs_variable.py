for filename in ['test4.py', 'fix4.py']:
    with open(filename, 'r') as f:
        content = f.read()
    
    # Đổi rs -> r_sh và ls -> l_sh để không trùng với import pyrealsense2 as rs
    content = content.replace("ls = kp_j[5]", "l_sh = kp_j[5]")
    content = content.replace("rs = kp_j[6]", "r_sh = kp_j[6]")
    
    content = content.replace("d_ls = math.hypot(hx - ls[0].item(), hy - ls[1].item()) if len(ls)>=3 and ls[2].item() > 0.3 else float('inf')",
                              "d_ls = math.hypot(hx - l_sh[0].item(), hy - l_sh[1].item()) if len(l_sh)>=3 and l_sh[2].item() > 0.3 else float('inf')")
    content = content.replace("d_rs = math.hypot(hx - rs[0].item(), hy - rs[1].item()) if len(rs)>=3 and rs[2].item() > 0.3 else float('inf')",
                              "d_rs = math.hypot(hx - r_sh[0].item(), hy - r_sh[1].item()) if len(r_sh)>=3 and r_sh[2].item() > 0.3 else float('inf')")
    
    with open(filename, 'w') as f:
        f.write(content)
