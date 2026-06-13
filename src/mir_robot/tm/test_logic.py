import math
import torch
import numpy as np

# Dummy data
hx, hy = 100, 100
detected_hands = [(hx, hy, 5, True)]

class DummyBox:
    def __init__(self):
        self.id = torch.tensor([1])
        self.xyxy = torch.tensor([[50.0, 50.0, 150.0, 150.0]])

class DummyResult:
    def __init__(self):
        self.boxes = [DummyBox()]

results_pose = [DummyResult()]
boxes = results_pose[0].boxes

keypoints = torch.rand(1, 17, 3)

hand_assignments = {}
try:
    for h_idx, (hx, hy, fingers, open5) in enumerate(detected_hands):
        best_track_id = -1
        min_score = float('inf')
        
        for j, box_j in enumerate(boxes):
            if box_j.id is None: continue
            t_id_j = int(box_j.id[0].item())
            x1_j, y1_j, x2_j, y2_j = box_j.xyxy[0].cpu().numpy()
            
            if not (x1_j - 50 < hx < x2_j + 50 and y1_j - 50 < hy < y2_j + 50):
                continue
                
            score = float('inf')
            if keypoints is not None and j < len(keypoints):
                kp_j = keypoints[j]
                if len(kp_j) >= 11:
                    lw = kp_j[9]
                    rw = kp_j[10]
                    d_lw = math.hypot(hx - lw[0].item(), hy - lw[1].item()) if len(lw)>=3 and lw[2].item() > 0.3 else float('inf')
                    d_rw = math.hypot(hx - rw[0].item(), hy - rw[1].item()) if len(rw)>=3 and rw[2].item() > 0.3 else float('inf')
                    
                    ls = kp_j[5]
                    rs = kp_j[6]
                    d_ls = math.hypot(hx - ls[0].item(), hy - ls[1].item()) if len(ls)>=3 and ls[2].item() > 0.3 else float('inf')
                    d_rs = math.hypot(hx - rs[0].item(), hy - rs[1].item()) if len(rs)>=3 and rs[2].item() > 0.3 else float('inf')
                    
                    score = min(d_lw, d_rw, d_ls, d_rs)
            
            if score == float('inf'):
                score = math.hypot(float(hx - (x1_j+x2_j)/2), float(hy - (y1_j+y2_j)/2))
                
            if score < min_score:
                min_score = score
                best_track_id = t_id_j
                
        if best_track_id != -1:
            hand_assignments[h_idx] = best_track_id
    print("SUCCESS!")
    print(hand_assignments)
except Exception as e:
    import traceback
    traceback.print_exc()
