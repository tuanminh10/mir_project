import os
from ultralytics import YOLO

device = 0 if os.path.exists('/opt/ai_venv/bin/python') else 'cpu'
print(f"Device: {device}")

try:
    print("Loading pose...")
    model_pose = YOLO('yolo11s-pose.pt')
    if device == 0: model_pose.to('cuda')
    
    print("Loading seg...")
    model_seg = YOLO('yolo11s-seg.pt')
    if device == 0: model_seg.to('cuda')
    
    print("Loading best.pt...")
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/detect_laviecoca.v1-laviecoca.yolov11/runs/drink_v11s/weights/best.pt')
    laptop_yolo = YOLO(model_path)
    if device == 0: laptop_yolo.to('cuda')
    
    print("ALL MODELS LOADED SUCCESSFULLY!")
except Exception as e:
    print(f"ERROR: {e}")
