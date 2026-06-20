import os
from ultralytics import YOLO

os.environ['YOLO_OFFLINE'] = 'True'
try:
    y = YOLO('/home/tuanminh/mir_project/src/mir_robot/tm/best/best.pt')
    print("LOADED SUCCESSFULLY")
except Exception as e:
    print(f"FAILED TO LOAD: {e}")
