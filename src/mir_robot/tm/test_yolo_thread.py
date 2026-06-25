import time
import threading
import numpy as np
from ultralytics import YOLO
import os

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/detect_laviecoca.v1-laviecoca.yolov11/runs/drink_v11s/weights/best.pt')
try:
    print(f"Loading model... {model_path}")
    model = YOLO(model_path)
    print("Model loaded.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

def test_inference():
    print("Background thread started.")
    try:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        print("Running predict...")
        res = model.predict(frame, conf=0.40, verbose=False)
        print("Predict success!")
        print(f"Result length: {len(res)}")
    except Exception as e:
        print(f"Exception during predict: {e}")

t = threading.Thread(target=test_inference)
t.start()
t.join()
print("Main thread finished.")
