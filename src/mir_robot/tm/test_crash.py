from PyQt5.QtWidgets import QApplication
import sys
import threading
import time
import math
import cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np

def run():
    try:
        mp_hands = mp.solutions.hands
        hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        print("Mediapipe initialized!")
        
        # Test math.hypot with np.float32
        x1_j = np.float32(100.0)
        hx = 50
        print("Trying math.hypot with np.float32...")
        score = math.hypot(hx - x1_j, hx - x1_j)
        print("math.hypot SUCCESS:", score)
    except Exception as e:
        import traceback
        traceback.print_exc()

# Create app first (required by PyQt5)
app = QApplication(sys.argv)
t = threading.Thread(target=run)
t.start()
t.join()
