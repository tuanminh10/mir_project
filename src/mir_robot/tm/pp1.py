import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
import sys
import time

# ==============================================================================
# HACK HỖ TRỢ GPU RTX 5060 (Blackwell - sm_120) TRÊN ROS NOETIC (PYTHON 3.8)
# TRÁNH LỖI "no kernel image is available for execution on the device"
# ==============================================================================
if os.path.exists('/opt/ai_venv/bin/python') and sys.executable != '/opt/ai_venv/bin/python':
    print("🚀 Auto-switched to Python 3.9 venv to unlock NVIDIA RTX 5060 (sm_120) GPU cho YOLO...")
    sys.stdout.flush()
    os.execv('/opt/ai_venv/bin/python', ['/opt/ai_venv/bin/python'] + sys.argv)

# ==========================================
# PHƯƠNG PHÁP 1: YOLO BOX + DBSCAN + 3D VISUALIZATION
# ==========================================

def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    profile = pipeline.get_active_profile()
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()
    
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        depth_intrinsics.width, depth_intrinsics.height, 
        depth_intrinsics.fx, depth_intrinsics.fy, 
        depth_intrinsics.ppx, depth_intrinsics.ppy
    )

    align_to = rs.stream.color
    align = rs.align(align_to)
    model = YOLO("yolov8n.pt") 

    print("--- HƯỚNG DẪN ---")
    print("Nhấn 'v' để MỞ CỬA SỔ 3D (xem các cụm DBSCAN). Đóng cửa sổ 3D để tiếp tục camera.")
    print("Nhấn 'q' để thoát.")

    prev_time = time.time()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame: continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            cv2.putText(color_image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            results = model(color_image, classes=[0], verbose=False)
            
            pcd_to_visualize = None

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    cropped_depth = depth_image[y1:y2, x1:x2].copy()
                    o3d_depth = o3d.geometry.Image(cropped_depth)
                    pcd = o3d.geometry.PointCloud.create_from_depth_image(
                        o3d_depth, pinhole_camera_intrinsic, depth_scale=1000.0, depth_trunc=3.0)

                    if len(pcd.points) < 100: continue

                    # 1. Lọc Statistical
                    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

                    # 2. Phân cụm DBSCAN
                    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
                        labels = np.array(pcd.cluster_dbscan(eps=0.1, min_points=50, print_progress=False))

                    max_label = labels.max()
                    if max_label > -1:
                        # TÔ MÀU HIỂN THỊ GIỐNG TUTORIAL
                        # Các điểm nhiễu (label -1) tô màu đỏ
                        # Các cụm khác tô màu bằng colormap của matplotlib
                        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
                        colors[labels < 0] = [1, 0, 0, 1] # Màu đỏ cho noise
                        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

                        pcd_to_visualize = pcd # Lưu lại để hiện 3D khi bấm 'v'

                        counts = np.bincount(labels[labels >= 0])
                        target_label = np.argmax(counts)
                        target_indices = np.where(labels == target_label)[0]
                        target_pcd = pcd.select_by_index(target_indices)

                        points = np.asarray(target_pcd.points)
                        median_z = np.median(points[:, 2])

                        cv2.putText(color_image, f"Dist: {median_z:.2f}m", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cv2.putText(color_image, "Press 'v' for 3D View", (x1, y2 + 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            cv2.imshow('PP1: YOLO Box + DBSCAN', color_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('v') and pcd_to_visualize is not None:
                # HIỂN THỊ OPEN3D GIỐNG TRONG DOCS
                print("Đang mở cửa sổ 3D... Đóng cửa sổ 3D để tiếp tục camera.")
                o3d.visualization.draw_geometries([pcd_to_visualize], window_name="DBSCAN Clusters (Red = Noise)")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
