import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
from ultralytics import YOLO
import os
import sys
import time

# ==============================================================================
# HACK HỖ TRỢ GPU RTX 5060 (Blackwell - sm_120) TRÊN ROS NOETIC
# ==============================================================================
if os.path.exists('/opt/ai_venv/bin/python') and sys.executable != '/opt/ai_venv/bin/python':
    print("🚀 Auto-switched to Python 3.9 venv to unlock NVIDIA RTX 5060 (sm_120) GPU cho YOLO...")
    sys.stdout.flush()
    os.execv('/opt/ai_venv/bin/python', ['/opt/ai_venv/bin/python'] + sys.argv)

# ==========================================
# PHƯƠNG PHÁP 2: YOLO SEGMENTATION + RADIUS FILTER + 3D VISUALIZATION
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
    
    # Model Segmentation
    model = YOLO("yolov8n-seg.pt")

    print("--- HƯỚNG DẪN ---")
    print("Nhấn 'v' để MỞ CỬA SỔ 3D (xem Outlier Đỏ, Inlier Xám). Đóng cửa sổ 3D để tiếp tục camera.")
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
            
            clouds_to_visualize = []

            for r in results:
                if r.masks is None: continue
                
                masks = r.masks.data.cpu().numpy()
                boxes = r.boxes.xyxy.cpu().numpy()

                for i, mask in enumerate(masks):
                    mask_resized = cv2.resize(mask, (color_image.shape[1], color_image.shape[0]))
                    binary_mask = (mask_resized > 0.5).astype(np.uint8)

                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(color_image, contours, -1, (255, 0, 0), 2)

                    masked_depth = depth_image * binary_mask
                    o3d_depth = o3d.geometry.Image(masked_depth)
                    pcd = o3d.geometry.PointCloud.create_from_depth_image(
                        o3d_depth, pinhole_camera_intrinsic, depth_scale=1000.0, depth_trunc=3.0)

                    if len(pcd.points) < 100: continue

                    # GỌT VIỀN (RADIUS OUTLIER) VÀ LẤY INDEX
                    cl_pcd, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
                    
                    # TẠO 2 ĐÁM MÂY ĐIỂM INLIER VÀ OUTLIER ĐỂ HIỂN THỊ (Giống tutorial)
                    inlier_cloud = pcd.select_by_index(ind)
                    outlier_cloud = pcd.select_by_index(ind, invert=True)

                    # TÔ MÀU
                    outlier_cloud.paint_uniform_color([1, 0, 0])      # Đỏ cho nhiễu
                    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8]) # Xám cho điểm chuẩn

                    # Lưu lại để popup khi bấm 'v'
                    clouds_to_visualize = [inlier_cloud, outlier_cloud]

                    points = np.asarray(inlier_cloud.points)
                    if len(points) == 0: continue
                    median_z = np.median(points[:, 2])

                    x1, y1, x2, y2 = map(int, boxes[i])
                    cv2.putText(color_image, f"Dist: {median_z:.2f}m", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(color_image, "Press 'v' for 3D View", (x1, y2 + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow('PP2: YOLO Seg + Radius Filter', color_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('v') and len(clouds_to_visualize) > 0:
                # HIỂN THỊ OPEN3D GIỐNG TRONG DOCS
                print("Đang mở cửa sổ 3D... Đóng cửa sổ 3D để tiếp tục camera.")
                o3d.visualization.draw_geometries(clouds_to_visualize, window_name="Radius Filter (Red = Outliers, Gray = Inliers)")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
