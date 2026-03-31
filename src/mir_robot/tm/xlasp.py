import cv2
from ultralytics import YOLO

# 1. Đường dẫn đến file model của bạn 
# Thay 'yolo11m.pt' bằng 'best.pt' sau khi bạn đã train model của riêng mình
model_path = '/home/tuanminh/mir_project/src/mir_robot/tm/best/best.pt'
model = YOLO(model_path)

# 2. Khởi tạo camera laptop (ID = 0 là camera mặc định)
cap = cv2.VideoCapture(0)

# Cài đặt độ phân giải (Tùy chọn)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    success, img = cap.read()
    if not success:
        print("Không thể kết nối đến Camera!")
        break

    # 3. Đưa ảnh vào model để nhận diện
    # Dùng model.track để theo dõi vật thể mượt mà (chống giật lag)
    # conf=0.40: Tăng confidence để lọc nhiễu / nhận diện nhầm tốt hơn
    results = model.track(img, persist=True, stream=True, conf=0.40, iou=0.45, imgsz=640)

    # Khởi tạo biến đếm
    count_lavie = 0
    count_coca = 0
    total_products = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Lấy tọa độ
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Lọc nhiễu: Bỏ qua các khung hình quá nhỏ (diện tích < 1000 pixel)
            if (x2 - x1) * (y2 - y1) < 1000:
                continue

            # Độ tự tin (Confidence)
            conf = int(box.conf[0] * 100)
            
            # Id của class 
            cls = int(box.cls[0])
            class_name = model.names[cls]

            # (!!!) CHÚ Ý: Logic đếm bên dưới giả định model custom của bạn có class là 'lavie' và 'coca'.
            # Nếu chạy model yolo mặc định, class_name cho chai sẽ là 'bottle'.
            if class_name in ['lavie', 'coca', 'bottle']:
                total_products += 1
                if class_name == 'lavie': count_lavie += 1
                if class_name == 'coca': count_coca += 1

                # Vẽ khung chữ nhật (Màu đổi theo class)
                color = (255, 0, 0) if class_name in ['lavie', 'bottle'] else (0, 0, 255) # Xanh cho lavie/bottle, Đỏ cho coca
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

                # Hiển thị text nhãn hiệu + % tự tin lên màn hình hình
                text = f'{class_name.upper()} {conf}%'
                cv2.putText(img, text, (max(0, x1), max(35, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # 4. Hiển thị bảng tổng kết lên góc trên bên trái màn hình
    # Tạo nền mờ (Alpha Blending) để bảng tổng kết không che khuất hoàn toàn tầm nhìn camera
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (450, 150), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    cv2.putText(img, f'Ghi nhan Tong cong: {total_products} san pham', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, f'Lavie: {count_lavie}', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(img, f'Coca: {count_coca}', (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Hiển thị cửa sổ
    cv2.imshow("Nhan dien San Pham (Lavie & Coca) - Nhan 'Q' de thoat", img)

    # Bấm phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Dọn dẹp bộ nhớ
cap.release()
cv2.destroyAllWindows()
