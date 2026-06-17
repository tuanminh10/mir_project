#include <Arduino.h>
#include <Servo.h>

Servo myServo;
const int servoPin = 9; // Chân tín hiệu (dây vàng/cam) nối với Arduino
int currentAngle = 90;  // Lưu trạng thái góc quay hiện tại

void setup() {
  Serial.begin(115200);
  myServo.attach(servoPin); // Gắn servo vào chân số 9
  myServo.write(90);        // Đưa góc quay về chính giữa (90 độ)
  Serial.println("Arduino Servo Ready. Doi lenh tu Python...");
}

void loop() {
  if (Serial.available() > 0) {
    // Đọc lệnh từ máy tính gửi xuống (kết thúc bằng dấu xuống dòng \n)
    String input = Serial.readStringUntil('\n'); 
    input.trim(); // Xóa khoảng trắng thừa
    
    if (input.length() > 0) {
      int angle = input.toInt();
      // Giới hạn an toàn từ 0 đến 180 độ
      if (angle >= 0 && angle <= 180) {
        // QUAY TỪ TỪ ĐỂ HÃM GIA TỐC (Smooth sweep)
        if (angle > currentAngle) {
          for (int pos = currentAngle; pos <= angle; pos += 1) {
            myServo.write(pos);
            delay(15); // Đợi 15ms mỗi 1 độ (Khoảng 0.15s / 10 độ)
          }
        } else {
          for (int pos = currentAngle; pos >= angle; pos -= 1) {
            myServo.write(pos);
            delay(15); // Đợi 15ms mỗi 1 độ
          }
        }
        currentAngle = angle;
        
        Serial.print("Da quay den goc: ");
        Serial.println(angle);
      } else {
        Serial.println("Loi: Goc quay phai nam trong khoang 0 - 180.");
      }
    }
  }
}
