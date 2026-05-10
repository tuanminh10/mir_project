#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Điều khiển MiR robot thật bằng tay cầm (Xbox/PS4/PS5).
- Giữ R1/RB (nút 5) = Deadman Switch -> Robot mới chạy
- Cần gạt trái: Lên/Xuống = Tiến/Lùi, Trái/Phải = Xoay
- Nhả R1 => Robot dừng ngay lập tức

Tương thích: ROS1 Noetic, MiR 100/200/250
"""
import rospy
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist


class MirJoyTeleop:
    def __init__(self):
        rospy.init_node('mir_joy_teleop')

        # Đọc tham số từ launch file (có thể điều chỉnh tốc độ mà không sửa code)
        self.max_linear_speed = rospy.get_param('~max_linear_speed', 0.3)   # m/s
        self.max_angular_speed = rospy.get_param('~max_angular_speed', 0.3)  # rad/s

        # Mapping nút tay cầm (chỉnh nếu dùng tay cầm khác)
        self.axis_linear = rospy.get_param('~axis_linear', 1)     # Cần gạt trái Lên/Xuống
        self.axis_angular = rospy.get_param('~axis_angular', 0)   # Cần gạt trái Trái/Phải
        self.deadman_button = rospy.get_param('~deadman_button', 5)  # Nút R1/RB

        # Nút tăng/giảm tốc (tuỳ chọn)
        self.speed_up_button = rospy.get_param('~speed_up_button', 3)    # Nút Y/Triangle
        self.speed_down_button = rospy.get_param('~speed_down_button', 0)  # Nút A/Cross

        self.current_linear_scale = self.max_linear_speed
        self.current_angular_scale = self.max_angular_speed

        # Publisher: gửi cmd_vel đến mir_bridge -> robot thật
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Subscriber: nhận tín hiệu từ tay cầm
        rospy.Subscriber('/joy', Joy, self.joy_callback)

        # Gửi lệnh dừng khi node tắt
        rospy.on_shutdown(self.shutdown_hook)

        rospy.loginfo("=" * 50)
        rospy.loginfo("  MiR Joystick Teleop - Sẵn sàng!")
        rospy.loginfo("  Tốc độ tuyến tính: %.2f m/s", self.max_linear_speed)
        rospy.loginfo("  Tốc độ xoay:       %.2f rad/s", self.max_angular_speed)
        rospy.loginfo("  GIỮ NÚT R1/RB TRƯỚC KHI ĐẨY CẦN GẠT!")
        rospy.loginfo("=" * 50)

    def joy_callback(self, joy_msg):
        twist = Twist()

        # Xử lý nút tăng/giảm tốc
        if len(joy_msg.buttons) > max(self.speed_up_button, self.speed_down_button):
            if joy_msg.buttons[self.speed_up_button] == 1:
                self.current_linear_scale = min(self.current_linear_scale + 0.05, 1.0)
                self.current_angular_scale = min(self.current_angular_scale + 0.05, 1.0)
                rospy.loginfo("Tốc độ TĂNG: linear=%.2f, angular=%.2f",
                              self.current_linear_scale, self.current_angular_scale)
            elif joy_msg.buttons[self.speed_down_button] == 1:
                self.current_linear_scale = max(self.current_linear_scale - 0.05, 0.05)
                self.current_angular_scale = max(self.current_angular_scale - 0.05, 0.05)
                rospy.loginfo("Tốc độ GIẢM: linear=%.2f, angular=%.2f",
                              self.current_linear_scale, self.current_angular_scale)

        # Kiểm tra Deadman Switch (R1/RB)
        if len(joy_msg.buttons) > self.deadman_button and joy_msg.buttons[self.deadman_button] == 1:
            # R1 đang giữ -> Cho phép điều khiển
            twist.linear.x = joy_msg.axes[self.axis_linear] * self.current_linear_scale
            twist.angular.z = joy_msg.axes[self.axis_angular] * self.current_angular_scale
            self.cmd_pub.publish(twist)
            self.is_moving = True
        else:
            # Chỉ gửi lệnh phanh nếu trước đó đang chạy
            if getattr(self, 'is_moving', False):
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_pub.publish(twist)
                self.cmd_pub.publish(twist) # Gửi thêm 1 lần cho chắc
                self.is_moving = False

    def shutdown_hook(self):
        """Gửi lệnh dừng khi tắt node."""
        rospy.loginfo("Đang dừng robot...")
        stop_msg = Twist()
        self.cmd_pub.publish(stop_msg)
        rospy.sleep(0.5)


if __name__ == '__main__':
    try:
        MirJoyTeleop()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
