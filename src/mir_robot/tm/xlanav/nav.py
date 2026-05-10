import os
import time
import math
import threading
import subprocess
import requests

import rospy
from geometry_msgs.msg import Twist

import config

class MirNavigator:
    def __init__(self):
        rospy.loginfo("[MirNav] KHOI DONG DAEMON SUBPROCESS...")
        # Tìm file helper nằm ở thư mục cha (tm/)
        self._helper_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'send_goal_helper.py')
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        self.api_url = config.MIR_API_URL
        self.headers = {"Content-Type": "application/json", "Authorization": config.MIR_AUTH}

        self.is_navigating = False
        
        # CHẠY BACKGROUND DAEMON
        clean_env = self._clean_environment()
        self._daemon_process = subprocess.Popen(
            ['/usr/bin/python3', self._helper_script], 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            env=clean_env,
            bufsize=1
        )

        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _clean_environment(self):
        """Loại bỏ môi trường venv khỏi PATH/PYTHONPATH để ROS 3.8 chạy đúng."""
        clean_env = os.environ.copy()
        clean_env.pop('PYTHONHOME', None)
        clean_env.pop('VIRTUAL_ENV', None)
        if 'PATH' in clean_env:
            paths = clean_env['PATH'].split(':')
            clean_env['PATH'] = ':'.join([p for p in paths if 'ai_venv' not in p and '.venv' not in p])
        if 'PYTHONPATH' in clean_env:
            ppaths = clean_env['PYTHONPATH'].split(':')
            clean_env['PYTHONPATH'] = ':'.join([p for p in ppaths if 'ai_venv' not in p and '.venv' not in p])
        return clean_env

    def _read_stdout(self):
        while self._daemon_process.poll() is None:
            line = self._daemon_process.stdout.readline()
            if line:
                rospy.loginfo(f"[Daemon] {line.strip()}")
                if line.startswith("STATE_3"):
                    rospy.loginfo(f"[MirNav move_base] 🎯 Đã cập bến điểm tĩnh (Helper báo thành công)!")
                    self.is_navigating = False
                elif line.startswith("STATE_8"):
                    rospy.loginfo(f"[MirNav move_base] ❌ Lệnh bị hủy ngang/Preempt code 8.")
                    self.is_navigating = False
                elif line.startswith("ERROR_"):
                    rospy.logerr(f"[MirNav move_base] LỖI DAEMON: {line.strip()}")
    
    def _read_stderr(self):
        while self._daemon_process.poll() is None:
            err = self._daemon_process.stderr.readline()
            if err:
                rospy.logwarn(f"[Daemon Err] {err.strip()}")

    def __del__(self):
        if hasattr(self, '_daemon_process') and self._daemon_process:
            self._daemon_process.terminate()

    def ensure_ready(self):
        """Đảm bảo MiR ở trạng thái Ready (State 3)"""
        try:
            requests.delete(f"{self.api_url}/status", headers=self.headers, timeout=2)
            time.sleep(0.2)
            requests.put(f"{self.api_url}/status", headers=self.headers, json={"state_id": 3}, timeout=2)
            rospy.loginfo("[MirNav] Đã ép MiR Web Dashboard sang trạng thái Ready (Màu xanh)")
        except Exception as e:
            rospy.logerr(f"[MirNav] Lỗi ensure_ready: {e}")

    def cancel_all(self):
        """Hủy mọi lệnh và dừng xe"""
        self.is_navigating = False
        if hasattr(self, '_daemon_process') and self._daemon_process.poll() is None:
            try:
                self._daemon_process.stdin.write("CANCEL\n")
                self._daemon_process.stdin.flush()
            except Exception as e:
                rospy.logerr(f"Lỗi gửi CANCEL tới daemon: {e}")

        try:
            self.cmd_vel_pub.publish(Twist())
        except Exception as e:
            rospy.logerr(f"Lỗi cancel cmd_vel: {e}")

    def send_goal(self, goal_x, goal_y, goal_yaw=0.0):
        """Gửi goal qua file helper Python 3.8"""
        self.is_navigating = True
        if hasattr(self, '_daemon_process') and self._daemon_process.poll() is None:
            try:
                self._daemon_process.stdin.write(f"{goal_x},{goal_y},{goal_yaw}\n")
                self._daemon_process.stdin.flush()
                return True
            except Exception as e:
                rospy.logerr(f"Lỗi gửi lệnh tới daemon: {e}")
                return False
        return False
