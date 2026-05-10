import re

with open("/home/tuanminh/mir_project/src/mir_robot/tm/xlanav.py", "r") as f:
    text = f.read()

# 1. Update MirNavigator to use subprocess
navigator_code = """
class MirNavigator:
    def __init__(self, ip="192.168.0.177"):
        self.ip = ip
        self.api_url = f"http://{self.ip}/api/v2.0.0"
        auth = "Basic YWRtaW46OGM2OTc2ZTViNTQxMDQxNWJkZTkwOGJkNGRlZTE1ZGZiMTY3YTljODczZmM0YmI4YTgxZjZmMmFiNDQ4YTkxOA=="
        self.headers = {"Content-Type": "application/json", "Authorization": auth}
        
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self._helper_script = __file__.replace('xlanav.py', 'send_goal_helper.py')
        self._goal_process = None
        rospy.loginfo("[MirNav] Sẵn sàng (dùng subprocess Python3.8 ActionLib)")

    def ensure_ready(self):
        try:
            requests.put(f"{self.api_url}/status", headers=self.headers, json={"clear_error": True, "state_id": 3}, timeout=1)
        except: pass

    def cancel_all(self):
        self.is_navigating = False
        global robot_planned_path, goal_pose
        robot_planned_path = []
        goal_pose = None
        try:
            requests.put(f"{self.api_url}/status", headers=self.headers, json={"state_id": 4}, timeout=2)
            requests.delete(f"{self.api_url}/mission_queue", headers=self.headers, timeout=2)
            if self._goal_process and self._goal_process.poll() is None:
                self._goal_process.terminate()
            self.cmd_vel_pub.publish(Twist())
        except Exception as e:
            rospy.logerr(f"Lỗi cancel: {e}")

    def send_goal(self, goal_x, goal_y, goal_yaw=0.0):
        self.ensure_ready()
        try:
            import os, subprocess, threading
            cmd = ["/usr/bin/python3", self._helper_script, str(goal_x), str(goal_y), str(goal_yaw)]
            rospy.loginfo(f"[MirNav] Gửi goal ({goal_x:.2f}, {goal_y:.2f}) qua subprocess Python3.8...")
            
            clean_env = os.environ.copy()
            clean_env.pop("VIRTUAL_ENV", None)
            if "PYTHONPATH" in clean_env:
                clean_env["PYTHONPATH"] = ":".join(p for p in clean_env["PYTHONPATH"].split(":") if "ai_venv" not in p and p)
            clean_env.pop("PYTHONHOME", None)
            if "PATH" in clean_env:
                clean_env["PATH"] = ":".join(p for p in clean_env["PATH"].split(":") if "ai_venv" not in p)
            clean_env["ROS_MASTER_URI"] = clean_env.get("ROS_MASTER_URI", "http://localhost:11311")
            clean_env["PYTHONUNBUFFERED"] = "1"

            self._goal_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=clean_env
            )
            def _log_output():
                for line in iter(self._goal_process.stdout.readline, b''):
                    rospy.loginfo(line.decode().strip())
            threading.Thread(target=_log_output, daemon=True).start()
            return True
        except Exception as e:
            rospy.logerr(f"[MirNav] Không thể gửi goal: {e}")
            return False

    def send_goal_cmd_vel(self, goal_x, goal_y):
"""

import re
pattern = re.compile(r"class MirNavigator:.*?def send_goal_cmd_vel\(self, goal_x, goal_y\):", re.DOTALL)
text = pattern.sub(navigator_code.strip() + r"\n        ", text)

# 2. Replace acquire_coords logic
acquire_code = """
                goal_pose = (final_goal_x, final_goal_y) # Để vẽ lên map xanh
                
                signal_bus.status_update.emit(f"Robot đang tiếp cận Mục Tiêu theo đường chéo tự động!")
                # mir_tts.speak_on_mir("Bắt đầu tự động quét tọa độ và lộ trình di chuyển tới đối tượng.")
                
                # 4. Phát lệnh lái qua Planner ActionLib
                if not self.nav.send_goal(final_goal_x, final_goal_y, final_yaw):
                    signal_bus.status_update.emit("Lỗi gửi goal. Dùng Bypass CMD_VEL...")
                    self.nav.send_goal_cmd_vel(final_goal_x, final_goal_y)

            except Exception as e:
                signal_bus.status_update.emit(f"Lỗi tính toán không gian Map (TF)")
"""
pattern_acquire = re.compile(r"goal_pose = \(final_goal_x, final_goal_y\).*?except Exception as e:", re.DOTALL)
text = pattern_acquire.sub(acquire_code.strip() + r"\n            except Exception as e:", text)

with open("/home/tuanminh/mir_project/src/mir_robot/tm/xlanav.py", "w") as f:
    f.write(text)

