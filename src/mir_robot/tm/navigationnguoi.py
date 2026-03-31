#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Navigation node: di chuyen robot den vi tri nguoi da lock.
Su dung MiR REST API truc tiep (giong navigationcacdiem.py) thay vi ROS move_base.
"""

import hashlib
import base64
import json
import math
import os
import socket
import threading
import time

import rospy
import tf

from geometry_msgs.msg import Twist, PointStamped

try:
	import requests
except ImportError:
	import subprocess, sys
	subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
	import requests

try:
	from mir_driver.rosbridge import RosbridgeSetup
except ImportError:
	RosbridgeSetup = None


class PersonLockNavigator:
	def __init__(self):
		self.input_topic = rospy.get_param("~input_topic", "/person_locked_relative")
		self.stop_distance = float(rospy.get_param("~stop_distance", 1.5))
		self.min_person_distance = float(rospy.get_param("~min_person_distance", 0.5))
		self.max_person_distance = float(rospy.get_param("~max_person_distance", 6.0))
		self.collect_seconds = int(rospy.get_param("~collect_seconds", 5))
		self.countdown_seconds = int(rospy.get_param("~countdown_seconds", 5))
		self.arrive_dist = float(rospy.get_param("~arrive_dist", 0.5))
		self.navigate_timeout = float(rospy.get_param("~navigate_timeout", 120.0))

		# Toc do cmd_vel (fallback khi REST API + WS that bai)
		self.max_linear_speed = float(rospy.get_param("~max_linear_speed", 0.25))
		self.max_angular_speed = float(rospy.get_param("~max_angular_speed", 0.4))
		self.goal_tolerance = float(rospy.get_param("~goal_tolerance", 0.3))

		# MiR connection
		self.mir_ip = rospy.get_param("~mir_ip", os.getenv("MIR_IP", "192.168.0.177"))
		self.mir_ws_port = int(rospy.get_param("~mir_ws_port", 9090))
		self.api_url = f"http://{self.mir_ip}/api/v2.0.0"

		self.tf_listener = tf.TransformListener()
		self.lock = threading.Lock()

		# Two-phase state
		self._phase = "idle"  # "idle" | "collecting" | "countdown" | "moving"
		self._collected_positions = []
		self.busy = False

		# Publisher cmd_vel (fallback cuoi cung)
		self._cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

		# REST API login
		self._api_headers = None
		self._move_mission_guid = None
		self._move_param_name = None
		self._ws_robot = None

		self._setup_mir_connection()

		rospy.Subscriber(self.input_topic, PointStamped, self._person_locked_cb, queue_size=1)
		rospy.loginfo("[NavNguoi] Dang nghe topic: %s", self.input_topic)

		# UDP listener
		self.udp_port = int(rospy.get_param("~udp_port", 9877))
		udp_thread = threading.Thread(target=self._udp_listener_worker, daemon=True)
		udp_thread.start()
		rospy.loginfo("[NavNguoi] UDP listener started on 0.0.0.0:%d", self.udp_port)

	# ═══════════════════════════════════════════
	#  MiR REST API / WebSocket setup
	# ═══════════════════════════════════════════
	def _setup_mir_connection(self):
		"""Ket noi den MiR qua REST API va WebSocket."""
		rospy.loginfo("[NavNguoi] Ket noi MiR tai %s ...", self.mir_ip)

		# REST API login
		credentials = [
			("distributor", "distributor"),
			("admin", "admin"),
			("Admin", "admin"),
			("service", "service"),
		]
		for user, pw in credentials:
			pw_hash = hashlib.sha256(pw.encode()).hexdigest()
			auth = base64.b64encode(f"{user}:{pw_hash}".encode()).decode()
			headers = {
				"Content-Type": "application/json",
				"Accept": "application/json",
				"Authorization": f"Basic {auth}",
			}
			try:
				r = requests.get(f"{self.api_url}/missions", headers=headers, timeout=5)
				if r.status_code == 200:
					self._api_headers = headers
					rospy.loginfo("[NavNguoi] REST API OK (user=%s)", user)
					break
			except Exception:
				continue

		if self._api_headers is None:
			# Fallback: plain password
			for user, pw in credentials:
				auth = base64.b64encode(f"{user}:{pw}".encode()).decode()
				headers = {
					"Content-Type": "application/json",
					"Accept": "application/json",
					"Authorization": f"Basic {auth}",
				}
				try:
					r = requests.get(f"{self.api_url}/missions", headers=headers, timeout=5)
					if r.status_code == 200:
						self._api_headers = headers
						rospy.loginfo("[NavNguoi] REST API OK (user=%s, plain)", user)
						break
				except Exception:
					continue

		if self._api_headers:
			self._ensure_ready()
			self._find_move_mission()
		else:
			rospy.logwarn("[NavNguoi] REST API login that bai!")

		# WebSocket connection (backup)
		if RosbridgeSetup is not None:
			try:
				self._ws_robot = RosbridgeSetup(self.mir_ip, self.mir_ws_port)
				for i in range(50):  # cho toi da 5s
					if self._ws_robot.is_connected():
						rospy.loginfo("[NavNguoi] WebSocket OK")
						break
					if self._ws_robot.is_errored():
						rospy.logwarn("[NavNguoi] WebSocket loi!")
						self._ws_robot = None
						break
					time.sleep(0.1)
				else:
					rospy.logwarn("[NavNguoi] WebSocket timeout!")
					self._ws_robot = None
			except Exception as e:
				rospy.logwarn("[NavNguoi] WebSocket error: %s", e)
				self._ws_robot = None

	def _api_status(self):
		if not self._api_headers:
			return None
		try:
			return requests.get(f"{self.api_url}/status", headers=self._api_headers, timeout=5).json()
		except Exception:
			return None

	def _ensure_ready(self):
		"""Dam bao robot o trang thai READY. Xoa error + mission queue neu can."""
		if not self._api_headers:
			return False

		status = self._api_status()
		if not status:
			return False
		state = status.get("state_id", -1)
		rospy.loginfo("[NavNguoi] Robot state: %s (%s)", state, status.get("state_text", ""))

		if state == 3:  # Ready 
			return True

		# Xoa mission queue (nguyen nhan error lap lai)
		try:
			requests.delete(f"{self.api_url}/mission_queue", headers=self._api_headers, timeout=5)
		except Exception:
			pass

		# Xoa error status
		if state in (10, 12):
			try:
				requests.delete(f"{self.api_url}/status", headers=self._api_headers, timeout=5)
			except Exception:
				pass
			time.sleep(1)

		# Pause truoc roi Ready (giong navigationcacdiem.py)
		try:
			requests.put(f"{self.api_url}/status", headers=self._api_headers,
						 json={"state_id": 4}, timeout=5)
			time.sleep(0.5)
		except Exception:
			pass
		try:
			requests.put(f"{self.api_url}/status", headers=self._api_headers,
						 json={"state_id": 3}, timeout=5)
			time.sleep(1)
		except Exception:
			return False

		# Kiem tra lai
		for _ in range(3):
			st = self._api_status()
			if st and st.get("state_id") == 3:
				rospy.loginfo("[NavNguoi] Robot READY!")
				return True
			time.sleep(1)

		rospy.logwarn("[NavNguoi] Khong chuyen duoc ve READY!")
		return False

	def _find_move_mission(self):
		"""Tim mission Move tren MiR."""
		if not self._api_headers:
			return
		try:
			r = requests.get(f"{self.api_url}/missions", headers=self._api_headers, timeout=5)
			missions = r.json()
		except Exception:
			return

		move_guid = None
		for m in missions:
			name = m.get("name", "").lower()
			if name in ("move", "go to", "goto", "move to position", "di chuyen"):
				move_guid = m.get("guid")
				break
		if not move_guid:
			for m in missions:
				if "move" in m.get("name", "").lower():
					move_guid = m.get("guid")
					break
		if not move_guid:
			rospy.logwarn("[NavNguoi] Khong tim thay mission 'Move' tren MiR!")
			return

		# Lay parameter name
		param_name = "Position"
		try:
			r = requests.get(f"{self.api_url}/missions/{move_guid}/actions",
							 headers=self._api_headers, timeout=5)
			if r.status_code == 200:
				for act in r.json():
					for p in act.get("parameters", []):
						inp = p.get("input_name", "")
						if inp and inp != "None":
							param_name = inp
							break
					if param_name != "Position":
						break
		except Exception:
			pass

		self._move_mission_guid = move_guid
		self._move_param_name = param_name
		rospy.loginfo("[NavNguoi] Mission 'Move' found (param='%s')", param_name)

	# ═══════════════════════════════════════════
	#  TF + UDP
	# ═══════════════════════════════════════════
	def _get_robot_pose_map(self):
		try:
			trans, rot = self.tf_listener.lookupTransform("/map", "/base_link", rospy.Time(0))
			yaw = tf.transformations.euler_from_quaternion(rot)[2]
			return trans[0], trans[1], yaw
		except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
			return None

	def _udp_listener_worker(self):
		sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		sock.bind(("0.0.0.0", self.udp_port))
		sock.settimeout(1.0)
		rospy.loginfo("[NavNguoi] UDP socket dang lang nghe tren port %d", self.udp_port)
		while not rospy.is_shutdown():
			try:
				data, addr = sock.recvfrom(1024)
				payload = json.loads(data.decode("utf-8"))
				msg = PointStamped()
				msg.header.stamp = rospy.Time(0)
				msg.header.frame_id = "base_link"
				msg.point.x = float(payload.get("x", 0.0))
				msg.point.y = float(payload.get("y", 0.0))
				msg.point.z = 0.0
				self._person_locked_cb(msg)
			except socket.timeout:
				continue
			except Exception as e:
				rospy.logwarn_throttle(5.0, "[NavNguoi] UDP recv error: %s", e)
		sock.close()

	# ═══════════════════════════════════════════
	#  Navigation via REST API (chinh) / WebSocket (backup)
	# ═══════════════════════════════════════════
	def _navigate_rest_api(self, goal_x, goal_y, goal_yaw):
		"""Tao position + queue mission Move qua REST API. Tra ve True neu thanh cong."""
		if not self._api_headers or not self._move_mission_guid:
			return False

		self._ensure_ready()

		status = self._api_status()
		if not status:
			return False
		map_id = status.get("map_id", "")
		if not map_id:
			rospy.logwarn("[NavNguoi] Khong tim thay map_id!")
			return False

		# Chuyen yaw sang do
		orientation_deg = math.degrees(goal_yaw)

		# Thu tao position va queue mission (chi thu 1 lan, khong offset)
		pos_name = f"_nav_person_{int(time.time())}"
		try:
			r = requests.post(f"{self.api_url}/positions", headers=self._api_headers, json={
				"name": pos_name,
				"pos_x": goal_x,
				"pos_y": goal_y,
				"orientation": orientation_deg,
				"type_id": 0,
				"map_id": map_id,
			}, timeout=5)
		except Exception:
			return False

		if r.status_code not in (200, 201):
			return False

		pos_guid = r.json().get("guid", "")

		# Xoa queue cu
		try:
			requests.delete(f"{self.api_url}/mission_queue", headers=self._api_headers, timeout=5)
		except Exception:
			pass

		# Queue mission
		try:
			r = requests.post(f"{self.api_url}/mission_queue", headers=self._api_headers, json={
				"mission_id": self._move_mission_guid,
				"parameters": [{"input_name": self._move_param_name, "value": pos_guid}],
			}, timeout=5)
		except Exception:
			self._api_delete_position(pos_guid)
			return False

		if r.status_code not in (200, 201):
			self._api_delete_position(pos_guid)
			return False

		# Cho 1.5s xem co loi khong
		time.sleep(1.5)
		st = self._api_status()
		if st and st.get("state_id") in (10, 12):
			err = str(st.get("errors", "")).lower()
			rospy.logwarn("[NavNguoi] REST API loi: %s", str(st.get("errors", ""))[:150])
			try:
				requests.delete(f"{self.api_url}/status", headers=self._api_headers, timeout=3)
			except Exception:
				pass
			time.sleep(0.5)
			self._ensure_ready()
			self._api_delete_position(pos_guid)
			return False

		# Thanh cong!
		rospy.loginfo("[NavNguoi] Mission queued! Position (%.2f, %.2f)", goal_x, goal_y)
		return True

	def _navigate_websocket(self, goal_x, goal_y, goal_yaw):
		"""Gui goal truc tiep qua WebSocket (backup)."""
		if self._ws_robot is None or not self._ws_robot.is_connected():
			return False

		now = rospy.Time.now()

		# MirMoveBaseActionGoal (giong navigationcacdiem.py)
		goal_action = {
			"header": {
				"seq": 0,
				"stamp": {"secs": now.secs, "nsecs": now.nsecs},
				"frame_id": "",
			},
			"goal_id": {
				"stamp": {"secs": now.secs, "nsecs": now.nsecs},
				"id": f"nav_person_{now.secs}_{now.nsecs}",
			},
			"goal": {
				"target_pose": {
					"header": {"seq": 0, "stamp": {"secs": 0, "nsecs": 0}, "frame_id": "map"},
					"pose": {
						"position": {"x": goal_x, "y": goal_y, "z": 0.0},
						"orientation": {
							"x": 0.0, "y": 0.0,
							"z": math.sin(goal_yaw / 2.0),
							"w": math.cos(goal_yaw / 2.0),
						},
					},
				},
				"move_task": 1,
				"goal_dist_threshold": 0.25,
				"clear_costmaps": True,
			},
		}
		self._ws_robot.send({
			"op": "publish",
			"topic": "/move_base/goal",
			"type": "mir_actions/MirMoveBaseActionGoal",
			"msg": goal_action,
		})
		rospy.loginfo("[NavNguoi] Gui goal qua WebSocket (%.2f, %.2f)", goal_x, goal_y)
		return True

	def _stop_robot(self):
		"""Gui lenh dung robot."""
		twist = Twist()
		self._cmd_vel_pub.publish(twist)

	def _normalize_angle(self, angle):
		while angle > math.pi:
			angle -= 2.0 * math.pi
		while angle < -math.pi:
			angle += 2.0 * math.pi
		return angle

	def _drive_to_goal(self, goal_x, goal_y):
		"""
		Proportional controller: dieu khien robot bang cmd_vel truc tiep.
		Dung khi REST API + WebSocket that bai (forbidden area).
		"""
		rospy.loginfo("[NavNguoi] === CMD_VEL MODE ===")
		rospy.loginfo("[NavNguoi] Toc do max: linear=%.2f m/s, angular=%.2f rad/s",
					  self.max_linear_speed, self.max_angular_speed)

		rate = rospy.Rate(10)  # 10 Hz
		start_time = time.time()

		while not rospy.is_shutdown():
			if (time.time() - start_time) > self.navigate_timeout:
				rospy.logwarn("[NavNguoi] Timeout cmd_vel!")
				self._stop_robot()
				return False

			robot_pose = self._get_robot_pose_map()
			if robot_pose is None:
				rate.sleep()
				continue

			robot_x, robot_y, robot_yaw = robot_pose
			dx = goal_x - robot_x
			dy = goal_y - robot_y
			distance = math.hypot(dx, dy)

			if distance <= self.goal_tolerance:
				self._stop_robot()
				rospy.loginfo("[NavNguoi] DA DEN NOI! Cach: %.2fm", distance)
				return True

			target_yaw = math.atan2(dy, dx)
			angle_diff = self._normalize_angle(target_yaw - robot_yaw)

			twist = Twist()

			if abs(angle_diff) > math.radians(30):
				# Xoay tai cho truoc
				twist.angular.z = max(-self.max_angular_speed,
									  min(self.max_angular_speed, angle_diff * 1.0))
			else:
				# Di thang + dieu chinh huong
				linear_speed = min(self.max_linear_speed, distance * 0.5)
				twist.linear.x = max(0.05, linear_speed)
				twist.angular.z = max(-self.max_angular_speed,
									  min(self.max_angular_speed, angle_diff * 1.5))

			self._cmd_vel_pub.publish(twist)

			rospy.loginfo_throttle(2.0,
				"[NavNguoi] cmd_vel: khoang_cach=%.2fm, goc_lech=%.1f do",
				distance, math.degrees(angle_diff))

			rate.sleep()

		self._stop_robot()
		return False

	def _wait_arrival(self, goal_x, goal_y):
		"""Doi robot den dich. Dung REST API + WebSocket."""
		was_executing = False
		start_time = time.time()
		last_log = 0

		# Cho robot bat dau executing
		time.sleep(2.0)

		while not rospy.is_shutdown() and (time.time() - start_time) < self.navigate_timeout:
			now = time.time()

			# Kiem tra trang thai qua REST API
			if self._api_headers and now - last_log > 2.0:
				last_log = now
				st = self._api_status()
				if st:
					state_id = st.get("state_id", -1)
					pos = st.get("position", {})
					rx = pos.get("x", 0)
					ry = pos.get("y", 0)
					dist = math.hypot(rx - goal_x, ry - goal_y)

					rospy.loginfo("[NavNguoi] Cach dich: %.2fm | state=%s(%s)",
								  dist, state_id, st.get("state_text", ""))

					if dist < self.arrive_dist:
						rospy.loginfo("[NavNguoi] DA DEN DICH! (cach %.2fm)", dist)
						return True

					if state_id == 5:  # Executing
						was_executing = True

					# Da executing xong -> ve Ready
					if was_executing and state_id == 3:
						# Kiem tra mission queue
						try:
							eq = requests.get(f"{self.api_url}/mission_queue",
											   headers=self._api_headers, timeout=3)
							if eq.status_code == 200 and eq.json():
								q_state = eq.json()[-1].get("state", "")
								if q_state == "Done":
									rospy.loginfo("[NavNguoi] Mission Done!")
									return True
						except Exception:
							pass
						rospy.loginfo("[NavNguoi] Mission hoan thanh (Ready).")
						return True

					# Error
					if state_id in (10, 12):
						errs = st.get("errors", [])
						rospy.logwarn("[NavNguoi] Robot Error! state=%s", state_id)
						if errs:
							rospy.logwarn("[NavNguoi] %s", str(errs[0].get("description", ""))[:120])
						return False

			time.sleep(0.5)

		rospy.logwarn("[NavNguoi] Timeout sau %.0fs!", self.navigate_timeout)
		return False

	def _api_delete_position(self, guid):
		try:
			requests.delete(f"{self.api_url}/positions/{guid}", headers=self._api_headers, timeout=3)
		except Exception:
			pass

	def _cleanup_temp_positions(self):
		"""Xoa cac position tam (_nav_person_*)."""
		if not self._api_headers:
			return
		try:
			r = requests.get(f"{self.api_url}/positions", headers=self._api_headers, timeout=5)
			for p in r.json():
				if p.get("name", "").startswith("_nav_person_"):
					self._api_delete_position(p["guid"])
		except Exception:
			pass

	# ═══════════════════════════════════════════
	#  Main workflow: collect -> countdown -> navigate
	# ═══════════════════════════════════════════
	def _collect_and_navigate_worker(self):
		try:
			# ===== PHASE 1: Thu thap =====
			rospy.loginfo("=====================================================")
			rospy.loginfo("[NavNguoi] DA NHAN MUC TIEU LOCK!")
			rospy.loginfo("[NavNguoi] PHASE 1: Thu thap vi tri trong %d giay...", self.collect_seconds)
			rospy.loginfo("=====================================================")

			for remaining in range(self.collect_seconds, 0, -1):
				if rospy.is_shutdown():
					return
				with self.lock:
					n = len(self._collected_positions)
				rospy.loginfo("[NavNguoi] Thu thap... %ds con lai (%d mau da nhan)", remaining, n)
				time.sleep(1.0)

			with self.lock:
				positions = list(self._collected_positions)
				self._phase = "countdown"

			if not positions:
				rospy.logwarn("[NavNguoi] Khong nhan duoc du lieu vi tri nao!")
				return

			avg_x = sum(p[0] for p in positions) / len(positions)
			avg_y = sum(p[1] for p in positions) / len(positions)

			robot_pose = self._get_robot_pose_map()
			if robot_pose is None:
				rospy.logwarn("[NavNguoi] Khong doc duoc pose robot.")
				return

			# Tinh goal: cach nguoi stop_distance met
			robot_x, robot_y, _ = robot_pose
			dx = avg_x - robot_x
			dy = avg_y - robot_y
			dist = math.hypot(dx, dy)

			if dist < self.min_person_distance:
				rospy.logwarn("[NavNguoi] Nguoi qua gan (%.2fm).", dist)
				return
			if dist > self.max_person_distance:
				rospy.logwarn("[NavNguoi] Nguoi qua xa (%.2fm).", dist)
				return

			ux = dx / max(dist, 1e-6)
			uy = dy / max(dist, 1e-6)
			goal_x = avg_x - ux * self.stop_distance
			goal_y = avg_y - uy * self.stop_distance
			goal_yaw = math.atan2(dy, dx)

			rospy.loginfo("=====================================================")
			rospy.loginfo("[NavNguoi] Trung binh tu %d mau:", len(positions))
			rospy.loginfo("[NavNguoi]   Vi tri nguoi (map): (%.2f, %.2f)", avg_x, avg_y)
			rospy.loginfo("[NavNguoi]   Goal (cach nguoi %.1fm): (%.2f, %.2f)", self.stop_distance, goal_x, goal_y)
			rospy.loginfo("[NavNguoi]   Khoang cach hien tai: %.2fm", dist)
			rospy.loginfo("=====================================================")

			# ===== PHASE 2: Dem nguoc =====
			rospy.loginfo("[NavNguoi] PHASE 2: Dem nguoc %d giay truoc khi di chuyen...", self.countdown_seconds)
			for remaining in range(self.countdown_seconds, 0, -1):
				if rospy.is_shutdown():
					return
				rospy.loginfo("[NavNguoi] >>> DI CHUYEN SAU %d GIAY <<<", remaining)
				time.sleep(1.0)

			if rospy.is_shutdown():
				return

			# ===== PHASE 3: Di chuyen =====
			with self.lock:
				self._phase = "moving"

			rospy.loginfo("=====================================================")
			rospy.loginfo("[NavNguoi] BAT DAU DI CHUYEN den (%.2f, %.2f)!", goal_x, goal_y)
			rospy.loginfo("=====================================================")

			# Thu REST API truoc (giong navigationcacdiem.py)
			rest_ok = self._navigate_rest_api(goal_x, goal_y, goal_yaw)

			if not rest_ok:
				# REST API that bai (forbidden area, site mismatch, etc.)
				# -> Dung cmd_vel truc tiep (bypass MiR planner)
				rospy.logwarn("[NavNguoi] REST API that bai! Dung cmd_vel truc tiep...")
				self._ensure_ready()
				success = self._drive_to_goal(goal_x, goal_y)
			else:
				# REST API thanh cong, doi robot den dich
				rospy.loginfo("[NavNguoi] Dang cho robot di chuyen (REST API)...")
				success = self._wait_arrival(goal_x, goal_y)

			if success:
				rospy.loginfo("=====================================================")
				rospy.loginfo("[NavNguoi] THANH CONG! Da den vi tri gan nguoi.")
				rospy.loginfo("=====================================================")
			else:
				rospy.logwarn("[NavNguoi] Khong den duoc vi tri muc tieu.")

		finally:
			self._stop_robot()
			self._cleanup_temp_positions()
			with self.lock:
				self._phase = "idle"
				self._collected_positions = []
				self.busy = False

	def _person_locked_cb(self, msg):
		with self.lock:
			phase = self._phase

		if phase == "collecting":
			try:
				self.tf_listener.waitForTransform("/map", msg.header.frame_id, rospy.Time(0), rospy.Duration(2.0))
				msg.header.stamp = rospy.Time(0)
				pt = self.tf_listener.transformPoint("/map", msg)
				with self.lock:
					self._collected_positions.append((pt.point.x, pt.point.y))
			except Exception:
				pass
			return

		if phase in ("countdown", "moving"):
			return

		# Phase "idle" -> bat dau thu thap
		try:
			self.tf_listener.waitForTransform("/map", msg.header.frame_id, rospy.Time(0), rospy.Duration(2.0))
			msg.header.stamp = rospy.Time(0)
			person_map_pt = self.tf_listener.transformPoint("/map", msg)
		except Exception as e:
			rospy.logwarn_throttle(2.0, "[NavNguoi] Chua transform duoc: %s", e)
			return

		robot_pose = self._get_robot_pose_map()
		if robot_pose is None:
			rospy.logwarn_throttle(2.0, "[NavNguoi] Chua doc duoc pose robot.")
			return

		px, py = person_map_pt.point.x, person_map_pt.point.y
		dist = math.hypot(px - robot_pose[0], py - robot_pose[1])
		if dist < self.min_person_distance or dist > self.max_person_distance:
			return

		with self.lock:
			self._phase = "collecting"
			self._collected_positions = [(px, py)]
			self.busy = True

		thread = threading.Thread(target=self._collect_and_navigate_worker, daemon=True)
		thread.start()


def main():
	rospy.init_node("navigation_nguoi_lock")
	_ = PersonLockNavigator()
	rospy.loginfo("[NavNguoi] Node san sang.")
	rospy.spin()


if __name__ == "__main__":
	main()
