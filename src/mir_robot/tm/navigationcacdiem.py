
def api_find_charge_mission(headers):
    try:
        r = requests.get(f"{API_URL}/missions", headers=headers, timeout=5)
        missions = r.json()
    except Exception:
        return None
    for m in missions:
        name = m.get("name", "").lower()
        if "charge" in name or "sac" in name or "docking" in name:
            return m.get("guid")
    return None
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Navigation đơn giản - di chuyển MiR đến tọa độ định sẵn.

Sử dụng:
  - REST API để tạo position + queue mission (cách chính thức)
  - WebSocket để gửi goal trực tiếp (backup)
  - WebSocket để theo dõi vị trí robot real-time

Cách dùng:
    python3 navigationcacdiem.py bep
    python3 navigationcacdiem.py ban1
    python3 navigationcacdiem.py "ban 1"
    python3 navigationcacdiem.py pos        # xem vị trí hiện tại
    python3 navigationcacdiem.py list       # liệt kê positions trên MiR
    python3 navigationcacdiem.py test       # test tọa độ có hợp lệ không
    python3 navigationcacdiem.py mirpos <tên>  # navigate tới MiR position
    python3 navigationcacdiem.py            # chế độ tương tác
"""

import sys
import math
import time
import base64
import json
import hashlib
import rospy
from mir_driver.rosbridge import RosbridgeSetup

try:
    import requests
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests

# ── Cấu hình ──
MIR_IP = "192.168.0.177"
MIR_PORT = 9090
TIMEOUT = 120       # giây
ARRIVE_DIST = 0.5   # mặc định nếu điểm không khai báo riêng

# REST API
API_URL = f"http://{MIR_IP}/api/v2.0.0"
CREDENTIALS = [
    ("distributor", "distributor"),
    ("admin", "admin"),
    ("Admin", "admin"),
    ("service", "service"),
]

# ── Các điểm đích ──
DIEM = {
    "sac":   {"x": 21.950, "y": 15.600, "qz": 0, "qw": 1, "arrive_dist": 0.5},
    "bep":   {"x": 5.5,  "y": 17.05, "qz": 0.707, "qw": 0.707, "arrive_dist": 1.0},
    "ban 1": {"x": 6.900, "y": 19.100, "qz": 0, "qw": 1, "arrive_dist": 0.9},
}


def get_arrive_dist(diem):
    d = diem.get("arrive_dist", ARRIVE_DIST)
    try:
        return float(d)
    except Exception:
        return ARRIVE_DIST


def tim_diem(ten):
    ten = ten.strip().lower()
    if ten in DIEM:
        return ten
    for k in DIEM:
        if k.replace(" ", "") == ten.replace(" ", ""):
            return k
    return None


def quat_to_deg(qz, qw):
    """Chuyển quaternion (qz, qw) thành góc độ."""
    return math.degrees(2 * math.atan2(qz, qw))


# ════════════════════════════════════════════
#  REST API
# ════════════════════════════════════════════
def api_login():
    """Đăng nhập REST API, trả về headers hoặc None.
    MiR REST API yêu cầu password SHA-256 hashed: Basic base64(user:sha256hex(pw))
    """
    # Thử SHA-256 hashed password (chuẩn MiR API)
    for user, pw in CREDENTIALS:
        pw_hash = hashlib.sha256(pw.encode()).hexdigest()
        auth = base64.b64encode(f"{user}:{pw_hash}".encode()).decode()
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Basic {auth}"
        }
        try:
            r = requests.get(f"{API_URL}/missions", headers=headers, timeout=5)
            if r.status_code == 200:
                print(f"  REST API OK ({user}, sha256)")
                return headers
        except Exception:
            pass

    # Fallback: thử plain password
    for user, pw in CREDENTIALS:
        auth = base64.b64encode(f"{user}:{pw}".encode()).decode()
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Basic {auth}"
        }
        try:
            r = requests.get(f"{API_URL}/missions", headers=headers, timeout=5)
            if r.status_code == 200:
                print(f"  REST API OK ({user}, plain)")
                return headers
        except Exception:
            pass

    print("  REST API: login failed!")
    print("  Đã thử SHA-256 và plain password cho:", [c[0] for c in CREDENTIALS])
    return None


def api_status(headers):
    try:
        return requests.get(f"{API_URL}/status", headers=headers, timeout=5).json()
    except Exception:
        return None


def api_set_state(headers, state_id):
    try:
        r = requests.put(f"{API_URL}/status", headers=headers, json={"state_id": state_id}, timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def api_ensure_ready(headers):
    """Đảm bảo robot READY, xóa error nếu cần."""
    status = api_status(headers)
    if not status:
        return False

    state = status.get("state_id", -1)
    print(f"  Robot state: {state} ({status.get('state_text', '')})")

    if state == 3:
        return True

    if state in (10, 12):  # Error
        print("  Đang xóa lỗi ...")
        try:
            # MiR v2 API yêu cầu PUT clear_error = true cho các lỗi cứng
            requests.put(f"{API_URL}/status", headers=headers, json={"clear_error": True}, timeout=3)
            time.sleep(0.5)
            requests.delete(f"{API_URL}/status", headers=headers, timeout=5)
            time.sleep(1)
        except Exception:
            pass

    if api_set_state(headers, 3):
        print("  Robot READY!")
        time.sleep(1)
        return True

    # Fallback: pause then ready
    api_set_state(headers, 4)
    time.sleep(0.5)
    if api_set_state(headers, 3):
        print("  Robot READY!")
        time.sleep(1)
        return True

    print("  Không thể chuyển về READY! Xóa lỗi trên web interface.")
    return False


def api_list_positions(headers):
    """Liệt kê tất cả position có sẵn trên MiR."""
    try:
        r = requests.get(f"{API_URL}/positions", headers=headers, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []


def api_create_position(headers, x, y, qz, qw, name):
    """Tạo position tạm, trả về (guid, True) hoặc (None, False)."""
    status = api_status(headers)
    if not status:
        return None, False
    map_id = status.get("map_id", "")
    if not map_id:
        return None, False
    orientation = quat_to_deg(qz, qw)
    r = requests.post(f"{API_URL}/positions", headers=headers, json={
        "name": name,
        "pos_x": x,
        "pos_y": y,
        "orientation": orientation,
        "type_id": 0,
        "map_id": map_id
    }, timeout=5)
    if r.status_code in (200, 201):
        return r.json().get("guid", ""), True
    return None, False


def api_delete_position(headers, guid):
    try:
        requests.delete(f"{API_URL}/positions/{guid}", headers=headers, timeout=3)
    except Exception:
        pass


def api_find_move_mission(headers):
    """Tìm mission Move, trả về (guid, param_name) hoặc (None, None)."""
    try:
        r = requests.get(f"{API_URL}/missions", headers=headers, timeout=5)
        missions = r.json()
    except Exception:
        return None, None

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
        return None, None

    # Lấy parameter name
    param_name = None
    try:
        r = requests.get(f"{API_URL}/missions/{move_guid}/actions", headers=headers, timeout=5)
        if r.status_code == 200:
            for act in r.json():
                for p in act.get("parameters", []):
                    inp = p.get("input_name", "")
                    if inp and inp != "None":
                        param_name = inp
                        break
                if param_name:
                    break
    except Exception:
        pass
    return move_guid, param_name or "Position"




def api_charge(headers, marker_guid, ten_diem):
    print("  Dùng mission ChargeAtStation...")
    
    # XÓA QUEUE CŨ ĐỂ KHÔNG BỊ TRÔI LỆNH TRƯỚC ĐÓ (ví dụ đang gọi rớt lại lệnh đi bếp)
    try:
        requests.delete(f"{API_URL}/mission_queue", headers=headers, timeout=5)
        print("  Đã xóa hàng đợi MiR cũ.")
    except Exception as e:
        print("  Không thể xóa hàng đợi MiR:", e)

    body = {
        "mission_id": "mirconst-guid-0000-0004-actionlist00",
        "parameters": [
            {"id": "chargingStationPosition", "value": marker_guid}
        ]
    }
    r = requests.post(f"{API_URL}/mission_queue", headers=headers, json=body, timeout=5)
    if r.status_code == 201:
        mission_id = r.json().get("id")
        print(f"  Đã đưa Charge mission vào queue (ID: {mission_id})")
        return True
    else:
        print(f"  Lỗi queue Charge mission: {r.status_code} {r.text}")
        return False

def api_navigate(headers, diem, ten_diem):
    # Di chuyển bình thường
    """
    Di chuyển robot bằng REST API.
    Nếu vị trí bị obstacle, tự dịch nhẹ để tìm vị trí hợp lệ.
    """
    api_ensure_ready(headers) # Đánh thức xe nếu đang Pause (đèn tím)
    
    status = api_status(headers)
    if not status:
        print("  Không đọc được status!")
        return False

    map_id = status.get("map_id", "")
    if not map_id:
        print("  Không tìm thấy map_id!")
        return False
    print(f"  Map ID: {map_id[:12]}...")

    # Tìm mission Move
    move_guid, param_name = api_find_move_mission(headers)
    if not move_guid:
        print("  Không tìm thấy mission Move!")
        return False
    print(f"  Mission 'Move' found, param='{param_name}'")

    # Tạo position và queue — nếu obstacle thì thử điểm tiếp theo trong danh sách ứng viên
    if isinstance(diem, list):
        candidates_to_try = diem
    else:
        candidates_to_try = [diem]

    for c_diem in candidates_to_try:
        x = c_diem["x"]
        y = c_diem["y"]
        dist_m = c_diem.get("dist_m", "N/A")
        
        suffix = f" (cự ly {dist_m}m)" if dist_m != "N/A" else ""

        pos_name = f"_nav_{ten_diem}_{int(time.time())}"
        orientation = quat_to_deg(c_diem["qz"], c_diem["qw"])

        r = requests.post(f"{API_URL}/positions", headers=headers, json={
            "name": pos_name,
            "pos_x": x,
            "pos_y": y,
            "orientation": orientation,
            "type_id": 0,
            "map_id": map_id
        }, timeout=5)

        if r.status_code not in (200, 201):
            continue

        pos_guid = r.json().get("guid", "")
        print(f"  Position ({x:.3f}, {y:.3f}){suffix}")

        # Xóa queue cũ
        try:
            requests.delete(f"{API_URL}/mission_queue", headers=headers, timeout=5)
        except Exception:
            pass

        # Queue mission
        r = requests.post(f"{API_URL}/mission_queue", headers=headers, json={
            "mission_id": move_guid,
            "parameters": [{"input_name": param_name, "value": pos_guid}]
        }, timeout=5)

        if r.status_code not in (200, 201):
            api_delete_position(headers, pos_guid)
            continue

        print(f"  Mission queued!")

        # Chờ 3 giây xem có lỗi obstacle không
        time.sleep(3)
        st = api_status(headers)
        if st and st.get("state_id") in (10, 12):
            err = str(st.get("errors", ""))
            # Thêm 'forbidden zone' vào điều kiện catch lỗi
            if "obstacle" in err.lower() or "forbidden area" in err.lower() or "forbidden zone" in err.lower():
                print(f"  ⚠ Lỗi ({err[:80]}), thử offset khác...")
                # Xóa error cực mạnh
                try:
                    requests.put(f"{API_URL}/status", headers=headers, json={"clear_error": True}, timeout=3)
                    requests.delete(f"{API_URL}/status", headers=headers, timeout=3)
                except Exception:
                    pass
                time.sleep(1)
                
                # Ép robot về Pause (4) trước để reset state machine, sau đó mới lên Ready (3)
                api_set_state(headers, 4)
                time.sleep(0.5)
                api_set_state(headers, 3)
                time.sleep(0.5)
                
                # Kiểm tra lại xem đã thoát lỗi chưa
                st_check = api_status(headers)
                if st_check and st_check.get("state_id") in (10, 12):
                    print("  ❌ KHÔNG THỂ CLEAR ERROR BẰNG API! Vui lòng ấn Reset trên Web MiR.")
                    api_delete_position(headers, pos_guid)
                    return False
                    
                api_delete_position(headers, pos_guid)
                continue
            else:
                print(f"  Lỗi không phải obstacle: {err[:200]}")
                api_delete_position(headers, pos_guid)
                return False

        # Đang executing hoặc ready — thành công!
        return True

    print("  ❌ Tất cả vị trí đều bị obstacle!")
    print("  Tọa độ này không hợp lệ trên map MiR.")
    print("  Hãy dùng: ./start.sh run navigationcacdiem.py list")
    print("  để xem các position có sẵn trên MiR.")
    return False


# ════════════════════════════════════════════
#  WebSocket
# ════════════════════════════════════════════
def ws_connect():
    """Kết nối WebSocket đến MiR."""
    print(f"Kết nối WebSocket đến MiR {MIR_IP}:{MIR_PORT} ...")
    robot = RosbridgeSetup(MIR_IP, MIR_PORT)

    for i in range(150):
        if rospy.is_shutdown():
            sys.exit(0)
        if robot.is_connected():
            print("Đã kết nối!")
            return robot
        if robot.is_errored():
            print("Lỗi kết nối!")
            sys.exit(1)
        time.sleep(0.1)

    print("Timeout!")
    sys.exit(1)


def ws_get_position(robot, timeout=5.0):
    """Lấy vị trí robot từ /robot_pose qua WebSocket."""
    result = [None, None]

    def cb(msg):
        try:
            result[0] = msg["position"]["x"]
            result[1] = msg["position"]["y"]
        except (KeyError, TypeError):
            pass

    robot.subscribe("/robot_pose", cb)
    try:
        t0 = time.time()
        while result[0] is None and (time.time() - t0) < timeout:
            time.sleep(0.1)
        return result[0], result[1]
    finally:
        robot.unhook(cb)


def ws_send_goal(robot, diem):
    """Gửi goal trực tiếp qua WebSocket (bao gồm type field)."""
    now = rospy.Time.now()

    # Cách 1: /move_base_simple/goal (PoseStamped - đơn giản)
    goal_simple = {
        "header": {
            "seq": 0,
            "stamp": {"secs": now.secs, "nsecs": now.nsecs},
            "frame_id": "map"
        },
        "pose": {
            "position": {"x": diem["x"], "y": diem["y"], "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": diem["qz"], "w": diem["qw"]}
        }
    }

    # Gửi với type field (quan trọng! MiR rosbridge cần type)
    robot.send({
        "op": "publish",
        "topic": "/move_base_simple/goal",
        "type": "geometry_msgs/PoseStamped",
        "msg": goal_simple
    })
    print("  Gửi goal qua WS /move_base_simple/goal")

    time.sleep(0.3)

    # Cách 2: /move_base/goal (MirMoveBaseActionGoal - đầy đủ hơn)
    now = rospy.Time.now()
    goal_action = {
        "header": {
            "seq": 0,
            "stamp": {"secs": now.secs, "nsecs": now.nsecs},
            "frame_id": ""
        },
        "goal_id": {
            "stamp": {"secs": now.secs, "nsecs": now.nsecs},
            "id": f"nav_{now.secs}_{now.nsecs}"
        },
        "goal": {
            "target_pose": {
                "header": {"seq": 0, "stamp": {"secs": 0, "nsecs": 0}, "frame_id": "map"},
                "pose": {
                    "position": {"x": diem["x"], "y": diem["y"], "z": 0.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": diem["qz"], "w": diem["qw"]}
                }
            },
            "move_task": 1,
            "goal_dist_threshold": 0.25,
            "clear_costmaps": True
        }
    }
    robot.send({
        "op": "publish",
        "topic": "/move_base/goal",
        "type": "mir_actions/MirMoveBaseActionGoal",
        "msg": goal_action
    })
    print("  Gửi goal qua WS /move_base/goal")


def wait_arrival(robot, diem, headers=None, timeout=TIMEOUT, rest_mode=False, cancel_event=None):
    """Chờ robot đến đích. Dùng WebSocket + REST API polling."""
    pos       = {"x": None, "y": None}
    ws_st     = {"val": None, "was_active": False}
    was_exec  = [False]   # dùng list để closure có thể ghi

    def pose_cb(msg):
        try:
            pos["x"] = msg["position"]["x"]
            pos["y"] = msg["position"]["y"]
        except (KeyError, TypeError):
            pass

    def status_cb(msg):
        try:
            sl = msg.get("status_list", [])
            if sl:
                ws_st["val"] = sl[-1].get("status", -1)
        except (KeyError, TypeError):
            pass

    robot.subscribe("/robot_pose", pose_cb)
    robot.subscribe("/move_base/status", status_cb)

    deadline  = time.time() + timeout
    last_log  = 0    # timer in log (mỗi 3s)
    last_rest = 0    # timer REST API — timer RIÊNG (mỗi 2s)

    arrive_dist = get_arrive_dist(diem)

    while not rospy.is_shutdown() and time.time() < deadline:
        if cancel_event and cancel_event.is_set():
            print("  [wait_arrival] Hủy vòng lặp theo yêu cầu!")
            if headers:
                # Đưa robot về Pause (4) để đảm bảo dừng di chuyển ngay lập tức
                api_set_state(headers, 4)
                try:
                    requests.delete(f"{API_URL}/mission_queue", headers=headers, timeout=3)
                except Exception:
                    pass
                # Khôi phục trạng thái Ready (3) để sẵn sàng nhận lệnh mới
                api_set_state(headers, 3)
            robot.unhook(pose_cb)
            robot.unhook(status_cb)
            return "cancelled"

        now = time.time()

        # ── Cập nhật vị trí từ REST nếu WS chưa có ──
        if pos["x"] is None and headers and now - last_rest > 2.0:
            try:
                st = api_status(headers)
                if st and "position" in st:
                    pos["x"] = st["position"].get("x")
                    pos["y"] = st["position"].get("y")
            except Exception:
                pass

        # ── Kiểm tra khoảng cách ──
        if pos["x"] is not None:
            dist = math.sqrt((pos["x"] - diem["x"])**2 + (pos["y"] - diem["y"])**2)

            if now - last_log > 3.0:
                extra = ""
                if headers:
                    try:
                        st = api_status(headers)
                        if st:
                            extra = f" | state={st.get('state_id','')}({st.get('state_text','')})"
                            mt = st.get('mission_text', '')
                            if mt:
                                extra += f" | {mt[:60]}"
                    except Exception:
                        pass
                print(f"  Cách đích: {dist:.2f}m | ws={ws_st['val']}{extra}")
                last_log = now

            if dist < arrive_dist:
                if not rest_mode:
                    print(f"  ✓ Đã đến (WebSocket Mode)! (cách đích {dist:.2f}m)")
                    robot.unhook(pose_cb)
                    robot.unhook(status_cb)
                    return True
                else:
                    # Trong chế độ REST, dù khoảng cách đã đạt yêu cầu nhưng xe vẫn có thể đang lăn bánh (Executing).
                    # Ta không return True ngay để tránh vỡ State Machine, phải chờ REST báo Done!
                    if now - last_log > 3.0:
                        print(f"  [Chờ dừng hẳn] Đã vào vùng đích (cách {dist:.2f}m), đang chờ MiR phanh lại...")

        # ── REST API polling — timer RIÊNG, không dùng last_log ──
        if headers and now - last_rest > 2.0:
            last_rest = now
            try:
                rst = api_status(headers)
                if rst:
                    mir_state = rst.get("state_id", -1)

                    if mir_state == 5:           # Executing
                        was_exec[0] = True

                    # Hoàn thành mission trong REST mode
                    if rest_mode and mir_state == 3:
                        try:
                            eq = requests.get(f"{API_URL}/mission_queue", headers=headers, timeout=3)
                            if eq.status_code == 200 and eq.json():
                                q_state = eq.json()[-1].get("state", "")
                                if q_state == "Done":
                                    print(f"  ✓ Mission Done!")
                                    robot.unhook(pose_cb)
                                    robot.unhook(status_cb)
                                    return True
                                elif q_state not in ("Executing", "Pending"):
                                    print(f"  ✓ Mission hoàn thành (queue={q_state})")
                                    robot.unhook(pose_cb)
                                    robot.unhook(status_cb)
                                    return True
                                # Nếu vẫn Executing/Pending: nếu từng thấy executing thì chờ tiếp,
                                # còn chưa từng thấy executing thì cũng chờ thêm để tránh false positive.
                            else:
                                print(f"  ✓ Mission hoàn thành (state Ready)")
                                robot.unhook(pose_cb)
                                robot.unhook(status_cb)
                                return True
                        except Exception:
                            if was_exec[0]:
                                print(f"  ✓ Mission hoàn thành (state Ready)")
                                robot.unhook(pose_cb)
                                robot.unhook(status_cb)
                                return True

                    # Lỗi
                    if mir_state in (10, 12):
                        errs = rst.get("errors", [])
                        mt   = rst.get("mission_text", "")
                        print(f"  Robot Error! (state={mir_state})")
                        if errs:
                            print(f"  {str(errs[0].get('description',''))[:120]}")
                        if mt:
                            print(f"  Mission: {mt}")
                        robot.unhook(pose_cb)
                        robot.unhook(status_cb)
                        return "error"

            except Exception:
                pass

        # ── WebSocket status (chỉ khi không dùng REST mode) ──
        if not rest_mode:
            sv = ws_st["val"]
            if sv == 1:
                ws_st["was_active"] = True
            elif ws_st["was_active"] and sv == 3:
                robot.unhook(pose_cb)
                robot.unhook(status_cb)
                return True
            elif ws_st["was_active"] and sv in (4, 5, 8):
                print(f"  WS status lỗi ({sv})")
                robot.unhook(pose_cb)
                robot.unhook(status_cb)
                return False

        time.sleep(0.5)

    print(f"  Timeout sau {timeout}s!")
    robot.unhook(pose_cb)
    robot.unhook(status_cb)
    return False


# ════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════
def show_menu():
    print("\n" + "─" * 45)
    print("Các điểm:")
    for k, v in DIEM.items():
        print(f"  {k:8s} -> x={v['x']:.3f}, y={v['y']:.3f}")
    print(f"  {'pos':8s} -> Vị trí hiện tại")
    print(f"  {'list':8s} -> Positions trên MiR")
    print(f"  {'test':8s} -> Test tọa độ")
    print(f"  {'quit':8s} -> Thoát")
    print("─" * 45)


def handle_command(ten, robot, headers, non_interactive=False, cancel_event=None):
    """Xử lý một lệnh. Trả về True nếu muốn thoát."""

    # ── pos ──
    if ten == "pos":
        x, y = ws_get_position(robot)
        if x is not None:
            print(f"Vị trí robot: x={x:.3f}, y={y:.3f}")
            for k, v in DIEM.items():
                d = math.sqrt((x - v["x"])**2 + (y - v["y"])**2)
                print(f"  {k:8s} -> {d:.2f}m")
        else:
            print("Không nhận được vị trí!")
        return False

    # ── list ──
    if ten == "list":
        if not headers:
            print("Không có REST API!")
            return False
        positions = api_list_positions(headers)
        if not positions:
            print("Không có position nào.")
            return False
        print(f"\n{len(positions)} positions trên MiR:")
        print(f"  {'Tên':30s} {'X':>10s} {'Y':>10s} {'Angle':>8s}")
        print("  " + "-" * 60)
        for p in positions:
            name = p.get('name', '?')
            if name.startswith("_nav_") or name.startswith("_test_"):
                continue
            print(f"  {name:30s} {p.get('pos_x',0):10.3f} {p.get('pos_y',0):10.3f} {p.get('orientation',0):8.1f}")
        return False

    # ── test ──
    if ten == "test":
        if not headers:
            print("Không có REST API!")
            return False
        move_guid, param_name = api_find_move_mission(headers)
        st = api_status(headers)
        map_id = st.get("map_id", "") if st else ""
        print(f"Map: {map_id[:12]}...")
        for name, d in DIEM.items():
            pos_name = f"_test_{name}_{int(time.time())}"
            r = requests.post(f"{API_URL}/positions", headers=headers, json={
                "name": pos_name, "pos_x": d["x"], "pos_y": d["y"],
                "orientation": quat_to_deg(d["qz"], d["qw"]),
                "type_id": 0, "map_id": map_id
            }, timeout=5)
            if r.status_code not in (200, 201):
                print(f"  {name:8s} -> ❌ Create failed"); continue
            guid = r.json().get("guid", "")
            try:
                requests.delete(f"{API_URL}/mission_queue", headers=headers, timeout=3)
            except Exception:
                pass
            api_ensure_ready(headers)
            r2 = requests.post(f"{API_URL}/mission_queue", headers=headers, json={
                "mission_id": move_guid,
                "parameters": [{"input_name": param_name, "value": guid}]
            }, timeout=5)
            if r2.status_code in (200, 201):
                time.sleep(3)
                check = api_status(headers)
                if check and check.get("state_id") in (10, 12):
                    err = str(check.get("errors", ""))
                    label = "OBSTACLE" if "obstacle" in err.lower() else f"Error: {err[:60]}"
                    print(f"  {name:8s} ({d['x']:7.3f}, {d['y']:7.3f}) -> ❌ {label}")
                    try:
                        requests.delete(f"{API_URL}/status", headers=headers, timeout=3)
                    except Exception:
                        pass
                    time.sleep(1)
                    api_set_state(headers, 3)
                else:
                    print(f"  {name:8s} ({d['x']:7.3f}, {d['y']:7.3f}) -> ✅ OK")
                    try:
                        requests.delete(f"{API_URL}/mission_queue", headers=headers, timeout=3)
                    except Exception:
                        pass
            else:
                print(f"  {name:8s} -> ❌ Queue failed")
            api_delete_position(headers, guid)
            time.sleep(1)
        api_ensure_ready(headers)
        return False

    # ── mirpos <tên> ──
    if ten.startswith("mirpos "):
        mirpos_name = ten[7:].strip()
        if not headers:
            print("Không có REST API!")
            return False
        positions = api_list_positions(headers)
        target = next((p for p in positions if p.get('name','').lower() == mirpos_name.lower()), None)
        if not target:
            target = next((p for p in positions if mirpos_name.lower() in p.get('name','').lower()), None)
        if not target:
            print(f"Không tìm thấy '{mirpos_name}'!")
            return False
        move_guid, param_name = api_find_move_mission(headers)
        if not move_guid:
            print("Không có mission Move!")
            return False
        api_ensure_ready(headers)
        try:
            requests.delete(f"{API_URL}/mission_queue", headers=headers, timeout=5)
        except Exception:
            pass
        r = requests.post(f"{API_URL}/mission_queue", headers=headers, json={
            "mission_id": move_guid,
            "parameters": [{"input_name": param_name, "value": target['guid']}]
        }, timeout=5)
        if r.status_code not in (200, 201):
            print(f"Queue lỗi: {r.status_code}")
            return False
        print(f"Đang đến '{target['name']}' ({target.get('pos_x',0):.3f}, {target.get('pos_y',0):.3f}) ...")
        diem_tmp = {"x": target.get('pos_x',0), "y": target.get('pos_y',0), "qz": 0, "qw": 1}
        result = wait_arrival(robot, diem_tmp, headers, rest_mode=True, cancel_event=cancel_event)
        if result == "cancelled":
            print(f"\n✋ Đã hủy lệnh đến '{target['name']}'!")
            return False
        print(f"\n{'✅ Đã đến' if result is True else '❌ Không đến được'} '{target['name']}'!")
        return False

    # ── navigate đến DIEM ──
    ten_chuan = tim_diem(ten)
    if ten_chuan is None:
        print(f"Không tìm thấy '{ten}'. Nhập 'list' để xem positions trên MiR.")
        return False

    diem = DIEM[ten_chuan]
    
    # Ưu tiên lấy vị trí từ REST API để luôn chính xác nhất (không bị trễ như websocket nếu chạy lâu)
    x, y = None, None
    if headers:
        try:
            st = api_status(headers)
            if st and "position" in st:
                x = st["position"].get("x")
                y = st["position"].get("y")
        except Exception:
            pass
            
    if x is None or y is None:
        x, y = ws_get_position(robot, timeout=2.0)
        
    if x is not None:
        d0 = math.sqrt((x - diem["x"])**2 + (y - diem["y"])**2)
        print(f"Vị trí hiện tại: x={x:.3f}, y={y:.3f} (cách đích {d0:.2f}m)")
        if d0 < get_arrive_dist(diem):
            print(f"✅ Robot đã ở gần '{ten_chuan}' rồi, bỏ qua lệnh di chuyển.")
            return False

    if headers:
        api_ensure_ready(headers)

    print(f"Đi đến '{ten_chuan}' (x={diem['x']:.3f}, y={diem['y']:.3f}) ...")

    MAX_RETRIES = 1 if non_interactive else 2
    ok = False
    for attempt in range(1, MAX_RETRIES + 1):
        if attempt > 1:
            print(f"\nThử lại lần {attempt}/{MAX_RETRIES} ...")
            if headers:
                api_ensure_ready(headers)
            time.sleep(2)

        rest_ok = False
        if headers:
            if ten_chuan == "sac":
                # marker guid cho 'sac'
                rest_ok = api_charge(headers, "ba61401e-2206-11f1-8f53-000129af8ea5", ten_chuan)
            else:
                rest_ok = api_navigate(headers, diem, ten_chuan)

        if not rest_ok:
            print("WebSocket goal (backup):")
            ws_send_goal(robot, diem)
        else:
            print("[REST API OK]")

        print("Đang chờ robot đến đích ...")
        timeout_sec = 90 if non_interactive else TIMEOUT
        result = wait_arrival(robot, diem, headers, timeout=timeout_sec, rest_mode=rest_ok, cancel_event=cancel_event)

        if result == "cancelled":
            ok = "cancelled"
            break
        elif result is True:
            ok = True
            break
        elif result == "error" and attempt < MAX_RETRIES:
            print("Sẽ thử lại...")
            if headers:
                try:
                    r = requests.get(f"{API_URL}/positions", headers=headers, timeout=5)
                    for p in r.json():
                        if p.get("name", "").startswith("_nav_"):
                            requests.delete(f"{API_URL}/positions/{p['guid']}", headers=headers, timeout=3)
                except Exception:
                    pass
        else:
            break

    if ok == "cancelled":
        print(f"\n✋ Đã hủy lệnh đến '{ten_chuan}' theo yêu cầu!")
    elif ok:
        print(f"\n✅ Đã đến '{ten_chuan}'!")
    else:
        print(f"\n❌ Không đến được '{ten_chuan}'.")
        if headers:
            try:
                st = api_status(headers)
                if st:
                    pos_r = st.get('position', {})
                    print(f"  Robot pos: x={pos_r.get('x','?'):.3f}, y={pos_r.get('y','?'):.3f}")
            except Exception:
                pass

    # Dọn dẹp position tạm
    if headers:
        try:
            r = requests.get(f"{API_URL}/positions", headers=headers, timeout=5)
            for p in r.json():
                if p.get("name", "").startswith("_nav_"):
                    requests.delete(f"{API_URL}/positions/{p['guid']}", headers=headers, timeout=3)
        except Exception:
            pass

    return False


def main():
    rospy.init_node("navigate_diem", anonymous=True)

    # ── Kết nối một lần duy nhất ──
    robot = ws_connect()

    print("\nKiểm tra robot qua REST API ...")
    headers = api_login()
    if headers:
        api_ensure_ready(headers)
    else:
        print("REST API không khả dụng - chỉ dùng WebSocket.")

    # ── Nếu có argument → chạy 1 lần rồi thoát (mode non-interactive) ──
    if len(sys.argv) >= 2:
        ten_arg = " ".join(sys.argv[1:]).strip().lower()
        print()
        handle_command(ten_arg, robot, headers, non_interactive=True)
        return

    # ── Vòng lặp tương tác ──
    while not rospy.is_shutdown():
        show_menu()
        try:
            ten = input("Nhập lệnh: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nThoát.")
            break

        if ten in ("quit", "exit", "q", "thoat"):
            print("Thoát.")
            break

        if not ten:
            continue

        handle_command(ten, robot, headers, non_interactive=False)


if __name__ == "__main__":
    main()
