#!/bin/bash
# =============================================================
#  MiR Robot Project - Khởi động 1 lệnh duy nhất
#  Cách dùng: ./start.sh [tuỳ chọn]
#    ./start.sh                    -> Khởi động tất cả (ROS1 + Bridge)
#    ./start.sh connect [port] [mir_ip] -> 1 lệnh kết nối web <-> robot (mặc định web 8080)
#    ./start.sh build              -> Build lại image rồi khởi động
#    ./start.sh stop               -> Dừng tất cả container
#    ./start.sh shell              -> Mở terminal vào container ROS1
#    ./start.sh gazebo             -> Chạy Gazebo simulation
#    ./start.sh joystick [IP]      -> Điều khiển MiR thật bằng tay cầm
#    ./start.sh rosbridge          -> Khởi động lại ROS WebSocket bridge (port 9090)
#    ./start.sh web [port]         -> Chạy web menu (mặc định port 8080)
#    ./start.sh roslaunch pkg file -> Chạy bất kỳ roslaunch nào
#    ./start.sh rviz [config]       -> Mở RViz (config: navigation|description|test)
#    ./start.sh run <file.py> [args] -> Chạy file Python trong thư mục tm
#    ./start.sh run list             -> Liệt kê các file Python trong thư mục tm
#    ./start.sh run-host <file.py> [args] -> Chạy file Python trên host (không qua Docker)
#    ./start.sh run-host list             -> Liệt kê file Python tm (host)
# =============================================================

set -e
cd "$(dirname "$0")"

WEB_DIR="$(pwd)/src/mir_robot/dung/web"

kill_host_port() {
    local port="$1"
    local pids=""

    if command -v lsof >/dev/null 2>&1; then
        pids="$(lsof -ti tcp:"${port}" 2>/dev/null || true)"
    elif command -v fuser >/dev/null 2>&1; then
        pids="$(fuser -n tcp "${port}" 2>/dev/null || true)"
    fi

    if [[ -n "${pids}" ]]; then
        echo "⚠️  Port ${port} đang được dùng bởi PID: ${pids} -> đang dừng..."
        kill ${pids} 2>/dev/null || true
        sleep 0.5
    fi
}

wait_ros_master() {
    local retries="${1:-30}"
    local delay="${2:-1}"

    echo "⏳ Đang chờ ROS Master sẵn sàng..."
    for _ in $(seq 1 "${retries}"); do
        if docker exec mir_noetic_env bash -lc "python3 - <<'PY'
import xmlrpc.client
try:
    proxy = xmlrpc.client.ServerProxy('http://localhost:11311')
    code, _, _ = proxy.getUri('/start_sh_check')
    raise SystemExit(0 if code == 1 else 1)
except Exception:
    raise SystemExit(1)
PY" >/dev/null 2>&1; then
            echo "✅ ROS Master đã sẵn sàng."
            return 0
        fi
        sleep "${delay}"
    done

    echo "❌ ROS Master chưa sẵn sàng sau ${retries}s."
    return 1
}

wait_port_open() {
    local port="$1"
    local retries="${2:-30}"
    local delay="${3:-1}"

    for _ in $(seq 1 "${retries}"); do
        if command -v nc >/dev/null 2>&1; then
            if nc -z localhost "${port}" >/dev/null 2>&1; then
                return 0
            fi
        elif command -v bash >/dev/null 2>&1; then
            if (echo > /dev/tcp/localhost/"${port}") >/dev/null 2>&1; then
                return 0
            fi
        fi
        sleep "${delay}"
    done

    return 1
}

extract_mir_ip_from_navigation_config() {
    local cfg_file="$(pwd)/src/mir_robot/tm/navigationcacdiem.py"
    local detected_ip=""

    if [[ -f "${cfg_file}" ]]; then
        detected_ip="$(grep -E '^[[:space:]]*MIR_IP[[:space:]]*=' "${cfg_file}" | head -n1 | sed -E 's/.*"([^"]+)".*/\1/' || true)"
    fi

    if [[ "${detected_ip}" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
        echo "${detected_ip}"
        return 0
    fi

    echo ""
}

resolve_mir_ip() {
    local cli_ip="$1"

    if [[ -n "${cli_ip}" ]]; then
        echo "${cli_ip}"
        return 0
    fi

    if [[ -n "${MIR_IP:-}" ]]; then
        echo "${MIR_IP}"
        return 0
    fi

    local cfg_ip=""
    cfg_ip="$(extract_mir_ip_from_navigation_config)"
    if [[ -n "${cfg_ip}" ]]; then
        echo "${cfg_ip}"
        return 0
    fi

    echo "192.168.12.20"
}

wait_topic_message() {
    local topic="$1"
    local retries="${2:-25}"
    local delay="${3:-1}"

    for _ in $(seq 1 "${retries}"); do
        if docker exec mir_noetic_env bash -lc "source /opt/ros/noetic/setup.bash && timeout 5s rostopic echo -n 1 ${topic} >/dev/null 2>&1"; then
            return 0
        fi
        sleep "${delay}"
    done

    return 1
}

start_web_background() {
    local web_port="$1"
    local log_file="/tmp/mir_web_${web_port}.log"

    kill_host_port "${web_port}"
    echo "🌍 Đang chạy web menu nền tại http://localhost:${web_port} ..."
    nohup python3 -m http.server "${web_port}" --directory "${WEB_DIR}" >"${log_file}" 2>&1 &
    echo "📄 Log web: ${log_file}"
}

start_order_listener_background() {
    echo "🧾 Khởi động order_listener (nhận đơn /robot_orders) ở chế độ nền..."
    docker exec -i mir_noetic_env bash -lc \
        "pkill -f '[o]rder_listener.py' || true; \
         pkill -f '[n]avigationcacdiem.py' || true; \
         rm -f /tmp/order_listener.log"

    docker exec -d mir_noetic_env bash -lc \
        "source /opt/ros/noetic/setup.bash && \
         [ -f /root/catkin_ws/devel/setup.bash ] && source /root/catkin_ws/devel/setup.bash; \
         PYTHONUNBUFFERED=1 nohup python3 /root/catkin_ws/src/mir_robot/tm/order_listener.py >/tmp/order_listener.log 2>&1 &"

    sleep 1
    if docker exec mir_noetic_env bash -lc "source /opt/ros/noetic/setup.bash && rosnode list | grep -q '^/order_listener$'"; then
        echo "✅ order_listener đã chạy."
        return 0
    fi

    echo "❌ order_listener chưa chạy. Log:"
    docker exec mir_noetic_env bash -lc "tail -n 120 /tmp/order_listener.log 2>/dev/null || echo NO_ORDER_LISTENER_LOG"
    return 1
}

start_mir_bridge_background() {
    local mir_ip="$1"

    echo "🤖 Khởi động mir_bridge tới MiR tại ${mir_ip} ở chế độ nền..."

    docker exec -i mir_noetic_env bash -lc \
        "pkill -f '[m]ir_bridge.py' || true; \
         pkill -f '[r]oslaunch mir_driver mir.launch' || true; \
         rm -f /tmp/mir_bridge.log"

    docker exec -d mir_noetic_env bash -lc \
        "source /opt/ros/noetic/setup.bash && \
         [ -f /root/catkin_ws/devel/setup.bash ] && source /root/catkin_ws/devel/setup.bash; \
         PYTHONUNBUFFERED=1 nohup roslaunch mir_driver mir.launch mir_hostname:=${mir_ip} >/tmp/mir_bridge.log 2>&1 &"

    echo "⏳ Đang chờ mir_bridge sẵn sàng..."
    for _ in $(seq 1 20); do
        if docker exec mir_noetic_env bash -lc "source /opt/ros/noetic/setup.bash && rosnode list | grep -q '^/mir_bridge$'"; then
            echo "✅ mir_bridge đã chạy (MiR: ${mir_ip})."
            return 0
        fi
        sleep 1
    done

    echo "❌ mir_bridge chưa chạy. Log:"
    docker exec mir_noetic_env bash -lc "tail -n 120 /tmp/mir_bridge.log 2>/dev/null || echo NO_MIR_BRIDGE_LOG"
    return 1
}

ensure_rosbridge_server() {
    if docker exec mir_noetic_env bash -lc "source /opt/ros/noetic/setup.bash && rospack find rosbridge_server" >/dev/null 2>&1; then
        return 0
    fi

    echo "📦 Thiếu rosbridge_server trong container -> đang cài đặt..."
    docker exec mir_noetic_env bash -lc "apt-get update && apt-get install -y ros-noetic-rosbridge-server"

    if docker exec mir_noetic_env bash -lc "source /opt/ros/noetic/setup.bash && rospack find rosbridge_server" >/dev/null 2>&1; then
        echo "✅ Đã cài rosbridge_server."
        return 0
    fi

    echo "❌ Không cài được rosbridge_server."
    return 1
}

start_rosbridge_background() {
    echo "🌐 Khởi động ROS WebSocket bridge (port 9090) ở chế độ nền..."

    docker exec -i mir_noetic_env bash -lc \
        "pkill -f '[r]osbridge_websocket' || true; \
         pkill -f '[r]oslaunch rosbridge_server' || true; \
         rm -f /tmp/rosbridge.log"

    docker exec -d mir_noetic_env bash -lc \
        "source /opt/ros/noetic/setup.bash && \
         nohup roslaunch rosbridge_server rosbridge_websocket.launch port:=9090 >/tmp/rosbridge.log 2>&1 &"

    sleep 2

    if ! docker exec mir_noetic_env bash -lc "pgrep -f rosbridge_websocket >/dev/null"; then
        echo "❌ rosbridge không chạy được. Log:"
        docker exec mir_noetic_env bash -lc "tail -n 120 /tmp/rosbridge.log 2>/dev/null || echo NO_ROSBRIDGE_LOG"
        return 1
    fi

    if wait_port_open 9090 25 1; then
        echo "✅ ROS bridge đã mở tại ws://localhost:9090"
        return 0
    fi

    echo "❌ rosbridge có process nhưng cổng 9090 chưa mở. Log:"
    docker exec mir_noetic_env bash -lc "tail -n 120 /tmp/rosbridge.log 2>/dev/null || echo NO_ROSBRIDGE_LOG"
    return 1
}

# Cho phép Docker hiển thị GUI
xhost +local:docker 2>/dev/null || true

case "${1:-start}" in
    connect)
        WEB_PORT="${2:-8080}"
        MIR_IP="$(resolve_mir_ip "${3:-}")"

        echo "🚀 Khởi động containers..."
        docker compose up -d
        echo "🎯 MiR IP dùng cho kết nối: ${MIR_IP}"

        wait_ros_master 45 1
        ensure_rosbridge_server
        start_rosbridge_background
        start_mir_bridge_background "${MIR_IP}"

        echo "⏳ Đang chờ robot thật publish /mir_status_msg ..."
        if wait_topic_message "/mir_status_msg" 20 1; then
            echo "✅ Robot thật đã publish dữ liệu trạng thái."
        else
            echo "❌ Robot chưa publish /mir_status_msg. Kết nối chưa hoàn tất."
            echo "📄 Log mir_bridge:"
            docker exec mir_noetic_env bash -lc "tail -n 120 /tmp/mir_bridge.log 2>/dev/null || echo NO_MIR_BRIDGE_LOG"
            exit 1
        fi

        start_order_listener_background

        start_web_background "${WEB_PORT}"

        echo ""
        echo "✅ Hoàn tất kết nối."
        echo "   - Web:       http://localhost:${WEB_PORT}"
        echo "   - Rosbridge: ws://localhost:9090"
        echo "   - MiR host:  ${MIR_IP}"
        echo "   - Topic đơn: /robot_orders"
        echo ""
        echo "🔎 Kiểm tra đơn hàng từ robot:"
        echo "   docker exec -it mir_noetic_env bash -lc 'source /opt/ros/noetic/setup.bash && rostopic echo /robot_orders'"
        ;;

    build)
        echo "🔨 Đang build lại Docker images..."
        docker compose build
        echo "🚀 Đang khởi động containers..."
        docker compose up -d
        echo ""
        echo "✅ Đã khởi động xong!"
        echo "   - mir_noetic_env: ROS1 Noetic (MiR Robot)"
        echo "   - mir_ros1_bridge: ROS1 <-> ROS2 Bridge"
        echo ""
        echo "📌 Để vào container ROS1:  ./start.sh shell"
        echo "📌 Để chạy Gazebo:         ./start.sh gazebo"
        ;;

    start)
        echo "🚀 Đang khởi động containers..."
        docker compose up -d
        echo ""
        echo "✅ Đã khởi động xong!"
        echo "   - mir_noetic_env: ROS1 Noetic (MiR Robot)"
        echo "   - mir_ros1_bridge: ROS1 <-> ROS2 Bridge"
        echo ""
        echo "📌 Để vào container ROS1:  ./start.sh shell"
        echo "📌 Để chạy Gazebo:         ./start.sh gazebo"
        echo ""
        echo "📡 Từ host ROS2 Humble, bạn có thể:"
        echo "   ros2 topic list          -> Xem topic từ ROS1"
        echo "   ros2 topic echo /scan    -> Đọc dữ liệu laser"
        ;;

    stop)
        WEB_PORT="${2:-8080}"
        echo "🛑 Đang dừng tất cả containers..."
        docker compose down
        echo "🧹 Dọn tiến trình web nền trên host (port ${WEB_PORT})..."
        kill_host_port "${WEB_PORT}"
        pkill -f "[h]ttp.server ${WEB_PORT}" 2>/dev/null || true
        echo "✅ Đã dừng."
        echo "   - Containers: stopped"
        echo "   - Host web: port ${WEB_PORT} cleaned"
        ;;

    shell)
        echo "🔧 Đang mở terminal vào container ROS1..."
        docker exec -it mir_noetic_env bash
        ;;

    gazebo)
        echo "🌍 Đang khởi động Gazebo MiR Robot..."
        docker exec -it mir_noetic_env bash -c \
            "source /opt/ros/noetic/setup.bash && \
             source /root/catkin_ws/devel/setup.bash && \
             roslaunch mir_gazebo mir_maze_world.launch"
        ;;

    roslaunch)
        # Chạy bất kỳ roslaunch command nào
        # Ví dụ: ./start.sh roslaunch mir_navigation amcl.launch
        shift
        echo "🚀 Đang chạy: roslaunch $*"
        docker exec -it mir_noetic_env bash -c \
            "source /opt/ros/noetic/setup.bash && \
             source /root/catkin_ws/devel/setup.bash && \
             roslaunch $*"
        ;;

    rosbridge)
        ensure_rosbridge_server
        start_rosbridge_background
        ;;

    web)
        WEB_PORT="${2:-8080}"
        kill_host_port "${WEB_PORT}"
        echo "🌍 Đang chạy web menu tại http://localhost:${WEB_PORT} ..."
        python3 -m http.server "${WEB_PORT}" --directory "${WEB_DIR}"
        ;;

    joystick)
        # Điều khiển MiR thật bằng tay cầm (KẾT NỐI TRỰC TIẾP - không cần mir_bridge)
        # Ví dụ: ./start.sh joystick 192.168.0.177
        MIR_IP="${2:-192.168.12.20}"
        echo "🎮 Đang kết nối Joystick TRỰC TIẾP đến MiR tại $MIR_IP..."
        echo ""
        echo "📌 Hướng dẫn:"
        echo "   - GIỮ NÚT R1/RB trước khi đẩy cần gạt"
        echo "   - Cần gạt trái: Điều khiển tiến/lùi/xoay"
        echo "   - Nút Y: Tăng tốc   |   Nút A: Giảm tốc"
        echo "   - Nhả R1 = Robot dừng ngay"
        echo ""
        docker exec -it mir_noetic_env bash -c \
            "source /opt/ros/noetic/setup.bash && \
             source /root/catkin_ws/devel/setup.bash && \
             roslaunch mir_driver mir_joystick_direct.launch mir_hostname:=$MIR_IP"
        ;;

    rviz)
        # Mở RViz - tuỳ chọn config: navigation | description | test
        # Ví dụ: ./start.sh rviz
        #         ./start.sh rviz navigation
        #         ./start.sh rviz description
        RVIZ_CONFIG=""
        case "${2:-}" in
            navigation)
                RVIZ_CONFIG="-d /root/catkin_ws/src/mir_robot/mir_navigation/rviz/navigation.rviz"
                ;;
            description)
                RVIZ_CONFIG="-d /root/catkin_ws/src/mir_robot/mir_description/rviz/mir_description.rviz"
                ;;
            test)
                RVIZ_CONFIG="-d /root/catkin_ws/src/mir_robot/tm/rviz/testrviz.rviz"
                ;;
        esac
        echo "🖥️  Đang mở RViz${2:+ (config: $2)}..."
        docker exec -it mir_noetic_env bash -c \
            "source /opt/ros/noetic/setup.bash && \
             source /root/catkin_ws/devel/setup.bash && \
             rosrun rviz rviz $RVIZ_CONFIG"
        ;;

    run)
        # Chạy file Python trong thư mục tm
        # Ví dụ: ./start.sh run navigationcacdiem.py bep
        #         ./start.sh run navigationcacdiem.py "ban 1"
        #         ./start.sh run list
        TM_DIR="/root/catkin_ws/src/mir_robot/tm"
        if [[ -z "${2:-}" || "${2}" == "list" ]]; then
            echo "📂 Các file Python trong thư mục tm:"
            docker exec mir_noetic_env bash -c \
                "ls ${TM_DIR}/*.py 2>/dev/null | xargs -I{} basename {}"
        else
            SCRIPT="${2}"
            shift 2
            ARGS="$*"
            echo "🐍 Đang chạy: python3 ${SCRIPT} ${ARGS}"
            docker exec -it mir_noetic_env bash -c \
                "source /opt/ros/noetic/setup.bash && \
                 source /root/catkin_ws/devel/setup.bash && \
                 pkill -f '^python3[[:space:]]+${SCRIPT}([[:space:]]|$)' 2>/dev/null || true && \
                 sleep 0.3 && \
                 [ -z "$LIBGL_ALWAYS_SOFTWARE" ] && export LIBGL_ALWAYS_SOFTWARE=1; export QT_XCB_GL_INTEGRATION=none; export FORCE_PULSE_CAPTURE=1 && export QT_QPA_PLATFORM=xcb && \
                 cd ${TM_DIR} && python3 ${SCRIPT} ${ARGS}"
        fi
        ;;

    run-host)
        # Chạy file Python trên host để dùng trực tiếp camera/GPU của máy
        TM_DIR_HOST="$(pwd)/src/mir_robot/tm"
        if [[ -z "${2:-}" || "${2}" == "list" ]]; then
            echo "📂 Các file Python trong thư mục tm (host):"
            ls "${TM_DIR_HOST}"/*.py 2>/dev/null | xargs -I{} basename {}
        else
            SCRIPT="${2}"
            shift 2
            ARGS="$*"
            if [[ ! -f "${TM_DIR_HOST}/${SCRIPT}" ]]; then
                echo "❌ Không tìm thấy file: ${TM_DIR_HOST}/${SCRIPT}"
                exit 1
            fi

            PYTHON_BIN="python3"
            if [[ -x "$(pwd)/.venv-gpu/bin/python" ]]; then
                PYTHON_BIN="$(pwd)/.venv-gpu/bin/python"
            elif [[ -x "$(pwd)/.venv/bin/python" ]]; then
                PYTHON_BIN="$(pwd)/.venv/bin/python"
            fi

            if [[ "${SCRIPT}" == "khoangcach3d.py" ]]; then
                export KHOANGCACH3D_DEVICE="${KHOANGCACH3D_DEVICE:-gpu}"
                export KHOANGCACH3D_INFER_EVERY="${KHOANGCACH3D_INFER_EVERY:-2}"
                export KHOANGCACH3D_HANDS_EVERY="${KHOANGCACH3D_HANDS_EVERY:-3}"
                export KHOANGCACH3D_IMGSZ="${KHOANGCACH3D_IMGSZ:-416}"
            fi

            echo "🐍 Đang chạy host: ${PYTHON_BIN} ${SCRIPT} ${ARGS}"
            if [[ "${SCRIPT}" == "khoangcach3d.py" ]]; then
                echo "⚙️ Auto perf env: KHOANGCACH3D_DEVICE=${KHOANGCACH3D_DEVICE} KHOANGCACH3D_INFER_EVERY=${KHOANGCACH3D_INFER_EVERY} KHOANGCACH3D_HANDS_EVERY=${KHOANGCACH3D_HANDS_EVERY} KHOANGCACH3D_IMGSZ=${KHOANGCACH3D_IMGSZ}"
            else
                echo "📌 Gợi ý GPU: KHOANGCACH3D_DEVICE=gpu ./start.sh run-host ${SCRIPT}"
            fi
            export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
            export QT_QPA_FONTDIR="${QT_QPA_FONTDIR:-/usr/share/fonts/truetype/dejavu}"
            export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-none}"
            export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
            export OPENCV_FFMPEG_CAPTURE_OPTIONS="${OPENCV_FFMPEG_CAPTURE_OPTIONS:-video_codec;rawvideo}"
            export PYTHONUNBUFFERED=1
            cd "${TM_DIR_HOST}" && exec "${PYTHON_BIN}" "${SCRIPT}" ${ARGS}
        fi
        ;;

    joystick-bridge)
        # Điều khiển qua mir_bridge (cách cũ - cần mir_bridge)
        MIR_IP="${2:-192.168.12.20}"
        echo "🎮 Đang kết nối Joystick qua mir_bridge đến MiR tại $MIR_IP..."
        docker exec -it mir_noetic_env bash -c \
            "source /opt/ros/noetic/setup.bash && \
             source /root/catkin_ws/devel/setup.bash && \
             roslaunch mir_driver mir_joystick_teleop.launch mir_hostname:=$MIR_IP"
        ;;

    *)
        echo "Cách dùng: ./start.sh [connect|start|build|stop|shell|gazebo|rosbridge|web|rviz|joystick|roslaunch|run|run-host ...]"
        echo ""
        echo "Ví dụ chạy Python:"
        echo "   ./start.sh run list"
        echo "   ./start.sh run navigationcacdiem.py bep"
        echo "   ./start.sh run navigationcacdiem.py ban1"
        exit 1
        ;;
esac
