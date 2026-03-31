import cv2
import cv2.aruco as aruco
import math
import socket
import time
import threading

# ── Config ───────────────────────────────────────────────────────────────────

BOT_IDS = [3]

BOT_IPS = {
    3: "192.168.0.116",  # <-- update this to your bot's actual IP
}

PORT          = 80
CAMERA_INDEX  = 1
USE_SOCKETS   = True  # Set True once network is confirmed working

SEND_INTERVAL    = 0.1
REACH_DIST       = 10
ANGLE_TOLERANCE  = 35

WINDOW_NAME = "Path Control"

# ── Commands ─────────────────────────────────────────────────────────────────

CMD_STOP    = "S"
CMD_FORWARD = "F"
CMD_LEFT    = "L"
CMD_RIGHT   = "R"

# ── State ────────────────────────────────────────────────────────────────────

waypoints = []  # Left click in the window to add a waypoint

waypoints_lock    = threading.Lock()
current_bot_state = {bot_id: CMD_STOP for bot_id in BOT_IDS}
last_send_time    = {bot_id: 0.0      for bot_id in BOT_IDS}

# ── Networking ───────────────────────────────────────────────────────────────

_sockets = {}

def get_socket(bot_id):
    global _sockets
    if bot_id in _sockets:
        return _sockets[bot_id]
    host = BOT_IPS[bot_id]
    try:
        s = socket.socket()
        s.settimeout(0.5)
        s.connect((host, PORT))
        _sockets[bot_id] = s
        print(f"[NET] Connected to bot {bot_id} at {host}:{PORT}")
        return s
    except Exception as e:
        print(f"[NET] Could not connect to bot {bot_id}: {e}")
        return None

def send_command(bot_id, cmd):
    if not USE_SOCKETS:
        print(f"[SIM] Bot {bot_id} -> {cmd}")
        return
    s = get_socket(bot_id)
    if s is None:
        return
    try:
        s.sendall(cmd.encode())
        print(f"[NET] Sent to bot {bot_id}: {cmd}")
    except Exception as e:
        print(f"[NET] Send error for bot {bot_id}: {e} — dropping socket")
        _sockets.pop(bot_id, None)

# ── Mouse ─────────────────────────────────────────────────────────────────────

def mouse_click(event, x, y, flags, param):
    with waypoints_lock:
        if event == cv2.EVENT_LBUTTONDOWN:
            waypoints.append((x, y))
            print(f"[CLICK] Added waypoint at ({x}, {y}) — total: {len(waypoints)}")
        elif event == cv2.EVENT_RBUTTONDOWN:
            waypoints.clear()
            print("[CLICK] Cleared all waypoints")
            for bot_id in BOT_IDS:
                send_command(bot_id, CMD_STOP)
                current_bot_state[bot_id] = CMD_STOP

# ── Maths ─────────────────────────────────────────────────────────────────────

def wrap_angle_deg(a):
    return ((a + 180) % 360) - 180

def get_marker_pose(marker_corners):
    topLeft, topRight, bottomRight, bottomLeft = marker_corners

    cx = int((topLeft[0] + bottomRight[0]) / 2.0)
    cy = int((topLeft[1] + bottomRight[1]) / 2.0)

    front_x = (topLeft[0] + topRight[0]) / 2.0
    front_y = (topLeft[1] + topRight[1]) / 2.0

    angle_rad = math.atan2(cy - front_y, front_x - cx)
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360

    return cx, cy, int(front_x), int(front_y), angle_deg

def compute_command(cx, cy, angle_deg, target_x, target_y):
    distance = math.hypot(target_x - cx, target_y - cy)

    if distance < REACH_DIST:
        return CMD_STOP, distance, 0.0

    desired_angle = math.degrees(math.atan2(cy - target_y, target_x - cx))
    if desired_angle < 0:
        desired_angle += 360

    heading_error = wrap_angle_deg(desired_angle - angle_deg)

    if heading_error > ANGLE_TOLERANCE:
        return CMD_LEFT, distance, heading_error
    elif heading_error < -ANGLE_TOLERANCE:
        return CMD_RIGHT, distance, heading_error
    else:
        return CMD_FORWARD, distance, heading_error

# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_overlay(frame, bot_poses, waypoints_snapshot):
    for i, wp in enumerate(waypoints_snapshot):
        color = (0, 255, 255) if i == 0 else (255, 255, 0)
        cv2.circle(frame, wp, 8, color, -1)
        cv2.putText(frame, str(i + 1), (wp[0] + 8, wp[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if i > 0:
            cv2.line(frame, waypoints_snapshot[i - 1], waypoints_snapshot[i], (255, 255, 0), 2)

    for bot_id, bot in bot_poses.items():
        cx, cy = bot["cx"], bot["cy"]
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.line(frame, (cx, cy), (bot["front_x"], bot["front_y"]), (0, 255, 0), 2)
        cv2.putText(frame, f"Bot {bot_id}", (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if waypoints_snapshot:
            target_x, target_y = waypoints_snapshot[0]
            cmd = current_bot_state.get(bot_id, CMD_STOP)
            dist = math.hypot(target_x - cx, target_y - cy)
            cv2.line(frame, (cx, cy), (target_x, target_y), (255, 0, 0), 2)
            cv2.putText(frame, f"{cmd} d={dist:.0f}",
                        (cx + 10, cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.putText(frame, f"Waypoints: {len(waypoints_snapshot)}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, "SIM MODE" if not USE_SOCKETS else "LIVE", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255) if not USE_SOCKETS else (0, 255, 0), 2)

# ── Detect bots ───────────────────────────────────────────────────────────────

def detect_bots(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    bot_poses = {}
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        for i in range(len(ids)):
            bot_id = int(ids[i][0])
            if bot_id in BOT_IDS:
                cx, cy, fx, fy, angle = get_marker_pose(corners[i][0])
                bot_poses[bot_id] = {"cx": cx, "cy": cy,
                                     "front_x": fx, "front_y": fy,
                                     "angle_deg": angle}
    return bot_poses

# ── Control loop ──────────────────────────────────────────────────────────────

def control_loop(bot_poses, waypoints_snapshot, now):
    if not waypoints_snapshot:
        for bot_id in BOT_IDS:
            if now - last_send_time[bot_id] > SEND_INTERVAL:
                send_command(bot_id, CMD_STOP)
                current_bot_state[bot_id] = CMD_STOP
                last_send_time[bot_id] = now
        return

    target_x, target_y = waypoints_snapshot[0]
    reached = 0

    for bot_id in BOT_IDS:
        if bot_id not in bot_poses:
            print(f"[WARN] Bot {bot_id} not detected in frame")
            continue

        bot = bot_poses[bot_id]
        cmd, distance, heading_error = compute_command(
            bot["cx"], bot["cy"], bot["angle_deg"], target_x, target_y
        )

        print(f"[BOT {bot_id}] cmd={cmd}  dist={distance:.1f}  heading_err={heading_error:.1f}")

        if distance < REACH_DIST:
            reached += 1

        if now - last_send_time[bot_id] > SEND_INTERVAL:
            send_command(bot_id, cmd)
            current_bot_state[bot_id] = cmd
            last_send_time[bot_id] = now

    detected = len([b for b in BOT_IDS if b in bot_poses])
    if detected > 0 and reached == detected:
        print(f"[NAV] Reached waypoint {waypoints_snapshot[0]}")
        with waypoints_lock:
            if waypoints:
                waypoints.pop(0)

# ── Setup ─────────────────────────────────────────────────────────────────────

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector   = aruco.ArucoDetector(aruco_dict, parameters)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError("Could not open camera")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1200, 800)
cv2.setMouseCallback(WINDOW_NAME, mouse_click)

print("=" * 50)
print("Bot controller started")
print(f"Mode: {'LIVE SOCKETS' if USE_SOCKETS else 'SIM (no sockets)'}")
print("Left click  → add waypoint")
print("Right click → clear waypoints")
print("Q           → quit")
print("=" * 50)

# ── Main loop ─────────────────────────────────────────────────────────────────

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read camera frame")
        break

    bot_poses = detect_bots(frame)

    with waypoints_lock:
        waypoints_snapshot = list(waypoints)

    draw_overlay(frame, bot_poses, waypoints_snapshot)
    control_loop(bot_poses, waypoints_snapshot, time.time())

    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        for bot_id in BOT_IDS:
            send_command(bot_id, CMD_STOP)
        break

cap.release()
cv2.destroyAllWindows()