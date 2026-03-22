import cv2
import cv2.aruco as aruco
import math
import socket

# ------------------------ CONFIGURATION ------------------------
LEFT_PUSHER_ID = 1
RIGHT_PUSHER_ID = 2

BOT_IPS = {
    1: "192.168.0.101",
    2: "192.168.0.102",
}

PORT = 80
USE_SOCKETS = True

# clicked navigation waypoints for the TROLLEY
waypoints = []

# last sent state for each bot
current_bot_state = {
    LEFT_PUSHER_ID: "S",
    RIGHT_PUSHER_ID: "S",
}

# temporary trolley pose for testing
trolley_x = 500
trolley_y = 300
trolley_angle_deg = 0

# formation distances (pixels)
BACK_DIST = 100
SIDE_DIST = 70

# tolerances
REACH_DIST = 30
ANGLE_TOLERANCE = 15

# steering thresholds
STEER_BIAS_THRESHOLD = 12
BIG_TURN_THRESHOLD = 25


# ------------------------ NETWORK ------------------------
def send_command(bot_id, cmd):
    if not USE_SOCKETS:
        print(f"[SIMULATED] Bot {bot_id} -> {cmd}")
        return

    host = BOT_IPS[bot_id]

    try:
        s = socket.socket()
        s.settimeout(0.5)
        s.connect((host, PORT))
        s.send(cmd.encode())
        s.close()
    except Exception as e:
        print(f"Socket Error for bot {bot_id}: {e}")


# ------------------------ MOUSE INPUT ------------------------
def mouse_click(event, x, y, flags, param):
    global waypoints, current_bot_state, trolley_x, trolley_y

    if event == cv2.EVENT_LBUTTONDOWN:
        waypoints.append((x, y))
        print(f"Added trolley waypoint: ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        waypoints.clear()
        for bot_id in BOT_IPS:
            send_command(bot_id, "S")
            current_bot_state[bot_id] = "S"
        print("Waypoints cleared. All bots stopped.")

    elif event == cv2.EVENT_MBUTTONDOWN:
        trolley_x, trolley_y = x, y
        print(f"Set trolley center to: ({x}, {y})")


# ------------------------ HELPERS ------------------------
def wrap_angle_deg(a):
    while a > 180:
        a -= 360
    while a < -180:
        a += 360
    return a


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

    return cx, cy, front_x, front_y, angle_deg


def compute_command(cx, cy, angle_deg, target_x, target_y, reach_dist=40, angle_tolerance=15):
    distance = math.hypot(target_x - cx, target_y - cy)

    if distance < reach_dist:
        return "S", distance, 0.0

    target_angle_rad = math.atan2(cy - target_y, target_x - cx)
    target_angle_deg = math.degrees(target_angle_rad)
    if target_angle_deg < 0:
        target_angle_deg += 360

    heading_error = target_angle_deg - angle_deg
    heading_error = wrap_angle_deg(heading_error)

    if heading_error > angle_tolerance:
        return "L", distance, heading_error
    elif heading_error < -angle_tolerance:
        return "R", distance, heading_error
    else:
        return "F", distance, heading_error


def get_trolley_targets(tx, ty, trolley_angle_deg, back_dist=100, side_dist=70):
    """
    Returns two targets behind the trolley:
    - left rear pusher
    - right rear pusher
    """
    theta = math.radians(trolley_angle_deg)

    # forward direction in image coordinates
    forward_x = math.cos(theta)
    forward_y = -math.sin(theta)

    # left and right perpendicular directions
    left_x = -forward_y
    left_y = forward_x

    right_x = forward_y
    right_y = -forward_x

    rear_center_x = tx - back_dist * forward_x
    rear_center_y = ty - back_dist * forward_y

    left_target = (
        rear_center_x + side_dist * left_x,
        rear_center_y + side_dist * left_y
    )

    right_target = (
        rear_center_x + side_dist * right_x,
        rear_center_y + side_dist * right_y
    )

    return {
        LEFT_PUSHER_ID: left_target,
        RIGHT_PUSHER_ID: right_target
    }


def get_desired_trolley_angle(tx, ty, wx, wy):
    angle = math.degrees(math.atan2(ty - wy, wx - tx))
    if angle < 0:
        angle += 360
    return angle


# ------------------------ ARUCO SETUP ------------------------
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

cap = cv2.VideoCapture(1)

cv2.namedWindow("2-Bot Trolley Control")
cv2.setMouseCallback("2-Bot Trolley Control", mouse_click)

print("Starting 2-Bot Trolley Control")
print("LEFT CLICK   = add trolley waypoint")
print("MIDDLE CLICK = set trolley center")
print("RIGHT CLICK  = clear waypoints and stop")
print("[ and ]      = rotate trolley angle for testing")
print("q            = quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    # ------------------------ DRAW WAYPOINTS ------------------------
    for i in range(len(waypoints)):
        cv2.circle(frame, waypoints[i], 8, (255, 255, 0), -1)
        cv2.putText(
            frame, str(i + 1),
            (waypoints[i][0] + 10, waypoints[i][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2
        )
        if i > 0:
            cv2.line(frame, waypoints[i - 1], waypoints[i], (255, 255, 0), 2)

    # ------------------------ DETECT BOTS ------------------------
    bot_poses = {}

    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)

        for i in range(len(ids)):
            bot_id = int(ids[i][0])

            if bot_id in BOT_IPS:
                marker_corners = corners[i][0]
                cx, cy, front_x, front_y, angle_deg = get_marker_pose(marker_corners)

                bot_poses[bot_id] = {
                    "cx": cx,
                    "cy": cy,
                    "front_x": front_x,
                    "front_y": front_y,
                    "angle_deg": angle_deg,
                }

                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.line(frame, (cx, cy), (int(front_x), int(front_y)), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"Bot {bot_id}",
                    (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )

    # ------------------------ TROLLEY NAVIGATION ------------------------
    steering_bias = 0.0

    if len(waypoints) > 0:
        target_wx, target_wy = waypoints[0]

        cv2.line(
            frame,
            (int(trolley_x), int(trolley_y)),
            (int(target_wx), int(target_wy)),
            (0, 255, 255),
            2
        )

        trolley_to_wp_dist = math.hypot(target_wx - trolley_x, target_wy - trolley_y)

        if trolley_to_wp_dist < 40:
            print(f"Trolley reached waypoint: {waypoints[0]}")
            waypoints.pop(0)
        else:
            desired_trolley_angle = get_desired_trolley_angle(trolley_x, trolley_y, target_wx, target_wy)
            steering_bias = wrap_angle_deg(desired_trolley_angle - trolley_angle_deg)

            # fake trolley turning for testing only
            turn_rate = 2.0
            if steering_bias > 5:
                trolley_angle_deg += turn_rate
            elif steering_bias < -5:
                trolley_angle_deg -= turn_rate

            trolley_angle_deg %= 360

    # ------------------------ DRAW TROLLEY ------------------------
    theta = math.radians(trolley_angle_deg)
    heading_len = 60
    hx = int(trolley_x + heading_len * math.cos(theta))
    hy = int(trolley_y - heading_len * math.sin(theta))

    cv2.circle(frame, (int(trolley_x), int(trolley_y)), 10, (0, 255, 255), -1)
    cv2.line(frame, (int(trolley_x), int(trolley_y)), (hx, hy), (0, 255, 255), 3)
    cv2.putText(
        frame, f"Trolley angle: {trolley_angle_deg:.1f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
    )

    # ------------------------ FORMATION TARGETS ------------------------
    targets = get_trolley_targets(
        trolley_x,
        trolley_y,
        trolley_angle_deg,
        back_dist=BACK_DIST,
        side_dist=SIDE_DIST
    )

    # ------------------------ BOT CONTROL ------------------------
    for bot_id in [LEFT_PUSHER_ID, RIGHT_PUSHER_ID]:
        if bot_id not in bot_poses:
            continue

        bot = bot_poses[bot_id]
        target_x, target_y = targets[bot_id]

        color = (255, 0, 0) if bot_id == LEFT_PUSHER_ID else (0, 0, 255)

        cv2.circle(frame, (int(target_x), int(target_y)), 7, color, -1)
        cv2.line(frame, (bot["cx"], bot["cy"]), (int(target_x), int(target_y)), color, 2)

        desired_state, distance, heading_error = compute_command(
            bot["cx"],
            bot["cy"],
            bot["angle_deg"],
            target_x,
            target_y,
            reach_dist=REACH_DIST,
            angle_tolerance=ANGLE_TOLERANCE
        )

        # ------------------------ STEERING ASSIST ------------------------
        # Once near target positions, bias push strength using trolley heading error.
        # Since we only have F/L/R/S, we fake "push more" by letting one side go forward
        # while the other side pauses.
        if distance < 50 and len(waypoints) > 0:
            if bot_id == LEFT_PUSHER_ID:
                # if trolley should turn right, left side pushes more
                if steering_bias < -STEER_BIAS_THRESHOLD:
                    desired_state = "F"
                elif steering_bias > STEER_BIAS_THRESHOLD:
                    desired_state = "S"

            elif bot_id == RIGHT_PUSHER_ID:
                # if trolley should turn left, right side pushes more
                if steering_bias > STEER_BIAS_THRESHOLD:
                    desired_state = "F"
                elif steering_bias < -STEER_BIAS_THRESHOLD:
                    desired_state = "S"

            # if turn is very large, don't let both sides charge badly misaligned
            if abs(steering_bias) > BIG_TURN_THRESHOLD:
                if bot_id == LEFT_PUSHER_ID and steering_bias > 0:
                    desired_state = "S"
                if bot_id == RIGHT_PUSHER_ID and steering_bias < 0:
                    desired_state = "S"

        if desired_state != current_bot_state[bot_id]:
            print(f"Bot {bot_id}: {current_bot_state[bot_id]} -> {desired_state}")
            send_command(bot_id, desired_state)
            current_bot_state[bot_id] = desired_state

        cv2.putText(
            frame,
            f"BOT {bot_id}: {current_bot_state[bot_id]}",
            (bot["cx"] + 10, bot["cy"] + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # ------------------------ SAFETY ------------------------
    for bot_id in BOT_IPS:
        if bot_id not in bot_poses and current_bot_state[bot_id] != "S":
            print(f"Bot {bot_id} lost -> stop")
            send_command(bot_id, "S")
            current_bot_state[bot_id] = "S"

    cv2.imshow("2-Bot Trolley Control", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        for bot_id in BOT_IPS:
            send_command(bot_id, "S")
        break
    elif key == ord('['):
        trolley_angle_deg = (trolley_angle_deg - 5) % 360
    elif key == ord(']'):
        trolley_angle_deg = (trolley_angle_deg + 5) % 360

cap.release()
cv2.destroyAllWindows()