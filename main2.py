"""

MONA Bot – Freehand Path Follower  (definitive version)

=========================================================

Draw a path by clicking and dragging on the live camera feed.

The bot follows it using pure-pursuit with pulse-width turn correction.

 

CONTROLS

--------

  Left-click + drag   Draw path (replaces current path, bot starts immediately)

  Right-click         Clear path and stop

  SPACE               Pause / resume

  Q                   Quit (bot stopped first)

 

TUNING GUIDE

------------

  LOOKAHEAD_PX        Look-ahead radius for pure-pursuit.

                        ↑ = smoother but cuts corners more

                        ↓ = tighter corners but more oscillation

  WAYPOINT_SPACING_PX Resampling density.  ↓ = more faithful path reproduction

  ANGLE_TOLERANCE     Dead-band (deg).  Below this → go straight

  TURN_THRESHOLD      Above this → stop and spin in place rather than pulse-curve

  TURN_PULSE_RATIO    Duty cycle of turn command while curving (0=always fwd, 1=always turn)

  PULSE_CYCLE_TIME    Full F/turn cycle length (seconds)

  SMOOTHING_FRAMES    Pose-averaging window.  ↑ = smoother, ↑ = lag

  MIN_CMD_INTERVAL    Hard floor between Wi-Fi commands (seconds)

"""

 

import cv2

import cv2.aruco as aruco

import numpy as np

import math

import socket

import time

from collections import deque

 

# ═══════════════════════════════════════════════════════════════════════════════

#  CONFIGURATION

# ═══════════════════════════════════════════════════════════════════════════════

 

BOT_ID      = 3

HOST        = "192.168.0.116"

PORT        = 80

USE_SOCKETS = True

 

# Path

WAYPOINT_SPACING_PX = 10    # resample drawn path to this density (px)

 

# Navigation

LOOKAHEAD_PX    = 55        # pure-pursuit look-ahead distance (px)

ARRIVAL_PX      = 18        # distance to final point that counts as "arrived"

ANGLE_TOLERANCE = 8         # deg – error below this → straight forward

TURN_THRESHOLD  = 32        # deg – error above this → spin in place

 

# Pulse-width curve correction

# For errors between ANGLE_TOLERANCE and TURN_THRESHOLD, the bot alternates

# between F and a turn command to arc without dedicated curve motors.

TURN_PULSE_RATIO = 0.40     # fraction of each cycle spent turning

PULSE_CYCLE_TIME = 0.16     # seconds per full F/turn cycle

 

# Pose smoothing

SMOOTHING_FRAMES = 4        # frames to average position and heading

 

# Network

MIN_CMD_INTERVAL = 0.08     # seconds – rate limit (≈12 cmds/s max)

WATCHDOG_INTERVAL = 0.5     # re-send current command to keep ESP32 alive

 

# ═══════════════════════════════════════════════════════════════════════════════

#  STATE

# ═══════════════════════════════════════════════════════════════════════════════

 

path_points: list  = []     # full drawn path

waypoints:   list  = []     # remaining points the bot must visit

paused:      bool  = False

_drawing:    bool  = False

 

# Pose smoothing buffers

pos_buf   = deque(maxlen=SMOOTHING_FRAMES)

angle_buf = deque(maxlen=SMOOTHING_FRAMES)

 

# Command / pulse state

current_cmd  = "S"

last_cmd_t   = 0.0

_pulse_phase = "F"          # "F" or "turn"

_pulse_t     = 0.0

 

# Network

_sock = None

 

# ═══════════════════════════════════════════════════════════════════════════════

#  NETWORK

# ═══════════════════════════════════════════════════════════════════════════════

 

def _open_socket() -> None:

    global _sock

    try:

        if _sock:

            _sock.close()

    except Exception:

        pass

    _sock = None

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    s.settimeout(0.20)          # short connect timeout – never stall camera loop

    s.connect((HOST, PORT))

    s.settimeout(None)          # sends are tiny; blocking is fine once connected

    _sock = s

 

def _force_send(cmd: str) -> None:

    """Send immediately, bypassing the rate limiter. Use for safety-critical stops."""

    global _sock, current_cmd, last_cmd_t

    if not USE_SOCKETS:

        print(f"[SIM] FORCE {cmd}")

        current_cmd = cmd

        return

    for _ in range(2):

        try:

            if _sock is None:

                _open_socket()

            _sock.sendall(cmd.encode())

            current_cmd = cmd

            last_cmd_t  = time.time()

            return

        except Exception as e:

            print(f"Socket error (force): {e}")

            _sock = None

 

def send_command(cmd: str) -> None:

    """Send cmd if the rate-limit window has elapsed."""

    global _sock, current_cmd, last_cmd_t

    if not USE_SOCKETS:

        print(f"[SIM] {cmd}")

        current_cmd = cmd

        return

    now = time.time()

    if now - last_cmd_t < MIN_CMD_INTERVAL:

        return

    for _ in range(2):

        try:

            if _sock is None:

                _open_socket()

            _sock.sendall(cmd.encode())

            current_cmd = cmd

            last_cmd_t  = now

            return

        except Exception as e:

            print(f"Socket error: {e}")

            _sock = None

 

# ═══════════════════════════════════════════════════════════════════════════════

#  MATHS

# ═══════════════════════════════════════════════════════════════════════════════

 

def circular_mean(angles_deg: deque) -> float:

    """Correct circular (unit-vector) mean — handles 0/360 wrap."""

    s = sum(math.sin(math.radians(a)) for a in angles_deg)

    c = sum(math.cos(math.radians(a)) for a in angles_deg)

    return math.degrees(math.atan2(s, c)) % 360

 

def heading_error(current: float, target: float) -> float:

    """

    Signed shortest-path error in [-180, 180].

    +ve → turn CW / right (screen coords, Y-axis down)

    -ve → turn CCW / left

    """

    return (target - current + 180) % 360 - 180

 

def resample_path(pts: list, spacing: float) -> list:

    """

    Resample a polyline to uniform arc-length spacing.

    Guarantees consistent behaviour regardless of mouse draw speed.

    """

    if len(pts) < 2:

        return list(pts)

    out   = [pts[0]]

    carry = 0.0

    for i in range(1, len(pts)):

        x0, y0 = pts[i - 1]

        x1, y1 = pts[i]

        seg = math.hypot(x1 - x0, y1 - y0)

        if seg == 0:

            continue

        t = (spacing - carry) / seg

        while t <= 1.0:

            out.append((int(x0 + t * (x1 - x0)), int(y0 + t * (y1 - y0))))

            t += spacing / seg

        carry = (t - 1.0) * seg

    return out

 

def lookahead_target(bot_pos: tuple, wps: list, dist: float):

    """

    True distance-based pure-pursuit look-ahead.

    Walks the remaining path from the closest point until `dist` px of arc

    have been covered, then returns that interpolated point.

    Returns the final waypoint if the remaining path is shorter than `dist`.

    """

    if not wps:

        return None

    bx, by = bot_pos

 

    # Closest waypoint index

    best_i, best_d = 0, float('inf')

    for i, (wx, wy) in enumerate(wps):

        d = math.hypot(wx - bx, wy - by)

        if d < best_d:

            best_i, best_d = i, d

 

    # Walk forward until accumulated arc length >= dist

    acc = 0.0

    for i in range(best_i, len(wps) - 1):

        seg = math.hypot(wps[i+1][0]-wps[i][0], wps[i+1][1]-wps[i][1])

        if acc + seg >= dist:

            t  = (dist - acc) / seg

            lx = wps[i][0] + t * (wps[i+1][0] - wps[i][0])

            ly = wps[i][1] + t * (wps[i+1][1] - wps[i][1])

            return int(lx), int(ly)

        acc += seg

 

    return wps[-1]

 

def advance_waypoints(bot_pos: tuple, wps: list, radius: float) -> None:

    """Remove every leading waypoint the bot has already passed."""

    bx, by = bot_pos

    while len(wps) > 1 and math.hypot(wps[0][0]-bx, wps[0][1]-by) < radius:

        wps.pop(0)

 

# ═══════════════════════════════════════════════════════════════════════════════

#  MOUSE CALLBACK  (registered ONCE, before the loop)

# ═══════════════════════════════════════════════════════════════════════════════

 

def mouse_cb(event, x, y, flags, param):

    global _drawing, path_points, waypoints, paused

 

    if event == cv2.EVENT_LBUTTONDOWN:

        _drawing    = True

        path_points = [(x, y)]

 

    elif event == cv2.EVENT_MOUSEMOVE and _drawing:

        lx, ly = path_points[-1]

        if math.hypot(x - lx, y - ly) >= WAYPOINT_SPACING_PX:

            path_points.append((x, y))

 

    elif event == cv2.EVENT_LBUTTONUP and _drawing:

        _drawing = False

        if len(path_points) >= 2:

            path_points = resample_path(path_points, WAYPOINT_SPACING_PX)

            waypoints   = list(path_points)

            paused      = False

            print(f"New path: {len(path_points)} pts — bot starting.")

        else:

            path_points = []

            waypoints   = []

 

    elif event == cv2.EVENT_RBUTTONDOWN:

        _drawing    = False

        path_points = []

        waypoints   = []

        paused      = False

        _force_send("S")

        print("Path cleared — bot stopped.")

 

# ═══════════════════════════════════════════════════════════════════════════════

#  ARUCO DETECTOR  (sub-pixel refinement enabled)

# ═══════════════════════════════════════════════════════════════════════════════

 

_params = aruco.DetectorParameters()

_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

detector = aruco.ArucoDetector(

    aruco.getPredefinedDictionary(aruco.DICT_4X4_50), _params

)

 

# ═══════════════════════════════════════════════════════════════════════════════

#  CAPTURE

# ═══════════════════════════════════════════════════════════════════════════════

 

cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # always grab the freshest frame

 

cv2.namedWindow("Arena Control")

cv2.setMouseCallback("Arena Control", mouse_cb)   # ← registered ONCE here

 

print("=" * 54)

print("  MONA Freehand Path Follower")

print("  Drag          → draw path")

print("  Right-click   → clear & stop")

print("  SPACE         → pause / resume")

print("  Q             → quit")

print("=" * 54)

 

# ═══════════════════════════════════════════════════════════════════════════════

#  MAIN LOOP

# ═══════════════════════════════════════════════════════════════════════════════

 

while True:

    ret, frame = cap.read()

    if not ret:

        print("Camera read failed.")

        break

 

    now  = time.time()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = detector.detectMarkers(gray)

 

    # ── Draw stored path ───────────────────────────────────────────────────────

    if len(path_points) >= 2:

        pts_np = np.array(path_points, dtype=np.int32)

        cv2.polylines(frame, [pts_np], False, (255, 200, 0), 1)

        cv2.circle(frame, path_points[0],  6, (0, 220, 0),   -1)   # start (green)

        cv2.circle(frame, path_points[-1], 6, (0, 60, 255),  -1)   # end   (red)

 

    # ── Draw remaining waypoints (sparse overlay) ──────────────────────────────

    if waypoints:

        step = max(1, len(waypoints) // 40)

        for wp in waypoints[::step]:

            cv2.circle(frame, wp, 2, (0, 200, 255), -1)

 

    # ── Live drawing preview ───────────────────────────────────────────────────

    if _drawing and len(path_points) >= 2:

        pts_np = np.array(path_points, dtype=np.int32)

        cv2.polylines(frame, [pts_np], False, (80, 255, 80), 2)

 

    # ── Detect bot ────────────────────────────────────────────────────────────

    bot_visible = False

 

    if ids is not None:

        aruco.drawDetectedMarkers(frame, corners, ids)

 

        for i, mid in enumerate(ids.flatten()):

            if mid != BOT_ID:

                continue

 

            c = corners[i][0]           # shape (4, 2): TL, TR, BR, BL

            tl, tr, br, bl = c

 

            # Centroid from all 4 corners (more stable than diagonal pair)

            raw_cx = float(np.mean(c[:, 0]))

            raw_cy = float(np.mean(c[:, 1]))

 

            # Heading: centre → top-edge midpoint

            front_x = (tl[0] + tr[0]) / 2

            front_y = (tl[1] + tr[1]) / 2

            raw_angle = math.degrees(

                math.atan2(front_y - raw_cy, front_x - raw_cx)

            ) % 360

 

            # Accumulate into smoothing buffers

            pos_buf.append((raw_cx, raw_cy))

            angle_buf.append(raw_angle)

 

            # Smoothed pose

            scx   = sum(p[0] for p in pos_buf) / len(pos_buf)

            scy   = sum(p[1] for p in pos_buf) / len(pos_buf)

            angle = circular_mean(angle_buf)

            bot_pos = (scx, scy)

            bot_visible = True

 

            # Draw pose

            alen = 34

            ax = int(scx + alen * math.cos(math.radians(angle)))

            ay = int(scy + alen * math.sin(math.radians(angle)))

            cv2.circle(frame, (int(scx), int(scy)), 5, (0, 0, 220), -1)

            cv2.arrowedLine(frame, (int(scx), int(scy)), (ax, ay),

                            (0, 230, 0), 2, tipLength=0.3)

 

            # ── Navigation ────────────────────────────────────────────────────

            desired = "S"

 

            if waypoints and not paused:

                advance_waypoints(bot_pos, waypoints, ARRIVAL_PX)

 

                if not waypoints:

                    print("Path complete.")

                    desired = "S"

                else:

                    final_dist = math.hypot(

                        waypoints[-1][0] - scx, waypoints[-1][1] - scy

                    )

                    if len(waypoints) == 1 and final_dist < ARRIVAL_PX:

                        print("Arrived.")

                        waypoints = []

                        desired   = "S"

                    else:

                        # True distance-based pure-pursuit

                        target = lookahead_target(bot_pos, waypoints, LOOKAHEAD_PX)

                        if target:

                            tx, ty     = target

                            t_angle    = math.degrees(

                                math.atan2(ty - scy, tx - scx)

                            ) % 360

                            err        = heading_error(angle, t_angle)

                            abs_err    = abs(err)

                            turn_dir   = "R" if err > 0 else "L"

 

                            # Draw look-ahead target

                            cv2.circle(frame, (int(tx), int(ty)), 6, (255, 80, 0), -1)

                            cv2.line(frame, (int(scx), int(scy)),

                                     (int(tx), int(ty)), (255, 140, 0), 1)

 

                            if abs_err <= ANGLE_TOLERANCE:

                                # On-course: go straight

                                desired      = "F"

                                _pulse_phase = "F"

 

                            elif abs_err <= TURN_THRESHOLD:

                                # Pulse-width curve correction

                                phase_elapsed = now - _pulse_t

                                turn_t = PULSE_CYCLE_TIME * TURN_PULSE_RATIO

                                fwd_t  = PULSE_CYCLE_TIME * (1.0 - TURN_PULSE_RATIO)

 

                                if _pulse_phase == "F":

                                    if phase_elapsed >= fwd_t:

                                        _pulse_phase = "turn"

                                        _pulse_t     = now

                                    desired = "F"

                                else:

                                    if phase_elapsed >= turn_t:

                                        _pulse_phase = "F"

                                        _pulse_t     = now

                                    desired = turn_dir

 

                            else:

                                # Large error: spin in place

                                desired      = turn_dir

                                _pulse_phase = "turn"

                                _pulse_t     = now

 

                            cv2.putText(

                                frame, f"err={err:+.1f}  {desired}",

                                (int(scx) + 10, int(scy) + 54),

                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 255), 1,

                            )

 

            # ── Send command ──────────────────────────────────────────────────

            if desired != current_cmd:

                if desired == "S":

                    _force_send("S")                    # stops bypass rate-limit

                else:

                    send_command(desired)

            elif now - last_cmd_t > WATCHDOG_INTERVAL:

                send_command(current_cmd)               # keepalive

 

            # HUD

            label = "PAUSED" if paused else f"CMD: {current_cmd}"

            cv2.putText(frame, label,

                        (int(scx) + 10, int(scy) + 28),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 80), 2)

            break   # only handle first matching marker per frame

 

    # ── Safety stop if bot disappears mid-run ─────────────────────────────────

    if not bot_visible and current_cmd != "S" and not paused:

        print("Bot lost — stopping.")

        _force_send("S")

 

    # ── HUD ───────────────────────────────────────────────────────────────────

    h, w = frame.shape[:2]

    cv2.putText(frame,

                f"Path: {len(path_points)} pts  |  Remaining: {len(waypoints)}",

                (10, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (210, 210, 210), 1)

    cv2.putText(frame,

                "Drag=draw  RClick=clear  Space=pause  Q=quit",

                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (210, 210, 210), 1)

    if paused:

        cv2.putText(frame, "-- PAUSED --",

                    (w // 2 - 90, 44),

                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 60, 255), 2)

 

    cv2.imshow("Arena Control", frame)

 

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):

        _force_send("S")

        break

    elif key == ord(' '):

        paused = not paused

        if paused:

            _force_send("S")

            print("Paused.")

        else:

            print("Resumed.")

 

# ═══════════════════════════════════════════════════════════════════════════════

#  CLEANUP

# ═══════════════════════════════════════════════════════════════════════════════

 

cap.release()

cv2.destroyAllWindows()

if _sock:

    try:

        _sock.close()

    except Exception:

        pass

 

 