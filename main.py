import cv2
import cv2.aruco as aruco
import numpy as np
import math
import socket
import time

# ------------------------ CONFIGURATION ------------------------
BOT_ID = 3  
HOST = "192.168.0.101"
PORT = 80
USE_SOCKETS = True

waypoints = []
current_bot_state = "S" # Tracks Bot State (Forwards, Backwards, Left and Right)

# ------------------------ NETWORK CONFIGURATION ------------------------
def send_command(cmd):
    """Sends the simple char command to the ESP32"""
    if not USE_SOCKETS:
        print(f"[SIMULATED] Sending: {cmd}")
        return

    # Bot Control
    for _ in range(3):
        try:
            s = socket.socket()
            s.settimeout(0.5) # Don't hang forever if bot disconnects
            s.connect((HOST, PORT))
            s.send(cmd.encode())
            s.close()
            time.sleep(0.01) # Reduced sleep to prevent camera lag
        except Exception as e:
            print("Socket Error:", e)

# ------------ DRAWING PATH ------------
def mouse_click(event, x, y, flags, param):
    global waypoints
    if event == cv2.EVENT_LBUTTONDOWN:
        waypoints.append((x, y))
        print(f"Added waypoint: ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN:
        waypoints.clear()
        send_command("S") # Send a stop command if path is cleared
        print("Path cleared! Bot stopped.")

# ------------ CONTROL LOOP ------------
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

cap = cv2.VideoCapture(1)

cv2.namedWindow("Automated Arena Control")
cv2.setMouseCallback("Automated Arena Control", mouse_click)

print("Starting Automated Control.")
print("LEFT CLICK to add waypoints. RIGHT CLICK to clear path.")

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    # 1. Draw Checkpoints
    for i in range(len(waypoints)):
        cv2.circle(frame, waypoints[i], 8, (255, 255, 0), -1)
        cv2.putText(frame, str(i+1), (waypoints[i][0] + 10, waypoints[i][1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        if i > 0:
            cv2.line(frame, waypoints[i-1], waypoints[i], (255, 255, 0), 2)

    # 2. Navigation
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)

        for i in range(len(ids)):
            if ids[i][0] == BOT_ID:
                marker_corners = corners[i][0]
                topLeft, topRight, bottomRight, bottomLeft = marker_corners

                cx = int((topLeft[0] + bottomRight[0]) / 2.0)
                cy = int((topLeft[1] + bottomRight[1]) / 2.0)
                front_x = (topLeft[0] + topRight[0]) / 2.0
                front_y = (topLeft[1] + topRight[1]) / 2.0

                # Current Heading
                angle_rad = math.atan2(cy - front_y, front_x - cx) 
                angle_deg = math.degrees(angle_rad)
                if angle_deg < 0: angle_deg += 360

                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.line(frame, (cx, cy), (int(front_x), int(front_y)), (0, 255, 0), 2)

                # 3. Determine the required action
                desired_state = "S" # Default to stop

                if len(waypoints) > 0:
                    target_x, target_y = waypoints[0]
                    cv2.line(frame, (cx, cy), (target_x, target_y), (0, 255, 255), 1)

                    distance = math.sqrt((target_x - cx)**2 + (target_y - cy)**2)
                    
                    if distance < 40: # REACHED WAYPOINT
                        print(f"Reached waypoint: {waypoints[0]}")
                        waypoints.pop(0)
                        desired_state = "S" # Stop briefly when a point is reached
                    else:
                        # Calculate Heading Error
                        target_angle_rad = math.atan2(cy - target_y, target_x - cx)
                        target_angle_deg = math.degrees(target_angle_rad)
                        if target_angle_deg < 0: target_angle_deg += 360

                        heading_error = target_angle_deg - angle_deg
                        if heading_error > 180: heading_error -= 360
                        elif heading_error < -180: heading_error += 360

                        # State Machine Logic
                        angle_tolerance = 15 
                        
                        if heading_error > angle_tolerance:
                            desired_state = "L"
                        elif heading_error < -angle_tolerance:
                            desired_state = "R"
                        else:
                            desired_state = "F"

                # 4. Only send command over Wi-Fi if the state changed!
                if desired_state != current_bot_state:
                    print(f"State Change: {current_bot_state} -> {desired_state}")
                    send_command(desired_state)
                    current_bot_state = desired_state

                # Display Current State on screen
                cv2.putText(frame, f"STATE: {current_bot_state}", (cx + 10, cy + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Automated Arena Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        send_command("S") # Ensure bot stops when quitting
        break

cap.release()
cv2.destroyAllWindows()