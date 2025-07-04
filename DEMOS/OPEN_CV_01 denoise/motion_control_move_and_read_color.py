import threading
import sys
import os
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from time import sleep
from robot_control_001 import ConnectRobot, RunPoint, WaitArrive, GetFeed, ClearRobotError, true_camPointXY


setZ = 0
cameraX_offset = 53



import numpy as np

# Global variable for minimum area
min_area = 500  # Initial value

def on_trackbar(val):
    """Callback function for the trackbar"""
    global min_area
    min_area = val

def camera_thread(dashboard, move, camera_ready_event, stop_event):
    global min_area
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return

    window_name = "Camera Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Create a trackbar for area filtering
    cv2.createTrackbar("Min Area", window_name, min_area, 1000, on_trackbar)

    # Define color ranges (HSV) - Adjusted ranges
    color_ranges = {
        'blue': ([110, 50, 50], [130, 255, 255]),        # Blue
        'green': ([40, 40, 40], [80, 255, 255]),         # Green
        'red': ([0, 50, 50], [10, 255, 255]),           # Red
        'yellow': ([20, 100, 100], [40, 255, 255])      # Yellow
    }

    # Signal that camera is ready
    camera_ready_event.set()

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get frame dimensions and center pixel
        height, width, _ = frame.shape
        center_x = width // 2
        center_y = height // 2

        closest_contour = None
        min_distance = float('inf')
        closest_color = ""

        for color, (lower, upper) in color_ranges.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)

            # Create a mask
            mask = cv2.inRange(hsv, lower, upper)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Calculate the area of the contour
                area = cv2.contourArea(contour)

                # Filter contours based on area
                if area > min_area:
                    # Approximate the contour
                    epsilon = 0.04 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)

                    x, y, w, h = cv2.boundingRect(approx)
                    contour_center_x = x + w // 2
                    contour_center_y = y + h // 2
                    distance = ((contour_center_x - center_x) ** 2 + (contour_center_y - center_y) ** 2) ** 0.5

                    if distance < min_distance:
                        min_distance = distance
                        closest_contour = approx
                        closest_color = color

                    if color == 'yellow':
                        # Draw a square for yellow
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
                    elif color == 'red':
                        # Draw a circle for red
                        radius = min(w, h) // 2
                        cv2.circle(frame, (x + w // 2, y + h // 2), radius, (0, 0, 0), 2)
                    elif color == 'blue':
                        # Draw an isosceles triangle for blue
                        triangle_points = np.array([
                            [x + w // 2, y],
                            [x, y + h],
                            [x + w, y + h]
                        ], np.int32)
                        cv2.polylines(frame, [triangle_points], True, (0, 0, 0), 2)
                    elif color == 'green':
                        # Draw an isosceles pentagon for green
                        pentagon_points = np.array([
                            [x + w // 2, y],
                            [x, y + h // 3],
                            [x + w // 4, y + h],
                            [x + 3 * w // 4, y + h],
                            [x + w, y + h // 3]
                        ], np.int32)
                        cv2.polylines(frame, [pentagon_points], True, (0, 0, 0), 2)


        # Read BGR values at center pixel and convert to RGB
        b, g, r = frame[center_y, center_x]  # OpenCV uses BGR format
        rgb_text = f"RGB: ({r}, {g}, {b})"
        color = f"Color: {closest_color}"  # Display as RGB

        # Overlay RGB values on the frame
        cv2.putText(
            frame,
            rgb_text,
            (10, 30),  # Position near top-left corner
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,  # Font scale
            (255, 255, 255),  # White text
            2,  # Thickness
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            color,
            (10, 60),  # Position near top-left corner
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,  # Font scale
            (255, 255, 255),  # White text
            2,  # Thickness
            cv2.LINE_AA
        )
        # Draw red dot at center
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)




        # Display the frame
        cv2.imshow(window_name, frame)

        # Check for spacebar to stop
        if cv2.waitKey(1) & 0xFF == ord(' '):
            stop_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()
def robot_movement_thread(dashboard, move, camera_ready_event, stop_event):
    # Wait for camera to be ready
    camera_ready_event.wait()

    point_a = [347, 0, setZ, 0]
    dashboard.SpeedFactor(10)

    point_b = true_camPointXY([point_a[0], 50, setZ, 0], cameraX_offset)
    point_c = true_camPointXY([point_a[0], -50, setZ, 0], cameraX_offset)

    # Starting point
    start_point = [370 - cameraX_offset, 33, setZ, 0]
    dashboard.SpeedFactor(10)

    # Define 3x3 grid parameters
    x_step = -35  # Move 25mm in negative X direction
    y_step = -35   # Move 25mm in positive Y direction
    grid_size = 3

    while not stop_event.is_set():
        # Iterate through 3x3 grid
        for y in range(grid_size):
            for x in range(grid_size):
                # Calculate new position
                current_x = start_point[0] + (x * x_step)
                current_y = start_point[1] + (y * y_step)
                
                # Apply camera offset
                target_point = true_camPointXY([current_x, current_y, setZ, 0], cameraX_offset)
                
                # Move to position
                move.MovL(target_point[0], target_point[1], target_point[2], target_point[3])
                WaitArrive(target_point)
                print(f"Position ({x},{y}):", dashboard.GetPose())
                sleep(3)

                if stop_event.is_set():
                    return

if __name__ == '__main__':
    dashboard, move, feed = ConnectRobot()
    dashboard.EnableRobot()
    
    point_a = [335 - 53, 33, 0, 0]
    
    point_b = true_camPointXY([point_a[0], point_a[1], point_a[2], point_a[3]], cameraX_offset)
    move.MovL(point_b[0], point_b[1], point_b[2], point_b[3])

    # Create events for camera readiness and program termination
    camera_ready_event = threading.Event()
    stop_event = threading.Event()

    # Start feed and error clearing threads
    feed_thread = threading.Thread(target=GetFeed, args=(feed,))
    feed_thread.daemon = True
    feed_thread.start()

    feed_thread1 = threading.Thread(target=ClearRobotError, args=(dashboard,))
    feed_thread1.daemon = True
    feed_thread1.start()

    # Start camera thread
    camera_thread1 = threading.Thread(target=camera_thread, args=(dashboard, move, camera_ready_event, stop_event))
    camera_thread1.daemon = True
    camera_thread1.start()

    # Start robot movement thread
    robot_thread = threading.Thread(target=robot_movement_thread, args=(dashboard, move, camera_ready_event, stop_event))
    robot_thread.daemon = True
    robot_thread.start()

    # Keep main thread running
    try:
        while not stop_event.is_set():
            sleep(1)
    except KeyboardInterrupt:
        print("Program terminated by user")
        stop_event.set()
    
    # Ensure clean shutdown
    dashboard.DisableRobot()
    cv2.destroyAllWindows()
    sys.exit(0)