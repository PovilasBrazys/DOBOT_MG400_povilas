import threading
import sys
import os
import cv2
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from time import sleep
from robot_control_001 import ConnectRobot, RunPoint, WaitArrive, GetFeed, ClearRobotError, true_camPointXY

setZ = 0
cameraX_offset = 48
min_area = 500  # Initial value for color detection

def on_trackbar(val):
    """Callback function for the trackbar"""
    global min_area
    min_area = val

def camera_thread(dashboard, move, camera_ready_event, stop_event, shared_state):
    global min_area
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return

    window_name = "Camera Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.createTrackbar("Min Area", window_name, min_area, 1000, on_trackbar)

    color_ranges = {
        'blue': ([110, 50, 50], [130, 255, 255]),
        'green': ([40, 40, 40], [80, 255, 255]),
        'red': ([0, 50, 50], [10, 255, 255]),
        'yellow': ([20, 100, 100], [40, 255, 255])
    }

    camera_ready_event.set()
    closest_color = ""  # Local variable for current frame's color

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width, _ = frame.shape
        center_x = width // 2
        center_y = height // 2

        closest_contour = None
        min_distance = float('inf')
        closest_color = "none"  # Default if no color detected

        for color, (lower, upper) in color_ranges.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
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

                    # Drawing logic remains the same...

        # Update shared color state with lock
        with shared_state['color_lock']:
            shared_state['current_color'] = closest_color

        # Store color if capture requested
        if shared_state['capture_requested'].is_set():
            with shared_state['grid_lock']:
                x, y = shared_state['current_grid_pos']
                if 0 <= x < 3 and 0 <= y < 3:
                    shared_state['color_grid'][y][x] = closest_color
                    print(f"Stored color: {closest_color} at position ({x}, {y})")
            shared_state['capture_requested'].clear()
            shared_state['capture_complete'].set()

        # Display logic remains the same...
        b, g, r = frame[center_y, center_x]
        rgb_text = f"RGB: ({r}, {g}, {b})"
        color_text = f"Color: {closest_color}"
        
        cv2.putText(frame, rgb_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, color_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            stop_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()

def robot_movement_thread(dashboard, move, camera_ready_event, stop_event, shared_state):
    camera_ready_event.wait()
    dashboard.SpeedFactor(10)
    
    start_point = [370 - cameraX_offset, 33, setZ, 0]
    x_step = -35
    y_step = -35
    grid_size = 3
    grab_z = -143

    # First pass: Scan the grid and build color map
    for y in range(grid_size):
        for x in range(grid_size):
            if stop_event.is_set():
                return
                
            current_x = start_point[0] + (x * x_step)
            current_y = start_point[1] + (y * y_step)
            target_point = true_camPointXY([current_x, current_y, setZ, 0], cameraX_offset)
            
            move.MovL(target_point[0], target_point[1], target_point[2], target_point[3])
            WaitArrive(target_point)
            print(f"Arrived at position ({x}, {y})")
            
            # Set grid position and request capture
            with shared_state['grid_lock']:
                shared_state['current_grid_pos'] = (x, y)
            
            shared_state['capture_requested'].set()
            if not shared_state['capture_complete'].wait(timeout=5.0):
                print(f"Timeout waiting for color capture at ({x}, {y})")
            shared_state['capture_complete'].clear()
            sleep(0.5)  # Short stabilization delay

    # Print final color grid
    print("\nFinal Color Grid Results:")
    with shared_state['grid_lock']:
        for i, row in enumerate(shared_state['color_grid']):
            print(f"Row {i}: {row}")
    
    # Second pass: Visit all red objects
    print("\nMoving to red objects...")
    with shared_state['grid_lock']:
        for y in range(grid_size):
            for x in range(grid_size):
                if shared_state['color_grid'][y][x] == 'red':
                    print(f"Found red object at ({x}, {y}) - moving to position")
                    
                    # Calculate target position
                    current_x = start_point[0] + (x * x_step)
                    current_y = start_point[1] + (y * y_step)
                    target_point = true_camPointXY([current_x, current_y, setZ, 0], cameraX_offset)

                    # Move to position above the red object
                    move.MovL(target_point[0], target_point[1], target_point[2], target_point[3])
                    WaitArrive(target_point)
                    print(f"Now above red object at ({x}, {y})")
                    sleep(2)  # Pause for 2 seconds at each red object

                    # Move to position
                    target_grab_point = [current_x + cameraX_offset, current_y, setZ, 0]
                    move.MovL(target_grab_point[0], target_grab_point[1], setZ, target_grab_point[3])
                    WaitArrive(target_grab_point)
                    
                    print(f"Position ({x},{y}):", dashboard.GetPose())
                    
                    target_grab_point = [current_x + cameraX_offset, current_y, grab_z, 0]
                    move.MovL(target_grab_point[0], target_grab_point[1], target_grab_point[2], target_grab_point[3])
                    WaitArrive(target_grab_point)
                    dashboard.DO(1, 1)

                    sleep(1)
                    target_grab_point = [current_x + cameraX_offset, current_y, setZ, 0]
                    move.MovL(target_grab_point[0], target_grab_point[1], target_grab_point[2], target_grab_point[3])
                    WaitArrive(target_grab_point)

                    target_grab_point = [current_x + cameraX_offset, current_y, grab_z, 0]
                    move.MovL(target_grab_point[0], target_grab_point[1], target_grab_point[2], target_grab_point[3])
                    WaitArrive(target_grab_point)
                    dashboard.DO(1, 0)
                    dashboard.DO(2, 1)
                    
                    sleep(1)
                    dashboard.DO(2, 0)
                    target_grab_point = [current_x + cameraX_offset, current_y, setZ, 0]
                    move.MovL(target_grab_point[0], target_grab_point[1], target_grab_point[2], target_grab_point[3])
                    WaitArrive(target_grab_point)
                    

                    
    
    print("\nCompleted visiting all red objects")
    stop_event.set()  # Signal program completion

if __name__ == '__main__':
    dashboard, move, feed = ConnectRobot()
    dashboard.EnableRobot()
    
    point_a = [335 - 53, 33, 0, 0]
    point_b = true_camPointXY([point_a[0], point_a[1], point_a[2], point_a[3]], cameraX_offset)
    move.MovL(point_b[0], point_b[1], point_b[2], point_b[3])

    # Shared state between threads
    shared_state = {
        'color_grid': [['' for _ in range(3)] for _ in range(3)],  # 3x3 grid
        'current_grid_pos': (-1, -1),  # (x, y) position
        'current_color': "none",
        'grid_lock': threading.Lock(),
        'color_lock': threading.Lock(),
        'capture_requested': threading.Event(),
        'capture_complete': threading.Event()
    }

    # Create synchronization events
    camera_ready_event = threading.Event()
    stop_event = threading.Event()

    # Start support threads
    feed_thread = threading.Thread(target=GetFeed, args=(feed,))
    feed_thread.daemon = True
    feed_thread.start()

    error_thread = threading.Thread(target=ClearRobotError, args=(dashboard,))
    error_thread.daemon = True
    error_thread.start()

    # Start main threads
    camera_thread = threading.Thread(
        target=camera_thread,
        args=(dashboard, move, camera_ready_event, stop_event, shared_state)
    )
    camera_thread.daemon = True
    camera_thread.start()

    robot_thread = threading.Thread(
        target=robot_movement_thread,
        args=(dashboard, move, camera_ready_event, stop_event, shared_state)
    )
    robot_thread.daemon = True
    robot_thread.start()

    # Main thread management
    try:
        while not stop_event.is_set():
            sleep(0.5)
    except KeyboardInterrupt:
        print("Program interrupted by user")
        stop_event.set()
    
    # Clean shutdown
    print("Shutting down...")
    dashboard.DisableRobot()
    cv2.destroyAllWindows()
    sys.exit(0)