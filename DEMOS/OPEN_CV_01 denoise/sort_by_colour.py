import threading
import sys
import os
import cv2
import numpy as np
import keyboard  # Added for keyboard input detection
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

        # Only process color detection during scanning phase
        if not shared_state['scanning_complete'].is_set():
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

        # Display logic
        height, width, _ = frame.shape
        center_x = width // 2
        center_y = height // 2
        
        b, g, r = frame[center_y, center_x]
        rgb_text = f"RGB: ({r}, {g}, {b})"
        
        # Show current color only during scanning
        if not shared_state['scanning_complete'].is_set():
            color_text = f"Color: {closest_color}"
        else:
            color_text = "Scanning complete - Sorting in progress"
        
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
    grab_z = -148  # Lowered by 5 from original -143

    # First pass: Scan the grid and build color map
    print("Starting initial grid scan...")
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
            if x == 0 and y == 0:
                keyboard.wait('s')

            # Set grid position and request capture
            with shared_state['grid_lock']:
                shared_state['current_grid_pos'] = (x, y)
            
            shared_state['capture_requested'].set()
            if not shared_state['capture_complete'].wait(timeout=5.0):
                print(f"Timeout waiting for color capture at ({x}, {y})")
            shared_state['capture_complete'].clear()
            sleep(0.5)  # Short stabilization delay

    # Mark scanning as complete - no more color detection
    shared_state['scanning_complete'].set()
    print("Initial scan complete - color detection disabled")

    # Print final color grid
    print("\nFinal Color Grid Results:")
    with shared_state['grid_lock']:
        for i, row in enumerate(shared_state['color_grid']):
            print(f"Row {i}: {row}")

    # Build lists of positions per color
    color_positions = {'red': [], 'blue': [], 'green': []}
    with shared_state['grid_lock']:
        for y in range(grid_size):
            for x in range(grid_size):
                c = shared_state['color_grid'][y][x]
                if c in color_positions:
                    color_positions[c].append((x, y))

    print(f"\nColor positions: {color_positions}")

    # Define target rows for sorting
    target_rows = {
        'blue': 0,
        'red': 1,
        'green': 2
    }

    # Create a copy of the color grid to track current state
    current_grid = [row[:] for row in shared_state['color_grid']]
    
    # Temporary storage area (outside grid in y direction)
    temp_storage_base_x = start_point[0]
    temp_storage_y = start_point[1] + 150  # 150mm further in y direction
    temp_objects = []  # List of (color, temp_index) for objects in temp storage

    def move_object(from_x, from_y, to_x, to_y, is_temp_storage=False):
        """Move an object from (from_x, from_y) to (to_x, to_y)"""
        # Calculate source position
        source_x = start_point[0] + (from_x * x_step)
        source_y = start_point[1] + (from_y * y_step)
        
        # Calculate target position
        if is_temp_storage:
            target_x = temp_storage_base_x + (to_x * x_step)
            target_y = temp_storage_y
        else:
            target_x = start_point[0] + (to_x * x_step)
            target_y = start_point[1] + (to_y * y_step)

        # Move above source
        move.MovL(source_x + cameraX_offset, source_y, setZ, 0)
        WaitArrive([source_x + cameraX_offset, source_y, setZ, 0])
        
        # Move down to grab
        move.MovL(source_x + cameraX_offset, source_y, grab_z, 0)
        WaitArrive([source_x + cameraX_offset, source_y, grab_z, 0])
        
        # Turn on vacuum for 0.5 seconds
        dashboard.DO(1, 1)
        sleep(0.5)
        
        # Move up
        move.MovL(source_x + cameraX_offset, source_y, setZ, 0)
        WaitArrive([source_x + cameraX_offset, source_y, setZ, 0])
        
        # Move to target position
        move.MovL(target_x + cameraX_offset, target_y, setZ, 0)
        WaitArrive([target_x + cameraX_offset, target_y, setZ, 0])
        
        # Move down to place
        move.MovL(target_x + cameraX_offset, target_y, grab_z, 0)
        WaitArrive([target_x + cameraX_offset, target_y, grab_z, 0])
        
        # Turn off vacuum and turn on pressure for 0.5 seconds
        dashboard.DO(1, 0)
        dashboard.DO(2, 1)
        sleep(0.5)
        dashboard.DO(2, 0)
        
        # Move up
        move.MovL(target_x + cameraX_offset, target_y, setZ, 0)
        WaitArrive([target_x + cameraX_offset, target_y, setZ, 0])

    def move_from_temp_storage(temp_index, to_x, to_y):
        """Move object from temporary storage to target position"""
        # Source is in temp storage
        source_x = temp_storage_base_x + (temp_index * x_step)
        source_y = temp_storage_y
        
        # Target position
        target_x = start_point[0] + (to_x * x_step)
        target_y = start_point[1] + (to_y * y_step)

        # Move above temp storage
        move.MovL(source_x + cameraX_offset, source_y, setZ, 0)
        WaitArrive([source_x + cameraX_offset, source_y, setZ, 0])
        
        # Move down to grab
        move.MovL(source_x + cameraX_offset, source_y, grab_z, 0)
        WaitArrive([source_x + cameraX_offset, source_y, grab_z, 0])
        
        # Turn on vacuum for 0.5 seconds
        dashboard.DO(1, 1)
        sleep(0.5)
        
        # Move up
        move.MovL(source_x + cameraX_offset, source_y, setZ, 0)
        WaitArrive([source_x + cameraX_offset, source_y, setZ, 0])
        
        # Move to target position
        move.MovL(target_x + cameraX_offset, target_y, setZ, 0)
        WaitArrive([target_x + cameraX_offset, target_y, setZ, 0])
        
        # Move down to place
        move.MovL(target_x + cameraX_offset, target_y, grab_z, 0)
        WaitArrive([target_x + cameraX_offset, target_y, grab_z, 0])
        
        # Turn off vacuum and turn on pressure for 0.5 seconds
        dashboard.DO(1, 0)
        dashboard.DO(2, 1)
        sleep(0.5)
        dashboard.DO(2, 0)
        
        # Move up
        move.MovL(target_x + cameraX_offset, target_y, setZ, 0)
        WaitArrive([target_x + cameraX_offset, target_y, setZ, 0])

    # Sort objects by color into rows
    print("\nStarting color sorting...")
    
    # Process each color in order: blue to row 0, red to row 1, green to row 2
    for color in ['blue', 'red', 'green']:
        if stop_event.is_set():
            return
            
        target_row = target_rows[color]
        positions = color_positions[color]
        
        print(f"\nProcessing {color} objects for row {target_row}...")
        
        # Sort objects of this color into the target row
        for obj_x, obj_y in positions:
            if stop_event.is_set():
                return
                
            # Check if object is already in correct row
            if obj_y == target_row:
                print(f"{color} object at ({obj_x}, {obj_y}) already in correct row")
                continue
                
            # Find the leftmost free slot in target row
            target_slot = None
            for slot_x in range(3):
                if current_grid[target_row][slot_x] == '':  # Empty slot
                    target_slot = slot_x
                    break
            
            if target_slot is not None:
                # Check if target slot is already occupied by wrong color
                if current_grid[target_row][target_slot] != '' and current_grid[target_row][target_slot] != color:
                    # Need to move the occupying object to temp storage first
                    occupying_color = current_grid[target_row][target_slot]
                    print(f"Moving {occupying_color} object from ({target_slot}, {target_row}) to temp storage")
                    move_object(target_slot, target_row, len(temp_objects), 0, is_temp_storage=True)
                    temp_objects.append((occupying_color, len(temp_objects)))
                    current_grid[target_row][target_slot] = ''  # Mark as empty
                
                # Now move the current object to the target slot
                print(f"Moving {color} object from ({obj_x}, {obj_y}) to ({target_slot}, {target_row})")
                move_object(obj_x, obj_y, target_slot, target_row)
                
                # Update current grid state
                current_grid[obj_y][obj_x] = ''  # Source becomes empty
                current_grid[target_row][target_slot] = color  # Target now has this color
                
            else:
                # No free slot in target row, move to temporary storage
                print(f"No free slot in row {target_row}, moving {color} object to temp storage")
                move_object(obj_x, obj_y, len(temp_objects), 0, is_temp_storage=True)
                temp_objects.append((color, len(temp_objects)))
                current_grid[obj_y][obj_x] = ''  # Mark source as empty

    # Now move objects from temporary storage to their final positions
    print("\nMoving objects from temporary storage to final positions...")
    
    for temp_color, temp_index in temp_objects:
        if stop_event.is_set():
            return
            
        target_row = target_rows[temp_color]
        
        # Find free slot in target row
        target_slot = None
        for slot_x in range(3):
            if current_grid[target_row][slot_x] == '':  # Empty slot
                target_slot = slot_x
                break
        
        if target_slot is not None:
            print(f"Moving {temp_color} object from temp storage to ({target_slot}, {target_row})")
            move_from_temp_storage(temp_index, target_slot, target_row)
            current_grid[target_row][target_slot] = temp_color
        else:
            print(f"Warning: No free slot found for {temp_color} object in temp storage")

    print("\nFinal sorted grid:")
    for i, row in enumerate(current_grid):
        print(f"Row {i}: {row}")

    print("\nColor sorting complete!")
    stop_event.set()

if __name__ == '__main__':
    dashboard, move, feed = ConnectRobot()
    dashboard.EnableRobot()
    
    # Initialize outputs
    dashboard.DO(1, 0)
    dashboard.DO(2, 0)
                    
    point_a = [335 - 53, 33, 0, 0]
    point_b = true_camPointXY([point_a[0], point_a[1], point_a[2], point_a[3]], cameraX_offset)
    move.MovL(point_b[0], point_b[1], point_b[2], point_b[3])

    # Shared state between threads
    shared_state = {
        'color_grid': [['' for _ in range(3)] for _ in range(3)],  # 3x3 grid
        'current_grid_pos': (-1, -1),  # (x, y) position
        'current_color': "none",
        'scanning_complete': threading.Event(),  # Flag to stop color detection after scan
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