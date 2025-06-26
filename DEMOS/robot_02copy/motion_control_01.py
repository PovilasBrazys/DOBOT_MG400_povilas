import threading
import sys
import os
import cv2
import numpy as np
from datetime import datetime
from time import sleep, time
import re
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from robot_control_module import ConnectRobot, RunPoint, WaitArrive, GetFeed, ClearRobotError, current_actual, globalLockValue

def parse_pose_string(pose_str):
    """Parses the robot pose string into a list of floats."""
    try:
        # Extract numbers using regex, handling potential signs and decimals
        numbers = [float(n) for n in re.findall(r"[-+]?\d*\.?\d+", pose_str)]
        return numbers
    except ValueError as e:
        print(f"Error parsing pose string '{pose_str}': {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during pose parsing: {e}")
        return None

def generate_grid_path(center_x, center_y, size_x, size_y, stepX, stepY):
    """Generate a grid path for image capture."""
    path = []
    # Iterate through y first to create rows, then x for columns
    for y in np.arange(center_y - size_y / 2, center_y + size_y / 2 + stepY, stepY):
        # Alternate direction for x to create a snake-like path
        x_range = np.arange(center_x - size_x / 2, center_x + size_x / 2 + stepX, stepX)
        if len(path) % 2 == 0: # If it's an even row (0-indexed), reverse x direction
            x_range = x_range[::-1]
        for x in x_range:
            # Keep Z and R constant for photo capture height and tool orientation
            path.append([x, y, 0, 0]) # Example Z and R, adjust as needed for your setup
    return path

def capture_and_save(cam, output_dir, img_counter, pose, poses_file):
    """Capture an image and save it with robot pose data."""
    sleep(0.5) # Give camera time to adjust
    ret, frame = cam.read()
    if not ret:
        print("Failed to capture image")
        return img_counter

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] # Added milliseconds
    img_name = os.path.join(output_dir, f"image{img_counter + 1:03d}.jpg")
    cv2.imwrite(img_name, frame)

    # Append pose data to CSV file immediately
    poses_file.write(f"{img_name},{pose[0]:.2f},{pose[1]:.2f},{pose[2]:.2f},{pose[3]:.2f},{timestamp}\n")
    poses_file.flush() # Ensure data is written to disk

    print(f"Saved: {img_name} with pose {[f'{p:.2f}' for p in pose]}")
    return img_counter + 1

def robot_motion(move, path, output_dir, cam, max_photos=21):
    """Execute robot motion and capture images."""
    img_counter = 0

    # Define a safe initial position
    initial_position = [330, 0, 0, 0] # Example safe position, adjust as needed
    print(f"Moving to initial position: {initial_position}")
    RunPoint(move, initial_position)
    WaitArrive(initial_position)
    print("Arrived at initial position.")

    # Create and open the CSV file for writing poses
    poses_file_path = os.path.join(output_dir, "poses.csv")
    with open(poses_file_path, "w") as poses_file:
        poses_file.write("filename,x,y,z,r,timestamp\n") # Write header

        # Capture photo at the initial position (optional)
        with globalLockValue: # Access global current_actual safely
            # Use the actual robot pose for logging, not the target point
            pose = list(current_actual[:4]) if current_actual is not None else initial_position

        img_counter = capture_and_save(cam, output_dir, img_counter, pose, poses_file)

        for point in path:
            if img_counter >= max_photos:
                print(f"Maximum number of photos ({max_photos}) reached.")
                break
            print(f"Moving to point: {point}")
            RunPoint(move, point)
            WaitArrive(point)
            print("Arrived at point.")

            with globalLockValue: # Access global current_actual safely
                 # Use the actual robot pose for logging, not the target point
                pose = list(current_actual[:4]) if current_actual is not None else point

            img_counter = capture_and_save(cam, output_dir, img_counter, pose, poses_file)
            # sleep(motion_delay) # Optional short delay between points if needed

        # Move back to the initial position after completing the path
        print(f"Moving back to initial position: {initial_position}")
        RunPoint(move, initial_position)
        WaitArrive(initial_position)
        print("Arrived back at initial position.")

    print("Robot motion and image capture complete.")


if __name__ == '__main__':
    dashboard, move, feed = ConnectRobot()
    print("Enabling robot...")
    dashboard.EnableRobot()
    print("Robot enabled!")

    # Start feedback and error clearing threads
    feed_thread = threading.Thread(target=GetFeed, args=(feed,))
    feed_thread.setDaemon(True) # Allows the main program to exit even if this thread is running
    feed_thread.start()
    print("Robot feedback thread started.")

    error_thread = threading.Thread(target=ClearRobotError, args=(dashboard,))
    error_thread.setDaemon(True) # Allows the main program to exit even if this thread is running
    error_thread.start()
    print("Robot error clearing thread started.")

    dashboard.Tool(8)
    # Assuming the camera is mounted 53mm along the Z-axis of the tool frame
    dashboard.SetTool(8, 53, 0, 0, 0)
    
    move.MovL(330, 0, 0, 0)
    dashboard.SpeedFactor(10)

    # Set up camera
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        print("Error: Could not open camera.")
        # Attempt to disable robot before exiting
        try:
            dashboard.DisableRobot()
        except Exception as e:
            print(f"Error disabling robot: {e}")
        sys.exit()

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cam.set(cv2.CAP_PROP_FPS, 5)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(f"Camera resolution: {int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    # Create output directory
    session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    # Output directory is relative to the script's parent's parent directory, in calibration_images
    output_dir = os.path.join(os.path.dirname(__file__), "calibration_images", session_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving images and poses to: {output_dir}")

    # Generate grid path
    # Adjust these parameters based on your chessboard size and desired coverage
    center_x, center_y = 330, 0 # Center of the grid in robot base coordinates
    size_x, size_y = 100, 80     # Size of the grid in X and Y directions (mm)
    stepX = 25                 # Step size between points (mm)
    stepY = 40             # Step size between points (mm)
    path = generate_grid_path(center_x, center_y, size_x, size_y, stepX, stepY)
    print(f"Generated a grid path with {len(path)} points.")

    # Setup display for camera preview
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) # Optional fullscreen

    print("Align the chessboard or target. Press space to start the robot motion and image capture. Press 'q' to quit.")

    # Camera preview loop while waiting for user input to start
    start_capture = False
    while not start_capture:
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture frame from camera preview.")
            sleep(0.1)
            continue
        cv2.line(frame, (frame.shape[1] // 2, 0), (frame.shape[1] // 2, frame.shape[0]), (255, 0, 0), 1)
        cv2.line(frame, (0, frame.shape[0] // 2), (frame.shape[1], frame.shape[0] // 2), (255, 0, 0), 1)
        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            start_capture = True
            print("Starting robot motion and image capture...")
            break
        elif key == ord('q'):
            print("Quit command received from preview.")
            break

    if start_capture:
        # Execute robot motion and capture images in a separate thread
        motion_thread = threading.Thread(target=robot_motion, args=(move, path, output_dir, cam, 21))
        motion_thread.start()

        # Keep camera preview running while motion thread is active
        print("Capturing images...")
        while motion_thread.is_alive():
            ret, frame = cam.read()
            if not ret:
                 print("Failed to capture frame from camera during motion preview.")
                 sleep(0.1) # Prevent busy waiting if camera fails
                 continue
            
            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("'q' pressed, attempting to stop robot motion and preview.")
                # A more robust stop would involve a shared event between main and motion_thread
                # For simplicity here, pressing 'q' stops the preview and the main thread will
                # wait for the motion thread to finish or be manually interrupted (e.g. Ctrl+C).
                break # Exit preview loop

        motion_thread.join() # Wait for the motion thread to complete

    # Cleanup
    print("Releasing camera and disabling robot.")
    cam.release()
    cv2.destroyAllWindows()
    # Attempt to disable robot
    try:
        dashboard.DisableRobot()
    except Exception as e:
        print(f"Error disabling robot during cleanup: {e}")
    print("Cleanup complete.")
