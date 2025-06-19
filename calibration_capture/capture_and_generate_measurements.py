import cv2
import numpy as np
import os
import sys
import threading
from time import sleep, time
import re
from datetime import datetime
import pandas as pd

# Add parent directory to sys.path for Dobot API
script_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(parent_dir)
from dobot_api import DobotApiDashboard, DobotApi, DobotApiMove, MyType

# Global variables for robot feedback
current_actual = None
algorithm_queue = None
enableStatus_robot = None
robotErrorState = False
globalLockValue = threading.Lock()


def connect_robot():
    """Connect to the Dobot robot."""
    try:
        ip = "192.168.1.6"
        dashboardPort = 29999
        movePort = 30003
        feedPort = 30004
        print("Connecting to robot...")
        dashboard = DobotApiDashboard(ip, dashboardPort)
        move = DobotApiMove(ip, movePort)
        feed = DobotApi(ip, feedPort)
        print("Robot connected!")
        return dashboard, move, feed
    except Exception as e:
        print("Connection error:", e)
        raise e


def run_point(move: DobotApiMove, point_list: list):
    """Move robot to a specified point using linear motion."""
    try:
        print(f"Sending MovL command: {point_list}")
        move.MovL(point_list[0], point_list[1], point_list[2], point_list[3])
    except Exception as e:
        print(f"Error in run_point: {e}")


def get_feed(feed: DobotApi):
    """Retrieve real-time feedback from the robot."""
    global current_actual, algorithm_queue, enableStatus_robot, robotErrorState
    hasRead = 0
    while True:
        try:
            data = bytes()
            while hasRead < 1440:
                temp = feed.socket_dobot.recv(1440 - hasRead)
                if len(temp) > 0:
                    hasRead += len(temp)
                    data += temp
            hasRead = 0
            feedInfo = np.frombuffer(data, dtype=MyType)
            if hex((feedInfo['test_value'][0])) == '0x123456789abcdef':
                with globalLockValue:
                    current_actual = feedInfo["tool_vector_actual"][0]
                    algorithm_queue = feedInfo['isRunQueuedCmd'][0]
                    enableStatus_robot = feedInfo['EnableStatus'][0]
                    robotErrorState = feedInfo['ErrorStatus'][0]
        except Exception as e:
            print(f"Error in get_feed: {e}")
        sleep(0.001)


def wait_arrive(point_list, timeout=15):
    """Wait until the robot arrives at the target point or timeout."""
    start_time = time()
    while time() - start_time < timeout:
        is_arrive = True
        with globalLockValue:
            if current_actual is not None:
                for index in range(4):
                    if abs(current_actual[index] - point_list[index]) > 1:
                        is_arrive = False
                if is_arrive:
                    print("Robot arrived at target.")
                    return True
            else:
                print("No feedback data from robot.")
        sleep(0.001)
    print(f"Timeout waiting for robot to arrive at {point_list}")
    return False


def clear_robot_error(dashboard: DobotApiDashboard):
    """Clear robot errors periodically."""
    global robotErrorState
    while True:
        try:
            with globalLockValue:
                if robotErrorState:
                    numbers = dashboard.GetErrorID()
                    numbers = [int(num) for num in re.findall(r'-?\d+', numbers)] if isinstance(numbers,
                                                                                                str) else numbers
                    if numbers and (numbers[0] != 0 or len(numbers) > 1):
                        print("Robot error detected:", numbers)
                        dashboard.ClearError()
                        sleep(0.01)
                        dashboard.Continue()
        except Exception as e:
            print(f"Error in clear_robot_error: {e}")
        sleep(5)


def generate_grid_path(center_x, center_y, size_x, size_y, step):
    """Generate a grid path over the chessboard."""
    path = []
    for y in np.arange(center_y - size_y / 2, center_y + size_y / 2 + step, step):
        for x in np.arange(center_x - size_x / 2, center_x + size_x / 2 + step, step):
            path.append([x, y, 0, 0])  # Z=0, R=0
    return path


def capture_and_save(cam, output_dir, img_counter, pose, pattern_size=(9, 7), distance_mm=50.0):
    """Capture an image, detect chessboard corners, and log data."""
    sleep(0.5)
    ret, frame = cam.read()
    if not ret:
        print("Failed to capture image")
        return img_counter, None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_name = os.path.join(output_dir, f"image{img_counter + 1:03d}.jpg")
    cv2.imwrite(img_name, frame)

    # Detect chessboard corners
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ret, corners = cv2.findChessboardCorners(
        gray, pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        first_corner_x_y = f"{corners2[0].flatten()[0]:.4f}_{corners2[0].flatten()[1]:.4f}"
        last_corner_x_y = f"{corners2[-1].flatten()[0]:.4f}_{corners2[-1].flatten()[1]:.4f}"

        # Compute angle_deg from chessboard rotation
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * 10.0
        ok, rvec, tvec = cv2.solvePnP(objp, corners2, np.eye(3), None)
        if ok:
            R, _ = cv2.Rodrigues(rvec)
            angle_rad = np.arccos((np.trace(R) - 1) / 2)
            angle_deg = np.degrees(angle_rad)
            if np.isnan(angle_deg):
                angle_deg = 0.0
        else:
            angle_deg = 0.0
    else:
        print(f"Chessboard not found in {img_name}")
        first_corner_x_y = "0_0"
        last_corner_x_y = "0_0"
        angle_deg = 0.0

    # Log data
    data = {
        'filename': img_name,
        'x': pose[0],
        'y': pose[1],
        'z': pose[2],
        'r': pose[3],
        'distance_mm': distance_mm,
        'angle_deg': angle_deg,
        'first_corner_x_y': first_corner_x_y,
        'last_corner_x_y': last_corner_x_y,
        'timestamp': timestamp
    }
    print(f"Saved: {img_name} with pose {pose}, corners {first_corner_x_y}, {last_corner_x_y}, angle {angle_deg:.2f}°")
    return img_counter + 1, data


def robot_motion(move, path, stop_event, output_dir, cam, max_photos=21, motion_delay=0.1):
    """Execute robot motion and capture images."""
    img_counter = 0
    measurements = []
    initial_position = [300, 0, 0, 0]
    run_point(move, initial_position)
    wait_arrive(initial_position)
    with globalLockValue:
        pose = list(current_actual[:4]) if current_actual is not None else initial_position
    img_counter, data = capture_and_save(cam, output_dir, img_counter, pose)
    if data:
        measurements.append(data)
    for point in path:
        if stop_event.is_set():
            break
        run_point(move, point)
        wait_arrive(point)
        with globalLockValue:
            pose = list(current_actual[:4]) if current_actual is not None else point
        img_counter, data = capture_and_save(cam, output_dir, img_counter, pose)
        if data:
            measurements.append(data)
        if img_counter >= max_photos:
            stop_event.set()
            break
        sleep(motion_delay)
    run_point(move, initial_position)
    wait_arrive(initial_position)
    stop_event.set()

    # Save measurements.csv
    if measurements:
        df = pd.DataFrame(measurements)
        csv_path = os.path.join(output_dir, "measurements.csv")
        df.to_csv(csv_path, index=False, sep=',')
        print(f"Saved measurements.csv to {csv_path}")
        print("Generated DataFrame:")
        print(df.to_string(index=False))
    return measurements


def main():
    session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    output_dir = os.path.join("..", "calibration_images", session_name)
    os.makedirs(output_dir, exist_ok=True)

    # Connect robot
    dashboard, move, feed = connect_robot()
    dashboard.EnableRobot()
    print("Robot enabled!")

    # Start feedback and error clearing threads
    feed_thread = threading.Thread(target=get_feed, args=(feed,), daemon=True)
    feed_thread.start()
    error_thread = threading.Thread(target=clear_robot_error, args=(dashboard,), daemon=True)
    error_thread.start()

    # Initialize camera
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cam.set(cv2.CAP_PROP_FPS, 5)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cam.isOpened():
        print("Error: Could not open camera")
        dashboard.DisableRobot()
        return
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {frame_width}x{frame_height}")

    # Setup display
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Generate grid path
    center_x, center_y = 300, 0
    size_x, size_y = 80, 60
    step = 20
    path = generate_grid_path(center_x, center_y, size_x, size_y, step)

    # Wait for chessboard alignment
    stop_event = threading.Event()
    initial_position = [300, 0, 0, 0]
    run_point(move, initial_position)
    wait_arrive(initial_position)
    print(
        "Align the chessboard (9x7 inner corners) at the center. Press space to start capture when aligned. Press 'q' to quit.")
    aligned = False
    while not stop_event.is_set():
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture frame")
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 7))
        if ret:
            cv2.drawChessboardCorners(frame, (9, 7), corners, ret)
            center = np.mean(corners, axis=0)[0]
            img_center = (frame.shape[1] / 2, frame.shape[0] / 2)
            distance = np.linalg.norm(center - img_center)
            rect = cv2.minAreaRect(corners)
            angle = rect[2]
            if distance < 10 and (abs(angle) < 5 or abs(angle - 90) < 5):
                aligned = True
                cv2.putText(frame, "Aligned! Press space to start.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            2)
            else:
                aligned = False
                cv2.putText(frame, "Align the chessboard.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            aligned = False
            cv2.putText(frame, "Chessboard not detected.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.line(frame, (frame.shape[1] // 2, 0), (frame.shape[1] // 2, frame.shape[0]), (255, 0, 0), 1)
        cv2.line(frame, (0, frame.shape[0] // 2), (frame.shape[1], frame.shape[0] // 2), (255, 0, 0), 1)
        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and aligned:
            break
        elif key == ord('q'):
            stop_event.set()
            break

    # Capture images and generate measurements.csv
    if not stop_event.is_set():
        robot_thread = threading.Thread(
            target=robot_motion, args=(move, path, stop_event, output_dir, cam, 21), daemon=True
        )
        robot_thread.start()
        print("Capturing 21 photos. Press 'q' to quit early.")
        while not stop_event.is_set():
            ret, frame = cam.read()
            if not ret:
                print("Failed to capture frame")
                continue
            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
        robot_thread.join()

    # Cleanup
    cam.release()
    cv2.destroyAllWindows()
    dashboard.DisableRobot()
    print("Resources released and robot disabled.")


if __name__ == "__main__":
    main()