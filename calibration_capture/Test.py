import cv2
import numpy as np
import os
import sys
import threading
from time import sleep, time
from datetime import datetime

# Add parent directory to sys.path for Dobot API
script_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(parent_dir)
from dobot_api import DobotApiDashboard, DobotApi, DobotApiMove, MyType

# Global variables for robot feedback
current_actual = None
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
    global current_actual, enableStatus_robot, robotErrorState
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
                    dashboard.ClearError()
                    sleep(0.01)
                    dashboard.Continue()
        except Exception as e:
            print(f"Error in clear_robot_error: {e}")
        sleep(5)

def load_calibration_data(calib_file):
    """Load camera calibration data."""
    try:
        data = np.load(calib_file)
        mtx = data['mtx']
        dist = data['dist']
        print(f"Loaded calibration from {calib_file}")
        return mtx, dist
    except Exception as e:
        print(f"Error loading calibration: {e}")
        return None, None

def generate_grid_path(center_x, center_y, size_x, size_y, step):
    """Generate a grid path for the robot tool (camera at Z-52mm)."""
    path = []
    for y in np.arange(center_y - size_y / 2, center_y + size_y / 2 + step, step):
        for x in np.arange(center_x - size_x / 2, center_x + size_x / 2 + step, step):
            path.append([x, y, 0, 0])  # Tool Z=0, R=0
    return path

def get_chessboard_pose(gray, pattern_size=(9, 7), square_size_mm=10.0, mtx=None, dist=None):
    """Compute chessboard pose (rvec, tvec) from grayscale image."""
    if mtx is None or dist is None:
        print("No calibration data provided for pose estimation.")
        return None, None, None
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size_mm
    ret, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not ret:
        return None, None, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    ret, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)
    if not ret:
        return None, None, None
    return rvec, tvec, corners2

def compute_measurements(rvec, tvec, tool_pose, camera_offset_z=52.0):
    """Compute distance from camera to chessboard center and chessboard angle."""
    if rvec is None or tvec is None:
        return None, None, None, None
    # Chessboard center (4.5, 3.5) in board coordinates (mm)
    chessboard_center = np.array([[45.0, 35.0, 0.0]], dtype=np.float32)
    R, _ = cv2.Rodrigues(rvec)
    # Chessboard center in camera coordinates
    center_cam = R @ chessboard_center.T + tvec
    center_cam = center_cam.flatten()
    # Approximate T_cam2robot (identity with offset for live demo)
    T_cam2robot = np.eye(4)
    T_cam2robot[2, 3] = -camera_offset_z  # Camera at tool Z-52mm
    # Transform chessboard center to robot coordinates
    center_cam_hom = np.append(center_cam, 1.0)
    center_robot = T_cam2robot @ center_cam_hom
    center_robot = center_robot[:3]
    # Camera position in robot coordinates (tool pose with Z offset)
    camera_pose = np.array([tool_pose[0], tool_pose[1], tool_pose[2] - camera_offset_z])
    # Distance (Euclidean, mm)
    distance = np.linalg.norm(center_robot - camera_pose)
    # Chessboard angle (yaw relative to robot X-axis)
    yaw = np.arctan2(R[1, 0], R[0, 0])  # Angle of chessboard X-axis
    angle_deg = np.degrees(yaw)
    return distance, angle_deg, center_robot, camera_pose

def capture_and_save(cam, output_dir, img_counter, pose, distance, angle, frame, frame_undistorted, corners):
    """Capture image and save with measurements."""
    sleep(0.5)
    ret, frame = cam.read()
    if not ret:
        print("Failed to capture image")
        return img_counter
    # Undistort the captured frame
    frame_undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_name = os.path.join(output_dir, f"image{img_counter + 1:03d}.jpg")
    img_name_undistorted = os.path.join(output_dir, f"image{img_counter + 1:03d}_undistorted.jpg")
    cv2.imwrite(img_name, frame)
    cv2.imwrite(img_name_undistorted, frame_undistorted)
    distance_str = f"{distance:.2f}" if distance is not None else "N/A"
    angle_str = f"{angle:.2f}" if angle is not None else "N/A"
    first_corner = f"{corners[0][0][0]:.2f},{corners[0][0][1]:.2f}" if corners is not None else "N/A"
    last_corner = f"{corners[-1][0][0]:.2f},{corners[-1][0][1]:.2f}" if corners is not None else "N/A"
    with open(os.path.join(output_dir, "measurements.csv"), "a") as f:
        f.write(f"{img_name},{pose[0]:.6f},{pose[1]:.6f},{pose[2]:.6f},{pose[3]:.6f},{distance_str},{angle_str},{first_corner},{last_corner},{timestamp}\n")
    print(f"Saved: {img_name} (and undistorted) with tool pose {pose}, distance {distance_str} mm, angle {angle_str} deg, corners: ({first_corner}), ({last_corner})")
    return img_counter + 1

def check_workspace(pose):
    """Check if pose is within MG400 workspace."""
    x, y, z = pose[:3]
    if not (200 <= x <= 600 and -300 <= y <= 300 and z == 0.0):
        print(f"Invalid pose {pose}: Outside workspace (X: 200–600, Y: -300 to 300, Z: 0)")
        return False
    return True

def robot_motion(move, path, stop_event, output_dir, cam, mtx, dist, max_photos=21):
    """Execute robot motion and capture images with measurements."""
    global mapx, mapy
    img_counter = 0
    initial_position = [300, 0, 0, 0]  # Tool position (camera at 300, 0, -52, 0)
    if not check_workspace(initial_position):
        stop_event.set()
        return
    run_point(move, initial_position)
    wait_arrive(initial_position)
    with globalLockValue:
        pose = list(current_actual[:4]) if current_actual is not None else initial_position
    ret, frame = cam.read()
    if ret:
        frame_undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)
        rvec, tvec, corners = get_chessboard_pose(gray, mtx=mtx, dist=dist)
        distance, angle, _, _ = compute_measurements(rvec, tvec, pose)
        img_counter = capture_and_save(cam, output_dir, img_counter, pose, distance, angle, frame, frame_undistorted, corners)
    for point in path:
        if stop_event.is_set():
            break
        if not check_workspace(point):
            print(f"Skipping point {point}: Outside workspace")
            continue
        run_point(move, point)
        wait_arrive(point)
        with globalLockValue:
            pose = list(current_actual[:4]) if current_actual is not None else point
        ret, frame = cam.read()
        if ret:
            frame_undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
            gray = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)
            rvec, tvec, corners = get_chessboard_pose(gray, mtx=mtx, dist=dist)
            distance, angle, _, _ = compute_measurements(rvec, tvec, pose)
            img_counter = capture_and_save(cam, output_dir, img_counter, pose, distance, angle, frame, frame_undistorted, corners)
        if img_counter >= max_photos:
            stop_event.set()
            break
        sleep(0.1)
    run_point(move, initial_position)
    wait_arrive(initial_position)
    stop_event.set()

def main():
    global mapx, mapy
    # Prompt for calibration file
    calib_dir = input("Enter the calibration directory (e.g., ../calibration_images/session_20250517_034339): ")
    calib_file = os.path.join(calib_dir, "calibration_data.npz")
    if not os.path.exists(calib_file):
        print(f"Calibration file {calib_file} does not exist. Exiting.")
        return
    mtx, dist = load_calibration_data(calib_file)
    if mtx is None or dist is None:
        print("Failed to load calibration data. Exiting.")
        return
    session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    output_dir = os.path.join("..", "calibration_images", session_name)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "measurements.csv"), "w") as f:
        f.write("filename,x,y,z,r,distance_mm,angle_deg,first_corner_x_y,last_corner_x_y,timestamp\n")
    dashboard, move, feed = connect_robot()
    dashboard.EnableRobot()
    print("Robot enabled!")
    feed_thread = threading.Thread(target=get_feed, args=(feed,), daemon=True)
    feed_thread.start()
    error_thread = threading.Thread(target=clear_robot_error, args=(dashboard,), daemon=True)
    error_thread.start()
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
    print(f"Live resolution: {frame_width}x{frame_height}")
    # Initialize undistortion maps
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (frame_width, frame_height), 0, (frame_width, frame_height)
    )
    mapx, mapy = cv2.initUndistortRectifyMap(
        mtx, dist, None, newcameramtx, (frame_width, frame_height), cv2.CV_32FC1
    )
    cv2.namedWindow("Undistorted Camera", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Undistorted Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    center_x, center_y = 300, 0  # Tool center (camera at 300, 0, -52, 0)
    size_x, size_y = 80, 60  # 80x60 mm grid
    step = 20  # 20mm steps
    path = generate_grid_path(center_x, center_y, size_x, size_y, step)
    stop_event = threading.Event()
    initial_position = [300, 0, 0, 0]
    if not check_workspace(initial_position):
        print("Initial position outside workspace. Exiting.")
        cam.release()
        dashboard.DisableRobot()
        return
    run_point(move, initial_position)
    wait_arrive(initial_position)
    print("Align the chessboard (9x7 inner corners) at the center. Press space to start capture when aligned. Press 'q' to quit.")
    aligned = False
    while not stop_event.is_set():
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture frame")
            continue
        frame_undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)
        rvec, tvec, corners = get_chessboard_pose(gray, mtx=mtx, dist=dist)
        with globalLockValue:
            pose = list(current_actual[:4]) if current_actual is not None else initial_position
        distance, angle, center_robot, camera_pose = compute_measurements(rvec, tvec, pose)
        if corners is not None:
            cv2.drawChessboardCorners(frame_undistorted, (9, 7), corners, True)
            center = np.mean(corners, axis=0)[0]
            img_center = (frame_undistorted.shape[1] / 2, frame_undistorted.shape[0] / 2)
            pixel_distance = np.linalg.norm(center - img_center)
            rect = cv2.minAreaRect(corners)
            rect_angle = rect[2]
            if pixel_distance < 10 and (abs(rect_angle) < 5 or abs(rect_angle - 90) < 5):
                aligned = True
                cv2.putText(frame_undistorted, "Aligned! Press space to start.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                aligned = False
                cv2.putText(frame_undistorted, "Align the chessboard.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            aligned = False
            cv2.putText(frame_undistorted, "Chessboard not detected.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Display distance, angle, and positions
        distance_str = f"Distance: {distance:.2f} mm" if distance is not None else "Distance: N/A"
        angle_str = f"Angle: {angle:.2f} deg" if angle is not None else "Angle: N/A"
        center_str = f"Chessboard center: X={center_robot[0]:.2f}, Y={center_robot[1]:.2f}, Z={center_robot[2]:.2f}" if center_robot is not None else "Center: N/A"
        camera_str = f"Camera: X={camera_pose[0]:.2f}, Y={camera_pose[1]:.2f}, Z={camera_pose[2]:.2f}" if camera_pose is not None else "Camera: N/A"
        cv2.putText(frame_undistorted, distance_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame_undistorted, angle_str, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame_undistorted, center_str, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame_undistorted, camera_str, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.line(frame_undistorted, (frame_undistorted.shape[1] // 2, 0), (frame_undistorted.shape[1] // 2, frame_undistorted.shape[0]), (255, 0, 0), 1)
        cv2.line(frame_undistorted, (0, frame_undistorted.shape[0] // 2), (frame_undistorted.shape[1], frame_undistorted.shape[0] // 2), (255, 0, 0), 1)
        cv2.imshow("Undistorted Camera", frame_undistorted)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and aligned:
            break
        elif key == ord('q'):
            stop_event.set()
            break
    if not stop_event.is_set():
        robot_thread = threading.Thread(
            target=robot_motion, args=(move, path, stop_event, output_dir, cam, mtx, dist, 21), daemon=True
        )
        robot_thread.start()
        print("Capturing 21 photos. Press 'q' to quit early.")
        while not stop_event.is_set():
            ret, frame = cam.read()
            if not ret:
                print("Failed to capture frame")
                continue
            frame_undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
            gray = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)
            rvec, tvec, corners = get_chessboard_pose(gray, mtx=mtx, dist=dist)
            with globalLockValue:
                pose = list(current_actual[:4]) if current_actual is not None else initial_position
            distance, angle, center_robot, camera_pose = compute_measurements(rvec, tvec, pose)
            if corners is not None:
                cv2.drawChessboardCorners(frame_undistorted, (9, 7), corners, True)
            distance_str = f"Distance: {distance:.2f} mm" if distance is not None else "Distance: N/A"
            angle_str = f"Angle: {angle:.2f} deg" if angle is not None else "Angle: N/A"
            center_str = f"Chessboard center: X={center_robot[0]:.2f}, Y={center_robot[1]:.2f}, Z={center_robot[2]:.2f}" if center_robot is not None else "Center: N/A"
            camera_str = f"Camera: X={camera_pose[0]:.2f}, Y={camera_pose[1]:.2f}, Z={camera_pose[2]:.2f}" if camera_pose is not None else "Camera: N/A"
            cv2.putText(frame_undistorted, distance_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame_undistorted, angle_str, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame_undistorted, center_str, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame_undistorted, camera_str, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame_undistorted, "Capturing... Press 'q' to quit.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.line(frame_undistorted, (frame_undistorted.shape[1] // 2, 0), (frame_undistorted.shape[1] // 2, frame_undistorted.shape[0]), (255, 0, 0), 1)
            cv2.line(frame_undistorted, (0, frame_undistorted.shape[0] // 2), (frame_undistorted.shape[1], frame_undistorted.shape[0] // 2), (255, 0, 0), 1)
            cv2.imshow("Undistorted Camera", frame_undistorted)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
        robot_thread.join()
    cam.release()
    cv2.destroyAllWindows()
    dashboard.DisableRobot()
    print("Resources released and robot disabled.")

if __name__ == "__main__":
    main()