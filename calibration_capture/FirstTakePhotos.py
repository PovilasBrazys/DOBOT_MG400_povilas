import cv2
import numpy as np
import os
import sys
import threading
from time import sleep, time
import re
from datetime import datetime
import pandas as pd
import glob
#pip install opencv-python
#pip install pandas

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
                    numbers = [int(num) for num in re.findall(r'-?\d+', numbers)] if isinstance(numbers, str) else numbers
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

def capture_and_save(cam, output_dir, img_counter, pose):
    """Capture a single image, save it, and log the robot pose."""
    sleep(0.5)
    ret, frame = cam.read()
    if not ret:
        print("Failed to capture image")
        return img_counter
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_name = os.path.join(output_dir, f"image{img_counter + 1:03d}.jpg")
    cv2.imwrite(img_name, frame)
    with open(os.path.join(output_dir, "poses.csv"), "a") as f:
        f.write(f"{img_name},{pose[0]:.2f},{pose[1]:.2f},{pose[2]:.2f},{pose[3]:.2f},{timestamp}\n")
    print(f"Saved: {img_name} with pose {pose}")
    return img_counter + 1

def calibrate_camera(images, pattern_size=(9, 7), square_size_mm=10.0, output_dir="./calibration_output"):
    """Calibrate the camera using chessboard images."""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    nx, ny = pattern_size
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) * square_size_mm
    objpoints = []
    imgpoints = []
    debug_images_dir = os.path.join(output_dir, "debug_images")
    os.makedirs(debug_images_dir, exist_ok=True)
    success_files = []
    failed_files = []
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"❌ Could not read {fname}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            objpoints.append(objp)
            success_files.append(fname)
            vis_img = cv2.drawChessboardCorners(img.copy(), pattern_size, corners2, ret)
            cv2.imwrite(os.path.join(debug_images_dir, os.path.basename(fname)), vis_img)
            print(f"✅ Found corners in: {fname}")
            print(f"  First corner (0,0): {corners2[0].flatten()}")
            print(f"  Last corner ({nx-1},{ny-1}): {corners2[-1].flatten()}")
        else:
            failed_files.append(fname)
            print(f"⚠️ Chessboard NOT found in: {fname}")
    if len(objpoints) < 10:
        print("⚠️ WARNING: Fewer than 10 valid images. Calibration may be inaccurate.")
        return None, None
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    mean_error = total_error / len(objpoints)
    print(f"📐 Reprojection Error: {mean_error:.4f} pixels")
    np.savez(os.path.join(output_dir, "calibration_data.npz"), mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    print(f"💾 Calibration data saved to {output_dir}/calibration_data.npz")
    return mtx, dist

def get_reference_pose(mtx, dist, chessboard_image_path, pattern_size=(9, 7), square_size_mm=10.0):
    """Get chessboard pose from reference image."""
    img = cv2.imread(chessboard_image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    objp = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size_mm
    found, corners = cv2.findChessboardCorners(gray, pattern_size)
    if not found:
        raise RuntimeError("Chessboard not found in reference image.")
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
    ok, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)
    if not ok:
        raise RuntimeError("solvePnP failed.")
    R, _ = cv2.Rodrigues(rvec)
    print(f"Chessboard rotation matrix:\n{R}")
    print(f"Chessboard translation vector: {tvec.flatten()}")
    return rvec, tvec

def compute_camera_to_robot_transform(pose_csv, mtx, dist, pattern_size=(9, 7), square_size_mm=10.0, camera_offset_z=52.0):
    """Compute T_cam2robot using pose.csv data, aligning chessboard origin (0,0) with robot (300, 0)."""
    df = pd.read_csv(pose_csv)
    cam_points = []
    robot_points = []
    for _, row in df.iterrows():
        img_path = row['filename']
        robot_pose = [row['x'], row['y'], row['z']]
        try:
            rvec, tvec = get_reference_pose(mtx, dist, img_path, pattern_size, square_size_mm)
        except RuntimeError as e:
            print(f"Skipping {img_path}: {e}")
            continue
        col, row = 0, 0  # Chessboard origin (0,0)
        cam_point = get_camera_coords_from_board(col, row, rvec, tvec, square_size_mm)
        # Apply 52mm offset in robot's tool frame (negative Z, downward)
        robot_pose_tool = np.array(robot_pose + [0.0])  # Homogeneous robot pose
        tool_offset = np.array([0.0, 0.0, -camera_offset_z, 0.0])  # Offset in tool frame
        robot_pose_tool += tool_offset
        cam_points.append(cam_point)
        robot_points.append(robot_pose_tool[:3])
        print(f"Image: {img_path}, Cam point: {cam_point}, Robot point (with 52mm offset): {robot_pose_tool[:3]}")
    if len(cam_points) < 4:
        raise RuntimeError("Not enough valid poses for calibration.")
    cam_points = np.array(cam_points, dtype=np.float32)
    robot_points = np.array(robot_points, dtype=np.float32)
    cam_centroid = np.mean(cam_points, axis=0)
    robot_centroid = np.mean(robot_points, axis=0)
    cam_centered = cam_points - cam_centroid
    robot_centered = robot_points - robot_centroid
    H = cam_centered.T @ robot_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = robot_centroid - R @ cam_centroid
    T_cam2robot = np.eye(4)
    T_cam2robot[:3, :3] = R
    T_cam2robot[:3, 3] = t
    print("T_cam2robot:\n", T_cam2robot)
    return T_cam2robot

def get_camera_coords_from_board(col, row, rvec, tvec, square_size_mm=10.0):
    """Convert board coordinates to camera coordinates."""
    point_on_board = np.array([[col * square_size_mm, row * square_size_mm, 0.0]], dtype=np.float32)
    R, _ = cv2.Rodrigues(rvec)
    cam_point = R @ point_on_board.T + tvec
    return cam_point.flatten()

def pixel_to_board(u, v, K, rvec, tvec):
    """Project pixel coordinates to chessboard plane (mm)."""
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)
    R_inv = R.T
    t_inv = -R_inv @ t
    uv1 = np.array([u, v, 1.0], dtype=np.float32).reshape(3, 1)
    cam_ray = np.linalg.inv(K) @ uv1
    R3 = R_inv[2:3, :]
    num = t_inv[2, 0]
    den = (R3 @ cam_ray)[0]  # Explicitly extract scalar
    if abs(den) < 1e-6:
        return None
    s = -num / den
    world_pt = R_inv @ (s * cam_ray) + t_inv
    return world_pt[0, 0], world_pt[1, 0]

def transform_camera_to_robot(cam_point, T_cam2robot):
    """Transform camera point to robot coordinates."""
    cam_point_hom = np.append(cam_point, 1.0)
    robot_point_hom = T_cam2robot @ cam_point_hom
    return robot_point_hom[:3]

def move_robot_to_circle(u, v, mtx, dist, rvec, tvec, T_cam2robot, move, frame_width, frame_height):
    """Move robot to detected circle's position without X-axis flip."""
    board_point = pixel_to_board(u, v, mtx, rvec, tvec)
    if board_point is None:
        print("Error: Cannot project circle to board plane.")
        return None
    X_mm, Y_mm = board_point
    cam_point = np.array([X_mm, Y_mm, 0.0], dtype=np.float32)
    robot_point = transform_camera_to_robot(cam_point, T_cam2robot)
    center_u, center_v = frame_width / 2.0, frame_height / 2.0
    center_board_point = pixel_to_board(center_u, center_v, mtx, rvec, tvec)
    if center_board_point is None:
        print("Error: Cannot project screen center to board plane.")
        return None
    center_X_mm, center_Y_mm = center_board_point
    center_cam_point = np.array([center_X_mm, center_Y_mm, 0.0], dtype=np.float32)
    center_robot_point = transform_camera_to_robot(center_cam_point, T_cam2robot)
    center_y_robot = center_robot_point[1]
    pose = [robot_point[0], robot_point[1] - center_y_robot, -145.0, 0.0]
    print(f"Circle pixel (u,v): ({u}, {v}), Board (X,Y): ({X_mm:.2f}, {Y_mm:.2f}), Robot pose: {pose}")
    return pose

def detect_circle(gray, assumed_z_mm=100, mtx=None):
    """Detect a black circle (10mm ± 5mm diameter) in a grayscale image."""
    if mtx is None:
        return None, None
    focal_length = mtx[0, 0]
    diameter_range_mm = (5.0, 15.0)
    min_radius_px = int((diameter_range_mm[0] / 2) * focal_length / assumed_z_mm)
    max_radius_px = int((diameter_range_mm[1] / 2) * focal_length / assumed_z_mm)
    min_area = np.pi * min_radius_px ** 2
    max_area = np.pi * max_radius_px ** 2
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if 0.8 < circularity < 1.2:
                ((x, y), (major, minor), _) = cv2.fitEllipse(contour)
                radius = (major + minor) / 4
                if min_radius_px <= radius <= max_radius_px:
                    return (int(x), int(y)), int(radius)
    return None, None

def robot_motion(move, path, stop_event, output_dir, cam, max_photos=20, motion_delay=0.1):
    """Execute robot motion and capture images."""
    img_counter = 0
    initial_position = [300, 0, 0, 0]
    run_point(move, initial_position)
    wait_arrive(initial_position)
    with globalLockValue:
        pose = list(current_actual[:4]) if current_actual is not None else initial_position
    img_counter = capture_and_save(cam, output_dir, img_counter, pose)
    for point in path:
        if stop_event.is_set():
            break
        run_point(move, point)
        wait_arrive(point)
        with globalLockValue:
            pose = list(current_actual[:4]) if current_actual is not None else point
        img_counter = capture_and_save(cam, output_dir, img_counter, pose)
        if img_counter >= max_photos:
            stop_event.set()
            break
        sleep(motion_delay)
    run_point(move, initial_position)
    wait_arrive(initial_position)
    stop_event.set()

def check_workspace(pose):
    """Check if pose is within MG400 workspace."""
    x, y, z = pose[:3]
    if not (200 <= x <= 600 and -300 <= y <= 300 and z == -145.0):
        print(f"Invalid pose {pose}: Outside workspace (X: 200–600, Y: -300 to 300, Z: -145)")
        return False
    return True

def main():
    session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    output_dir = os.path.join("..", "calibration_images", session_name)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "poses.csv"), "w") as f:
        f.write("filename,x,y,z,r,timestamp\n")
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
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    center_x, center_y = 300, 0
    size_x, size_y = 80, 60
    step = 20
    path = generate_grid_path(center_x, center_y, size_x, size_y, step)
    stop_event = threading.Event()
    initial_position = [300, 0, 0, 0]
    run_point(move, initial_position)
    wait_arrive(initial_position)
    print("Align the chessboard (9x7 inner corners) at the center. Press space to start capture when aligned. Press 'q' to quit.")
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
                cv2.putText(frame, "Aligned! Press space to start.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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
    images = sorted(glob.glob(os.path.join(output_dir, "image*.jpg")))
    mtx, dist = calibrate_camera(images, pattern_size=(9, 7), square_size_mm=10.0, output_dir=output_dir)
    if mtx is None or dist is None:
        print("Camera calibration failed. Exiting.")
        cam.release()
        cv2.destroyAllWindows()
        dashboard.DisableRobot()
        return
    pose_csv = os.path.join(output_dir, "poses.csv")
    try:
        T_cam2robot = compute_camera_to_robot_transform(pose_csv, mtx, dist, pattern_size=(9, 7), square_size_mm=10.0, camera_offset_z=52.0)
    except Exception as e:
        print(f"Failed to compute T_cam2robot: {e}")
        cam.release()
        cv2.destroyAllWindows()
        dashboard.DisableRobot()
        return
    # Debug: Test chessboard origin mapping
    reference_image = os.path.join(output_dir, "image001.jpg")
    rvec_ref, tvec_ref = get_reference_pose(mtx, dist, reference_image, pattern_size=(9, 7), square_size_mm=10.0)
    col, row = 0, 0  # Chessboard origin
    cam_point = get_camera_coords_from_board(col, row, rvec_ref, tvec_ref, square_size_mm=10.0)
    robot_point = transform_camera_to_robot(cam_point, T_cam2robot)
    print(f"Chessboard origin (0,0): Cam coords {cam_point}, Robot coords {robot_point}")
    # Validation: Move to known chessboard points
    print("Validating calibration...")
    test_points = [[0, 0], [4.5, 3.5], [9, 7]]  # Origin, center, opposite corner
    for col, row in test_points:
        cam_point = get_camera_coords_from_board(col, row, rvec_ref, tvec_ref, square_size_mm=10.0)
        robot_point = transform_camera_to_robot(cam_point, T_cam2robot)
        center_u, center_v = frame_width / 2.0, frame_height / 2.0
        center_board_point = pixel_to_board(center_u, center_v, mtx, rvec_ref, tvec_ref)
        center_X_mm, center_Y_mm = center_board_point if center_board_point else (0, 0)
        center_cam_point = np.array([center_X_mm, center_Y_mm, 0.0], dtype=np.float32)
        center_robot_point = transform_camera_to_robot(center_cam_point, T_cam2robot)
        center_y_robot = center_robot_point[1]
        pose = [robot_point[0], robot_point[1] - center_y_robot, -145.0, 0.0]
        print(f"Testing chessboard point ({col}, {row}): Robot pose {pose}")
        if check_workspace(pose):
            #run_point(move, pose)
            #wait_arrive(pose)
            with globalLockValue:
                actual_pose = list(current_actual[:4]) if current_actual is not None else pose
            print(f"Actual pose: {actual_pose}")
        else:
            print(f"Skipping pose {pose}: Outside workspace")
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (frame_width, frame_height), 0, (frame_width, frame_height)
    )
    mapx, mapy = cv2.initUndistortRectifyMap(
        mtx, dist, None, newcameramtx, (frame_width, frame_height), cv2.CV_32FC1
    )
    cv2.namedWindow("Undistorted Live View", cv2.WINDOW_NORMAL)
    vision_paused = False
    last_move_time = None
    at_initial_position = True
    print("Starting live demo. Place a 10mm black circle on the chessboard. Press space to move robot, 'p' to pause/resume, 'q' to quit.")
    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Failed to capture frame.")
                continue
            undistorted_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
            center_x, center_y = frame_width // 2, frame_height // 2
            cv2.line(undistorted_frame, (center_x - 20, center_y), (center_x + 20, center_y), (255, 0, 0), 2)
            cv2.line(undistorted_frame, (center_x, center_y - 20), (center_x, center_y + 20), (255, 0, 0), 2)
            cv2.circle(undistorted_frame, (center_x, center_y), 5, (255, 0, 0), -1)
            target_position = None
            if not vision_paused:
                gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
                center, radius = detect_circle(gray, assumed_z_mm=100, mtx=newcameramtx)
                if center is not None:
                    u, v = center
                    target_position = move_robot_to_circle(
                        u, v, newcameramtx, dist, rvec_ref, tvec_ref, T_cam2robot, move, frame_width, frame_height
                    )
                    if target_position is not None:
                        cv2.circle(undistorted_frame, (u, v), radius, (0, 0, 255), 2)
                        cv2.circle(undistorted_frame, (u, v), 5, (0, 255, 0), -1)
                        cv2.putText(
                            undistorted_frame,
                            f"Xr={target_position[0]:.1f}mm, Yr={target_position[1]:.1f}mm, Zr={target_position[2]:.1f}mm",
                            (u + 10, v - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                        )
            if not at_initial_position and last_move_time is not None and (time() - last_move_time) >= 5:
                print(f"Returning to initial position: X={initial_position[0]}, Y={initial_position[1]}, Z={initial_position[2]}")
                run_point(move, initial_position)
                if wait_arrive(initial_position):
                    print("Robot returned to initial position.")
                else:
                    print("Failed to return to initial position.")
                last_move_time = None
                at_initial_position = True
            status_text = "Paused" if vision_paused else "Running"
            cv2.putText(
                undistorted_frame,
                f"Undistorted Live View (Space to Move, P to Pause/Resume, Q to Quit) ({status_text})",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2
            )
            cv2.imshow("Undistorted Live View", undistorted_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                vision_paused = not vision_paused
                print(f"Vision detection {'paused' if vision_paused else 'resumed'}.")
            elif key == 32 and target_position is not None and at_initial_position and not vision_paused:
                if check_workspace(target_position):
                    print(f"Moving robot to: X={target_position[0]:.1f}, Y={target_position[1]:.1f}, Z={target_position[2]:.1f}")
                    run_point(move, target_position)
                    if wait_arrive(target_position):
                        print("Robot arrived at target position.")
                        last_move_time = time()
                        at_initial_position = False
                    else:
                        print("Failed to reach target position. Returning to initial position.")
                        run_point(move, initial_position)
                        wait_arrive(initial_position)
                        at_initial_position = True
                else:
                    print(f"Target position out of workspace: X={target_position[0]:.1f}, Y={target_position[1]:.1f}")
    finally:
        cam.release()
        cv2.destroyAllWindows()
        dashboard.DisableRobot()
        print("Resources released and robot disabled.")

if __name__ == "__main__":
    main()