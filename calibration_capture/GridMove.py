import cv2
import numpy as np
import os
import sys
import threading
from time import sleep, time
import re

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
                    print(
                        f"Feedback: current_actual={current_actual}, enableStatus={enableStatus_robot}, errorState={robotErrorState}")
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





def get_reference_pose(mtx, dist, chessboard_image_path,
                       pattern_size=(9, 7), square_size_mm=10.0):
    """
    pattern_size = (cols, rows) = (9,7) for 10×8 squares.
    square_size_mm = 10.0 for your 1cm squares.
    """
    img = cv2.imread(chessboard_image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Build object points: (0,0,0), (10,0,0), … (80,0,0) for 9 wide,
    # then up to y=60 for 7 high.
    objp = np.zeros((pattern_size[1]*pattern_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2) * square_size_mm

    # Find corners
    found, corners = cv2.findChessboardCorners(gray, pattern_size)
    if not found:
        raise RuntimeError("Chessboard not found in reference image.")

    cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))

    # Solve PnP
    ok, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)
    if not ok:
        raise RuntimeError("solvePnP failed.")

    return rvec, tvec


def draw_3d_grid_on_image(img, rvec, tvec, camera_matrix,
                          pattern_size=(9,7), square_size_mm=10.0,
                          color=(0,255,0), thickness=1):
    """
    Draws a grid exactly covering 10×8 squares = 100mm × 80mm, starting
    at the corner where the chessboard was detected.
    """
    cols, rows = pattern_size
    # X spans 0…(cols * square_size_mm), Y spans 0…(rows * square_size_mm)
    xs = np.linspace(0, cols * square_size_mm, cols+1, dtype=np.float32)
    ys = np.linspace(0, rows * square_size_mm, rows+1, dtype=np.float32)

    # Build every line in X direction
    for y in ys:
        p3d = np.vstack([
            [0,   y, 0],
            [cols * square_size_mm, y, 0]
        ]).astype(np.float32)
        p2d, _ = cv2.projectPoints(p3d, rvec, tvec, camera_matrix, None)
        p2d = p2d.reshape(-1,2).astype(int)
        cv2.line(img, tuple(p2d[0]), tuple(p2d[1]), color, thickness)

    # Build every line in Y direction
    for x in xs:
        p3d = np.vstack([
            [x, 0, 0],
            [x, rows * square_size_mm, 0]
        ]).astype(np.float32)
        p2d, _ = cv2.projectPoints(p3d, rvec, tvec, camera_matrix, None)
        p2d = p2d.reshape(-1,2).astype(int)
        cv2.line(img, tuple(p2d[0]), tuple(p2d[1]), color, thickness)

def transform_camera_to_robot(cam_point, T_cam2robot):
    cam_point_hom = np.append(cam_point, 1.0)        # [x, y, z, 1]
    robot_point_hom = T_cam2robot @ cam_point_hom    # 4×1
    return robot_point_hom[:3]                       # [x, y, z] in robot frame

def move_robot_to_circle(u, v, mtx, dist, rvec, tvec, T_cam2robot, move):
    # 1) normalize pixel → direction vector in camera frame
    x = (u - mtx[0,2]) / mtx[0,0]
    y = (v - mtx[1,2]) / mtx[1,1]
    dir_cam = np.array([x, y, 1.0])
    dir_cam /= np.linalg.norm(dir_cam)

    # 2) build board-to-camera transform
    R, _ = cv2.Rodrigues(rvec)
    T_board2cam = np.eye(4)
    T_board2cam[:3,:3] = R
    T_board2cam[:3, 3] = tvec.flatten()

    # 3) find intersection of ray (from cam origin) with board Z=0 plane
    Zc = T_board2cam[2,3]
    t = -Zc / dir_cam[2]
    cam_point = dir_cam * t + T_board2cam[:3,3]

    # 4) transform into robot frame
    robot_xyz = transform_camera_to_robot(cam_point, T_cam2robot)

    # 5) send the move
    pose = [robot_xyz[0], robot_xyz[1], robot_xyz[2], 0.0]
    run_point(move, pose)
    wait_arrive(pose)


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
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
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
def pixel_to_board(u, v, K, rvec, tvec):
    # (As defined previously)
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3,1)
    R_inv = R.T
    t_inv = -R_inv @ t
    uv1 = np.array([u, v, 1.0], dtype=np.float32).reshape(3,1)
    cam_ray = np.linalg.inv(K) @ uv1
    R3 = R_inv[2:3, :]
    num = t_inv[2,0]
    den = float(R3 @ cam_ray)
    if abs(den) < 1e-6:
        return None
    s = - num / den
    world_pt = R_inv @ (s * cam_ray) + t_inv
    return world_pt[0,0], world_pt[1,0]  # X_mm, Y_mm

def transform_camera_to_robot(cam_point, T_cam2robot):
    cam_point_hom = np.append(cam_point, 1.0)  # Make it homogeneous [x, y, z, 1]
    robot_point_hom = T_cam2robot @ cam_point_hom  # Matrix multiply
    return robot_point_hom[:3]

def main():
    calibration_npz_file = 'calibration_data.npz'
    reference_chessboard_image_path = '../calibration_images/image001.jpg'
    chessboard_pattern_size = (7, 9)
    chessboard_square_size_mm = 10.0
    circle_radius_mm = 5.0
    initial_position = [300.0, 0.0, 0.0, 0.0]  # [x, y, z=0, R]
    # Replace these with your measured offsets (in mm)
    T_cam2robot = np.array([
        [1, 0, 0, 352],  # camera X → robot X offset
        [0, 1, 0, 0],  # camera Y → robot Y offset
        [0, 0, 1, 0],  # camera Z → robot Z offset
        [0, 0, 0, 1],
    ])

    # Load calibration data
    if not os.path.exists(calibration_npz_file):
        print(f"FATAL ERROR: Calibration file '{calibration_npz_file}' not found.")
        return
    try:
        calib_data = np.load(calibration_npz_file)
        mtx_calib, dist_calib = calib_data['mtx'], calib_data['dist']
        print("Calibration data loaded successfully.")
    except Exception as e:
        print(f"FATAL ERROR: Could not load mtx/dist: {e}")
        return

    # Connect to the robot
    try:
        dashboard, move, feed = connect_robot()
        print("Enabling robot...")
        dashboard.EnableRobot()
        print("Robot enabled!")
    except Exception as e:
        print("Failed to connect or enable robot:", e)
        return

    # Start feedback and error handling threads
    feed_thread = threading.Thread(target=get_feed, args=(feed,), daemon=True)
    feed_thread.start()
    error_thread = threading.Thread(target=clear_robot_error, args=(dashboard,), daemon=True)
    error_thread.start()

    # Move robot to initial position
    print(
        f"Moving robot to initial position: X={initial_position[0]}, Y={initial_position[1]}, Z={initial_position[2]}")
    run_point(move, initial_position)
    if not wait_arrive(initial_position, timeout=15):
        print("Failed to reach initial position. Check robot status.")
        dashboard.DisableRobot()
        return
    print("Robot at initial position.")

    # Get reference pose
    rvec_ref, tvec_ref = get_reference_pose(
        mtx_calib, dist_calib, reference_chessboard_image_path,
        pattern_size=(9, 7), square_size_mm=10.0
    )

    if rvec_ref is None or tvec_ref is None:
        print("Warning: Could not establish reference pose. Grid will not be drawn.")
    else:
        print("Reference pose for 1cm grid acquired successfully.")

    # Initialize camera
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cam.set(cv2.CAP_PROP_FPS, 5)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cam.isOpened():
        print("Error: Could not open camera.")
        dashboard.DisableRobot()
        return

    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Live resolution: {frame_width}x{frame_height}")

    # Undistortion setup
    newcameramtx_live, roi_live = cv2.getOptimalNewCameraMatrix(
        mtx_calib, dist_calib, (frame_width, frame_height), 0, (frame_width, frame_height)
    )
    mapx_live, mapy_live = cv2.initUndistortRectifyMap(
        mtx_calib, dist_calib, None, newcameramtx_live, (frame_width, frame_height), cv2.CV_32FC1
    )
    cx_px = frame_width / 2.0
    cy_px = frame_height / 2.0
    board_center = pixel_to_board(cx_px, cy_px,
                                  newcameramtx_live, rvec_ref, tvec_ref)
    X0_mm, Y0_mm = board_center  # screen‐center in mm

    cv2.namedWindow('Original Live View', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Undistorted Live View with Grid and Position', cv2.WINDOW_NORMAL)

    vision_paused = False
    last_move_time = None
    at_initial_position = True

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Failed to capture frame.")
                continue

            # Undistort frame
            undistorted_frame = cv2.remap(frame, mapx_live, mapy_live, cv2.INTER_LINEAR)

            # Draw center point marker
            center_x, center_y = frame_width // 2, frame_height // 2
            cv2.line(undistorted_frame, (center_x - 20, center_y), (center_x + 20, center_y), (255, 0, 0), 2)
            cv2.line(undistorted_frame, (center_x, center_y - 20), (center_x, center_y + 20), (255, 0, 0), 2)
            cv2.circle(undistorted_frame, (center_x, center_y), 5, (255, 0, 0), -1)

            # Vision processing only when not paused
            target_position = None
            if not vision_paused and rvec_ref is not None and tvec_ref is not None:
                # Draw 1cm grid
                draw_3d_grid_on_image(
                    undistorted_frame,
                    rvec_ref, tvec_ref,
                    newcameramtx_live,
                    pattern_size=(9, 7), square_size_mm=10.0
                )

                # Detect circle
                gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
                center, radius = detect_circle(gray, assumed_z_mm=100, mtx=newcameramtx_live)

                cx_px, cy_px = frame_width / 2.0, frame_height / 2.0
                board_center = pixel_to_board(cx_px, cy_px,
                                              newcameramtx_live, rvec_ref, tvec_ref)
                if board_center is None:
                    print("Error: cannot project screen center to board plane.")
                    # skip the distance computation
                else:
                    X0_mm, Y0_mm = board_center

                # ... inside your frame loop, after detecting circle center (u,v) ...
                if center is not None:
                    u, v = center
                    # 1) Project pixel (u,v) into chessboard plane (Z=0) in mm
                    board_pt = pixel_to_board(u, v, newcameramtx_live, rvec_ref, tvec_ref)
                    if board_pt is None:
                        print("⚠️  Cannot project to chessboard plane")
                    else:
                        X_board, Y_board = board_pt
                        print(f"Circle in board coords: X={X_board:.1f} mm, Y={Y_board:.1f} mm")

                        # 2) Convert that 3D board‐point into camera frame
                        R, _ = cv2.Rodrigues(rvec_ref)
                        cam_pt = R @ np.array([X_board, Y_board, 0.0]) + tvec_ref.flatten()
                        print(f"Circle in camera coords: x={cam_pt[0]:.1f}, y={cam_pt[1]:.1f}, z={cam_pt[2]:.1f} mm")

                        # 3) Transform camera‐frame point into robot frame
                        robot_pt = transform_camera_to_robot(cam_pt, T_cam2robot)
                        print(f"Circle in robot coords: X={robot_pt[0]:.1f} mm, "
                              f"Y={robot_pt[1]:.1f} mm, Z={robot_pt[2]:.1f} mm")

                    # Project to chessboard plane (mm)
                    board_circle = pixel_to_board(u, v,
                                                  newcameramtx_live, rvec_ref, tvec_ref)
                    if board_circle is not None:
                        Xc_mm, Yc_mm = board_circle

                        # Project screen center once (already computed: X0_mm, Y0_mm)
                        # Compute raw offsets from center
                        dx_mm = Xc_mm - X0_mm
                        dy_mm = Yc_mm - Y0_mm

                        # --- SWAP and OFFSET ---
                        # If you want to treat the board‐Y axis as robot‐X and vice versa:
                        X_robot_mm = dy_mm + 352.0  # add 402 mm offset to X
                        Y_robot_mm = dx_mm  # no offset on Y

                        # Build your target_position [X, Y, Z, R]
                        target_position = [X_robot_mm, Y_robot_mm, -145.0, 0.0]

                        # Draw the circle
                        cv2.circle(undistorted_frame, (u, v), radius, (0, 0, 255), 2)
                        cv2.circle(undistorted_frame, (u, v), 5, (0, 255, 0), -1)

                        # Annotate both the raw and swapped/offset values
                        cv2.putText(undistorted_frame,
                                    f"dx={dx_mm:.1f}mm, dy={dy_mm:.1f}mm",
                                    (u + 10, v - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        cv2.putText(undistorted_frame,
                                    f"Xr={X_robot_mm:.1f}mm, Yr={Y_robot_mm:.1f}mm",
                                    (u + 10, v - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        print("Warning: cannot project circle center to board plane.")

            # Check if 5 seconds have passed since moving to circle
            if not at_initial_position and last_move_time is not None and (time() - last_move_time) >= 5:
                print(
                    f"Returning to initial position: X={initial_position[0]}, Y={initial_position[1]}, Z={initial_position[2]}")
                run_point(move, initial_position)
                if wait_arrive(initial_position, timeout=15):
                    print("Robot returned to initial position.")
                else:
                    print("Failed to return to initial position. Check robot status.")
                last_move_time = None
                at_initial_position = True

            # Display frames with status
            status_text = "Paused" if vision_paused else "Running"
            cv2.putText(frame, f"Original Live View ({status_text})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0, 255, 0), 2)
            cv2.putText(undistorted_frame,
                        f"Undistorted Live View (Press Space to Move Robot, P to Pause/Resume) ({status_text})",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.imshow('Original Live View', frame)
            cv2.imshow('Undistorted Live View with Grid and Position', undistorted_frame)

            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == ord('p'):
                vision_paused = not vision_paused
                print(f"Vision detection {'paused' if vision_paused else 'resumed'}.")
            elif key == 32 and target_position is not None and at_initial_position and not vision_paused:
                if 200 <= target_position[0] <= 600 and -300 <= target_position[1] <= 300:
                    print(
                        f"Moving robot to: X={target_position[0]:.1f}, Y={target_position[1]:.1f}, Z={target_position[2]:.1f}")
                    Xt, Yt, Zt, Rt = target_position
                    print(f"Moving robot to swapped/offset target: X={Xt:.1f}, Y={Yt:.1f}, Z={Zt:.1f}")
                    run_point(move, target_position)
                    if wait_arrive(target_position, timeout=15):
                        print("Robot arrived at target position.")
                        last_move_time = time()
                        at_initial_position = False
                    else:
                        print("Failed to reach target position. Returning to initial position.")
                        run_point(move, initial_position)
                        wait_arrive(initial_position, timeout=15)
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