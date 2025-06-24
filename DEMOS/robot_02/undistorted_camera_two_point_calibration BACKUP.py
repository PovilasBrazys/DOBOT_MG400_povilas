import cv2
import numpy as np
import os
import threading
import sys
import math
import time
import re # Import re for parsing

# Add parent directory to sys.path to import robot_control
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from robot_control import ConnectRobot, RunPoint, WaitArrive, GetFeed, ClearRobotError, current_actual, globalLockValue
# Note: current_actual and globalLockValue are still imported but will not be used
# for getting pose in the mouse callback as we will use GetPoseInFrame(0, 8)
# The GetFeed thread is still useful for robot error status and other feedback.


# --- Global Variables for Calibration State ---
# States:
# 0: Initial state, waiting for robot connection/setup
# 1: Ready to calibrate Point 1 - Robot needs to move to point 1
# 2: Ready to calibrate Point 1 - User needs to click in the camera feed
# 3: Ready to calibrate Point 2 - Robot needs to move to point 2
# 4: Ready to calibrate Point 2 - User needs to click in the camera feed
# 5: Ready to calibrate Point 3 - Robot needs to move to point 3
# 6: Ready to calibrate Point 3 - User needs to click in the camera feed
# 7: Ready to calibrate Point 4 - Robot needs to move to point 4
# 8: Ready to calibrate Point 4 - User needs to click in the camera feed
# 9: Calibration complete - Transformation calculated
# 10: Operational state - Ready to move robot by clicking
calibration_state = 0

calibration_points_pixel = [] # Stores [(u1, v1), (u2, v2), (u3, v3), (u4, v4)]
calibration_points_robot = [] # Stores [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]
robot_calib_z = None # Store the Z height at calibration points
robot_calib_r = None # Store the R orientation at calibration points (though not strictly needed for 2D calib)

pixel_to_robot_transform = None # Stores the 3x3 perspective transformation matrix

# --- Function to Parse Robot Pose String Response ---
def parse_pose_string(pose_str):
    """
    Parses the robot pose string response from GetPose(user, tool) into a list of floats [X, Y, Z, R, J1, J2].
    Example input: "0,{403.000000,0.000000,-0.000008,0.000000,0.000000,0.000000},GetPose(0,8);"
    """
    try:
        # Find the content within the curly braces {}
        match = re.search(r"\{([^}]+)\}", pose_str)
        if match:
            # Extract the comma-separated numbers
            pose_values_str = match.group(1).split(',')
            # Convert to floats
            numbers = [float(n) for n in pose_values_str]
            # Ensure we have at least 6 numbers (X, Y, Z, R, J1, J2)
            if len(numbers) >= 6:
                return numbers[:6] # Return X, Y, Z, R, J1, J2
            else:
                 print(f"Warning: Could not parse enough pose values from string: {pose_str}")
                 return None
        else:
            # If no curly braces, try parsing as a simple comma-separated list
            try:
                numbers = [float(n) for n in pose_str.split(',')]
                if len(numbers) >= 6:
                    return numbers[:6]
                else:
                     print(f"Warning: Could not parse enough pose values from string (no braces): {pose_str}")
                     return None
            except ValueError:
                 print(f"Warning: Could not parse pose values from string: {pose_str}")
                 return None
    except ValueError as e:
        print(f"Error converting pose values to float from string '{pose_str}': {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during pose string parsing: {e}")
        return None


# --- Mouse Callback Function ---
def mouse_callback(event, u, v, flags, param):
    global calibration_state, calibration_points_pixel, calibration_points_robot
    global robot_calib_z, robot_calib_r, pixel_to_robot_transform
    global dashboard, move  # Access robot control objects

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked pixel: ({u}, {v})")

        # Check if currently in one of the click-collection states (2, 4, 6, 8)
        if calibration_state in [2, 4, 6, 8]:
            # Get robot pose in User 0, Tool 8 using the new function
            # Use a short delay before getting pose to ensure robot has settled and data is updated
            time.sleep(0.1)  # Add a small delay

            pose_response_str = dashboard.GetPoseInFrame(0, 8)
            print(f"Robot pose response: {pose_response_str}") # Debug print
            robot_pose = parse_pose_string(pose_response_str)

            if robot_pose is not None:
                # Store the point and move to the next state
                calibration_points_robot.append(robot_pose[:3])  # Store X, Y, Z
                calibration_points_pixel.append((u, v))  # Store pixel u, v
                
                # Store Z and R from the first point as the operational plane
                if calibration_state == 2:
                    robot_calib_z = robot_pose[2]
                    robot_calib_r = robot_pose[3] # Store R too for potential use

                print(
                    f"Point {len(calibration_points_pixel)} recorded: Robot ({robot_pose[0]:.2f}, {robot_pose[1]:.2f}, {robot_pose[2]:.2f}) -> Pixel ({u}, {v})"
                )

                # Transition to the next state based on the number of points collected
                if len(calibration_points_pixel) == 1:
                    calibration_state = 3  # Point 1 clicked, move to state 3 (Robot Point 2)
                    print("Calibration Step 2: Please move the robot to the second calibration point.")
                elif len(calibration_points_pixel) == 2:
                    calibration_state = 5  # Point 2 clicked, move to state 5 (Robot Point 3)
                    print("Calibration Step 3: Please move the robot to the third calibration point.")
                elif len(calibration_points_pixel) == 3:
                    calibration_state = 7  # Point 3 clicked, move to state 7 (Robot Point 4)
                    print("Calibration Step 4: Please move the robot to the fourth calibration point.")
                elif len(calibration_points_pixel) == 4:
                    # Optional check for consistent Z height for all points
                    if robot_calib_z is not None:
                        z_heights = [p[2] for p in calibration_points_robot]
                        z_diffs = [abs(z - robot_calib_z) for z in z_heights]
                        max_z_diff = max(z_diffs) if z_diffs else 0
                        if max_z_diff > 3.0: # Tolerance in mm
                            print(f"Warning: Maximum Z height difference between calibration points is {max_z_diff:.2f}mm. Ensure points are on the same plane.")


                    calibration_state = 9  # Point 4 clicked, move to state 9 (Calculate Transform)
                    print("Calibration points collected. Calculating transformation...")
                    calculate_pixel_to_robot_transform()  # Calculate the matrix

                    if pixel_to_robot_transform is not None:
                        calibration_state = 10  # Move to operational state
                        print(
                            "Calibration complete. Click on any pixel in the undistorted view to move the robot there."
                        )
                    else:
                        calibration_state = 0  # Reset or handle error
                        print("Failed to calculate transformation matrix.")
                        # Clear collected points if calibration failed
                        calibration_points_pixel = []
                        calibration_points_robot = []

            else:
                print(
                    "Could not get or parse robot pose from GetPose(0, 8) command. Ensure robot is connected and command is supported."
                )

        elif (
            calibration_state == 10
        ):  # Calibrated, ready to move robot by clicking
            if (
                pixel_to_robot_transform is not None
                and robot_calib_z is not None
                and robot_calib_r is not None # Although R is not used in 2D movement, having it recorded is good
            ):
                # Convert clicked pixel (u,v) to robot (X,Y) using the transformation matrix
                target_robot_x, target_robot_y = apply_pixel_to_robot_transform(
                    u, v, pixel_to_robot_transform
                )

                # Command robot to move to the calculated X, Y, using the calibration Z and R
                # Use the Z and R recorded from the first calibration point
                target_robot_pose = [
                    target_robot_x,
                    target_robot_y,
                    robot_calib_z,
                    robot_calib_r,
                ]
                print(f"Attempting to move robot to: {target_robot_pose}")

                # Use a separate thread or ensure commands don't block the main loop too long
                # Simple approach: run directly, but be aware it might pause the camera feed momentarily
                try:
                    # Set robot to use User 0 and Tool 8 before moving (redundant if already set, but safe)
                    dashboard.User(0)
                    dashboard.Tool(8)
                    # Note: RunPoint uses the currently active frames set by User() and Tool()
                    RunPoint(move, target_robot_pose)
                    WaitArrive(target_robot_pose)
                    print("Robot arrived at target.")
                    # Optionally restore original user/tool frames here if needed
                except Exception as e:
                    print(f"Error commanding robot movement: {e}")

            else:
                print("Calibration not complete or transformation matrix not available.")


# --- Transformation Calculation Function ---
def calculate_pixel_to_robot_transform():
    """
    Calculates the 3x3 perspective transformation matrix from pixel coordinates to robot (X, Y) coordinates.
    Uses the four collected point pairs with cv2.getPerspectiveTransform.
    """
    global calibration_points_pixel, calibration_points_robot, pixel_to_robot_transform

    # Ensure we have exactly 4 pairs of points
    if len(calibration_points_pixel) != 4 or len(calibration_points_robot) != 4:
        print("Error: Need exactly four pairs of calibration points to calculate transformation.")
        pixel_to_robot_transform = None
        return

    # Convert points to numpy arrays for cv2.getPerspectiveTransform
    # Source points are pixel coordinates (u, v)
    src_pts = np.array(calibration_points_pixel, dtype=np.float32)

    # Destination points are robot coordinates (X, Y) on the calibration plane
    # We extract X and Y from the robot poses
    dst_pts = np.array([p[:2] for p in calibration_points_robot], dtype=np.float32)

    # Calculate the perspective transformation matrix (Homography)
    # M is a 3x3 matrix
    M, _ = cv2.findHomography(src_pts, dst_pts)
    # Alternatively, cv2.getPerspectiveTransform(src_pts, dst_pts) can also be used
    # M = cv2.getPerspectiveTransform(src_pts, dst_pts)


    if M is not None:
        # The resulting M is the 3x3 matrix that maps pixel points (u, v, 1)
        # to robot points (X', Y', W') such that X = X'/W' and Y = Y'/W'.
        pixel_to_robot_transform = M
        print("Calculated Pixel to Robot Perspective Transformation Matrix (3x3):")
        print(pixel_to_robot_transform)

    else:
        print("Error: cv2.findHomography failed. Check input points (e.g., if they are collinear).")
        pixel_to_robot_transform = None


# --- Transformation Application Function ---
def apply_pixel_to_robot_transform(u, v, transform_matrix):
    """
    Applies the 3x3 perspective transformation matrix to a pixel coordinate (u, v)
    to get the corresponding robot (X, Y) coordinate on the calibration plane.
    """
    if transform_matrix is None:
        print("Error: Transformation matrix is not calculated.")
        return None, None

    # Represent the pixel point as a numpy array for cv2.perspectiveTransform
    # The input needs to be in a specific format: [ [[u, v]] ]
    src_point = np.array([[[u, v]]], dtype=np.float32)

    # Apply the perspective transformation
    # The result is an array like [ [[X, Y]] ]
    dst_point = cv2.perspectiveTransform(src_point, transform_matrix)

    if dst_point is not None and dst_point.shape == (1, 1, 2):
        target_x = dst_point[0, 0, 0]
        target_y = dst_point[0, 0, 1]
        return target_x, target_y
    else:
        print("Error applying perspective transform.")
        return None, None

# Keep the get_reference_pose and draw_3d_grid_on_image functions as they are
# (They are used for drawing the grid, which is optional for calibration but needed for grid visualization)

def get_reference_pose(mtx, dist, chessboard_image_path, pattern_size=(7, 9), square_size_cm=1.0):
    """
    Calculates the rotation and translation vectors (pose) of a chessboard from a static image.
    This pose defines where the Z=0 plane of the grid will be in the world.
    """
    if not os.path.exists(chessboard_image_path):
        print(f"Error: Reference chessboard image not found at '{chessboard_image_image_path}'")
        return None, None

    img = cv2.imread(chessboard_image_path)
    if img is None:
        print(f"Error: Could not read reference chessboard image from '{chessboard_image_path}'")
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size_cm

    ret_corners, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret_corners:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        ret_pnp, rvec, tvec = cv2.solvePnP(objp, corners_refined, mtx, dist)

        if ret_pnp:
            print(f"Reference pose (rvec, tvec) obtained successfully from '{chessboard_image_path}'.")
            return rvec, tvec
        else:
            print(f"Error: solvePnP failed for the reference image '{chessboard_image_path}'.")
            return None, None
    else:
        print(f"Error: Chessboard corners not found in the reference image '{chessboard_image_path}'.")
        return None, None


def draw_3d_grid_on_image(img_to_draw_on, rvec_ref, tvec_ref, camera_matrix_for_projection,
                          pattern_size_chessboard=(7, 9), square_size_cm=1.0,
                          grid_expansion_cm=(3, 3), line_color=(0, 255, 0), line_thickness=1):
    """
    Draws a 3D grid on an image, based on a reference pose (rvec_ref, tvec_ref).
    """
    if rvec_ref is None or tvec_ref is None:
        return

    grid_min_x = -1.0 * square_size_cm - grid_expansion_cm[0]
    grid_max_x = (pattern_size_chessboard[0]) * square_size_cm + grid_expansion_cm[0]
    grid_min_y = -1.0 * square_size_cm - grid_expansion_cm[1]
    grid_max_y = (pattern_size_chessboard[1]) * square_size_cm + grid_expansion_cm[1]
    z_plane = 0.0

    points_3d_lines = []

    current_x = grid_min_x
    while current_x <= grid_max_x + 1e-9:
        points_3d_lines.append(
            np.array([[current_x, grid_min_y, z_plane], [current_x, grid_max_y, z_plane]], dtype=np.float32))
        current_x += square_size_cm

    current_y = grid_min_y
    while current_y <= grid_max_y + 1e-9:
        points_3d_lines.append(
            np.array([[grid_min_x, current_y, z_plane], [grid_max_x, current_y, z_plane]], dtype=np.float32))
        current_y += square_size_cm

    for line_seg_3d in points_3d_lines:
        points_2d_segment, _ = cv2.projectPoints(line_seg_3d, rvec_ref, tvec_ref, camera_matrix_for_projection, None)

        if points_2d_segment is not None:
            pt1 = (int(round(points_2d_segment[0][0][0])), int(round(points_2d_segment[0][0][1])))
            pt2 = (int(round(points_2d_segment[1][0][0])), int(round(points_2d_segment[1][0][1])))
            cv2.line(img_to_draw_on, pt1, pt2, line_color, line_thickness)


# --- Main Function ---
def main():
    global calibration_state, dashboard, move, feed

    # --- Robot Connection ---
    print("Connecting to robot...")
    try:
        dashboard, move, feed = ConnectRobot()
        print("Robot connected.")

        print("Enabling robot...")
        dashboard.EnableRobot()
        print("Robot enabled!")

        move.MovL(330, 0, 0, 0) # Move to a safe initial position

        # Start feedback and error clearing threads
        # GetFeed is still useful for error status and general robot state, even if
        # we use GetPoseInFrame for precise pose readings during calibration clicks.
        feed_thread = threading.Thread(target=GetFeed, args=(feed,))
        feed_thread.setDaemon(True)
        feed_thread.start()
        print("Robot feedback thread started.")

        error_thread = threading.Thread(target=ClearRobotError, args=(dashboard,))
        error_thread.setDaemon(True)
        error_thread.start()
        print("Robot error clearing thread started.")

        # Set tool (adjust tool index and payload/offset as needed)
        # Ensure this matches the tool used for calibration points
        dashboard.Tool(8)
        # Assuming the camera is mounted 53mm along the Z-axis of the tool frame
        dashboard.SetTool(8, 53, 0, 0, 0) # Example values, adjust for your setup

        # Set initial user frame to 0 (important as GetPoseInFrame(0, 8) uses this)
        # Ensure User Frame 0 is calibrated correctly on the robot
        dashboard.User(0)

        dashboard.SpeedFactor(10) # Set initial speed

        calibration_state = 1 # Move to state 1: Ready for Point 1 robot move

    except Exception as e:
        print(f"FATAL ERROR: Could not connect or enable robot: {e}")
        print("Please ensure the robot is connected and the DOBOT software is running.")
        return # Exit if robot connection fails

    # --- Camera Setup ---
    # Construct paths relative to the script's directory, including the session folder
    # NOTE: Update 'session_20250623_221126' if you are using a different calibration session
    script_dir = os.path.dirname(os.path.abspath(__file__))
    session_folder = 'session_20250623_221126' # <--- Specify the session folder name here
    calibration_npz_file = os.path.join(script_dir, 'calibration_images', session_folder, 'calibration_data.npz')
    # Path to the distorted image of the chessboard that defines the grid's reference plane
    # This is only needed for drawing the grid, not for the 4-point calibration itself.
    reference_chessboard_image_path = os.path.join(script_dir, 'calibration_images', session_folder, 'image001.jpg')


    # --- 1. Load Camera Calibration Data ---
    mtx_calib, dist_calib = None, None
    if not os.path.exists(calibration_npz_file):
        print(f"Warning: Calibration file '{calibration_npz_file}' not found.")
        print("Proceeding without camera calibration data (undistortion and grid will not be available).")
    else:
        try:
            calib_data = np.load(calibration_npz_file)
            mtx_calib, dist_calib = calib_data['camera_matrix'], calib_data['dist_coeffs']
            print("Calibration data (camera_matrix, dist_coeffs) loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load calibration data from '{calibration_npz_file}': {e}")
            print("Proceeding without camera calibration data (undistortion and grid will not be available).")


    # --- 2. Get Reference Pose (Optional for calibration but needed for grid) ---
    rvec_ref, tvec_ref = None, None
    # Only attempt to get reference pose if calibration data is available
    if mtx_calib is not None and dist_calib is not None and os.path.exists(reference_chessboard_image_path):
        print(f"\nAttempting to get reference pose from: {os.path.abspath(reference_chessboard_image_path)}")
        # Use original calibration matrix and distortion for solvePnP on the reference image
        # Ensure chessboard parameters match your reference board
        rvec_ref, tvec_ref = get_reference_pose(
            mtx_calib, dist_calib,
            reference_chessboard_image_path,
            pattern_size=(7, 9), # Adjust if your reference board was different
            square_size_cm=1.0   # Adjust if your reference board was different
        )
        if rvec_ref is None or tvec_ref is None:
             print("Warning: Could not establish reference pose from the image. The 1cm grid will not be drawn.")
        else:
             print("Reference pose for the 1cm grid acquired successfully.")
    else:
        print("\nSkipping reference pose calculation (calibration data or reference image not available). Grid will not be drawn.")


    # --- 3. Initialize Camera ---
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_V4L2 on Linux if DSHOW causes issues
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cam.set(cv2.CAP_PROP_FPS, 5)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Reduce buffer size for lower latency

    if not cam.isOpened():
        print("FATAL ERROR: Could not open camera.")
        # Attempt to disable robot before exiting
        try:
            dashboard.DisableRobot()
        except Exception as e:
            print(f"Error disabling robot: {e}")
        return # Exit if camera fails

    # Get actual frame dimensions from camera (might differ from what was set)
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Live camera operational resolution: {frame_width}x{frame_height}")

    center_x = frame_width // 2
    center_y = frame_height // 2

    # --- 4. Prepare for Undistortion of Live Frames ---
    mapx_live, mapy_live = None, None
    newcameramtx_live = None
    # Only prepare undistortion maps if calibration data was loaded
    if mtx_calib is not None and dist_calib is not None:
        # Get the optimal new camera matrix for the live view resolution (alpha=0 to keep all pixels)
        newcameramtx_live, roi_live = cv2.getOptimalNewCameraMatrix(
            mtx_calib, dist_calib, (frame_width, frame_height), 0, (frame_width, frame_height)
        )
        # Pre-compute the undistortion maps for live frames for efficiency
        mapx_live, mapy_live = cv2.initUndistortRectifyMap(
            mtx_calib, dist_calib, None, newcameramtx_live, (frame_width, frame_height), cv2.CV_32FC1
        )
        print("Undistortion maps generated.")
    else:
        print("Skipping undistortion setup (calibration data not available).")


    # --- 5. Setup Display Windows ---
    # We will only display the undistorted view for calibration and operation
    cv2.namedWindow('Undistorted Live View - Calibration', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Undistorted Live View - Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Set the mouse callback for the undistorted window
    cv2.setMouseCallback('Undistorted Live View - Calibration', mouse_callback)

    # Initial instructions
    print("\n--- Four-Point Robot-Camera Calibration ---")
    print("Calibration Step 1: Please move the robot tool to the first calibration point in its workspace.")
    print("Choose four points that form a rectangle or trapezoid on the plane of operation.")
    print("Ensure robot is using User Frame 0 and Tool Frame 8 (or update code).")
    print("Press Space in the camera window when the robot is precisely at the first point.")


    # --- Main Loop ---
    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Failed to capture frame from camera.")
                time.sleep(0.1) # Prevent busy waiting
                continue

            # --- Process and Display Frames ---
            # Apply undistortion if calibration data is available, otherwise use original frame
            undistorted_frame = frame
            if mapx_live is not None and mapy_live is not None:
                 undistorted_frame = cv2.remap(frame, mapx_live, mapy_live, cv2.INTER_LINEAR)

            # ─── Draw center markers ──────────────────────────────────────────────────────
            cv2.drawMarker(undistorted_frame,
                           (center_x, center_y),
                           color=(0, 0, 255),  # red
                           markerType=cv2.MARKER_CROSS,
                           markerSize=20,
                           thickness=2)

            # Draw collected calibration points
            colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0)] # Yellow, Magenta, Cyan, Green
            for i, (u, v) in enumerate(calibration_points_pixel):
                if i < len(colors):
                    color = colors[i]
                else:
                    color = (255, 255, 255) # White for extra points if somehow collected
                cv2.circle(undistorted_frame, (u, v), 8, color, -1) # Increased marker size
                text = f"P{i+1}"
                cv2.putText(undistorted_frame, text, (u + 15, v + 15), # Adjusted text position
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


            # If a reference pose was successfully obtained and undistortion is available, draw the 3D grid
            # Note: The grid is based on the chessboard pose detection, separate from the 4-point calib.
            if rvec_ref is not None and tvec_ref is not None and newcameramtx_live is not None:
                draw_3d_grid_on_image(
                    img_to_draw_on=undistorted_frame,
                    rvec_ref=rvec_ref,
                    tvec_ref=tvec_ref,
                    camera_matrix_for_projection=newcameramtx_live,
                    pattern_size_chessboard=(7, 9),
                    square_size_cm=1.0,
                    grid_expansion_cm=(5, 5),
                    line_color=(0, 255, 0),
                    line_thickness=1
                )

            # Add instruction text based on state
            instruction_text = "Robot-Camera Calibration"
            if calibration_state == 1:
                instruction_text = "Step 1/4: Move robot to point 1. Press SPACE."
            elif calibration_state == 2:
                 instruction_text = "Step 1/4: Click on point 1 in the image."
            elif calibration_state == 3:
                 instruction_text = "Step 2/4: Move robot to point 2. Press SPACE."
            elif calibration_state == 4:
                 instruction_text = "Step 2/4: Click on point 2 in the image."
            elif calibration_state == 5:
                 instruction_text = "Step 3/4: Move robot to point 3. Press SPACE."
            elif calibration_state == 6:
                 instruction_text = "Step 3/4: Click on point 3 in the image."
            elif calibration_state == 7:
                 instruction_text = "Step 4/4: Move robot to point 4. Press SPACE."
            elif calibration_state == 8:
                 instruction_text = "Step 4/4: Click on point 4 in the image."
            elif calibration_state == 9:
                 instruction_text = "Calculating Transformation..."
            elif calibration_state == 10:
                 instruction_text = f"Calibration Complete. Click pixel to move robot to Z={robot_calib_z:.2f}mm."
            elif calibration_state == 0:
                 instruction_text = "Initializing Robot/Camera..."


            cv2.putText(undistorted_frame, instruction_text, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)


            # Display the undistorted (with grid) view
            cv2.imshow('Undistorted Live View - Calibration', undistorted_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord('q'): # ESC or 'q' to exit
                print("Exiting...")
                break
            # Handle space bar presses for states 1, 3, 5, 7 to confirm robot position
            elif key == ord(' ') and calibration_state in [1, 3, 5, 7]:
                if calibration_state == 1:
                    print("Robot at Point 1 confirmed. Ready for pixel click.")
                    calibration_state = 2 # Move to state: ready for Point 1 click
                elif calibration_state == 3:
                     print("Robot at Point 2 confirmed. Ready for pixel click.")
                     calibration_state = 4 # Move to state: ready for Point 2 click
                elif calibration_state == 5:
                     print("Robot at Point 3 confirmed. Ready for pixel click.")
                     calibration_state = 6 # Move to state: ready for Point 3 click
                elif calibration_state == 7:
                     print("Robot at Point 4 confirmed. Ready for pixel click.")
                     calibration_state = 8 # Move to state: ready for Point 4 click


    finally:
        print("Releasing camera and disabling robot.")
        cam.release()
        cv2.destroyAllWindows()
        if 'dashboard' in globals() and dashboard is not None:
            try:
                 dashboard.DisableRobot()
                 print("Robot disabled.")
            except Exception as e:
                 print(f"Error disabling robot during cleanup: {e}")
        print("Cleanup complete.")


if __name__ == "__main__":
    main()