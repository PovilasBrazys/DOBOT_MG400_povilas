import math

import cv2
import numpy as np
import os
import threading
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from time import sleep
from robot_control_module import ConnectRobot, RunPoint, WaitArrive, GetFeed, ClearRobotError


from module.settings import session_folder, calibration_capture_folder

def live_camera():
    # Construct paths relative to the script's directory, including the session folder
    # NOTE: Update 'session_20250623_203239' if you are using a different calibration session
    script_dir = os.path.dirname(os.path.abspath(__file__))
     # <--- Specify the session folder name here
    calibration_npz_file = os.path.join(script_dir, 'calibration_images', session_folder, 'calibration_data.npz')
    # Path to the distorted image of the chessboard that defines the grid's reference plane
    reference_chessboard_image_path = os.path.join(script_dir, 'calibration_images', session_folder, 'image001.jpg')

    # Chessboard parameters (as per your description: 8x10cm board = 7x9 internal corners, 1cm squares)
    chessboard_pattern_size = (7, 9)  # (cols-1, rows-1) for internal corners
    chessboard_square_size_cm = 1.0  # Each square is 1cm x 1cm

    # --- 1. Load Camera Calibration Data ---
    if not os.path.exists(calibration_npz_file):
        print(f"FATAL ERROR: Calibration file '{calibration_npz_file}' not found.")
        print("Please ensure the calibration file is present in the correct location.")
        return
    try:
        calib_data = np.load(calibration_npz_file)
        # Load using the correct keys saved by generate_undistortion_file.py
        mtx_calib, dist_calib = calib_data['camera_matrix'], calib_data['dist_coeffs']
        print("Calibration data (camera_matrix, dist_coeffs) loaded successfully.")
    except Exception as e:
        print(f"FATAL ERROR: Could not load mtx/dist from '{calibration_npz_file}': {e}")
        return

    # --- 2. Get Reference Pose from the specified Chessboard Image ---
    # This pose (rvec_ref, tvec_ref) will "anchor" our 1cm grid in the 3D world,
    # based on the chessboard's position in 'image1.jpg'.
    print(f"Attempting to get reference pose from: {os.path.abspath(reference_chessboard_image_path)}")
    rvec_ref, tvec_ref = get_reference_pose(
        mtx_calib, dist_calib,
        reference_chessboard_image_path,
        pattern_size=chessboard_pattern_size,
        square_size_cm=chessboard_square_size_cm
    )

    if rvec_ref is None or tvec_ref is None:
        print("Warning: Could not establish reference pose from the image. The 1cm grid will not be drawn.")
    else:
        print("Reference pose for the 1cm grid acquired successfully.")

    # --- 3. Initialize Camera ---
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_V4L2 on Linux if DSHOW causes issues
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cam.set(cv2.CAP_PROP_FPS, 5)  # As per your original code
    # cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus if desired and supported

    if not cam.isOpened():
        print("Error: Could not open camera.")
        if rvec_ref is None or tvec_ref is None:  # If pose also failed, then exit
            return
        # If pose was ok, maybe continue without live feed or just display static? For now, exit.
        return

    # Get actual frame dimensions from camera (might differ from what was set)
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Live camera operational resolution: {frame_width}x{frame_height}")

    center_x = frame_width // 2
    center_y = frame_height // 2
    # --- 4. Prepare for Undistortion of Live Frames ---
    # Get the optimal new camera matrix for the live view resolution (alpha=0 to keep all pixels)
    newcameramtx_live, roi_live = cv2.getOptimalNewCameraMatrix(
        mtx_calib, dist_calib, (frame_width, frame_height), 0, (frame_width, frame_height)
    )
    # Pre-compute the undistortion maps for live frames for efficiency
    mapx_live, mapy_live = cv2.initUndistortRectifyMap(
        mtx_calib, dist_calib, None, newcameramtx_live, (frame_width, frame_height), cv2.CV_32FC1
    )


    # --- 5. Setup Display Windows ---
    cv2.namedWindow('Original Live View', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Undistorted Live View with 1cm Grid', cv2.WINDOW_NORMAL)
    # For fullscreen, uncomment and test:
    cv2.setWindowProperty('Original Live View', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty('Undistorted Live View with 1cm Grid', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Failed to capture frame from camera.")
                if cv2.waitKey(100) & 0xFF in [27, ord('q')]: break
                continue

            # --- 6. Process and Display Frames ---
            # Undistort the current live frame
            undistorted_frame = cv2.remap(frame, mapx_live, mapy_live, cv2.INTER_LINEAR)

            # ─── Draw center markers ──────────────────────────────────────────────────────
            # On the ORIGINAL frame:
            cv2.drawMarker(frame,
                           (center_x, center_y),
                           color=(0, 0, 255),  # red
                           markerType=cv2.MARKER_CROSS,
                           markerSize=20,
                           thickness=2)

            # On the UNDISTORTED frame:
            cv2.drawMarker(undistorted_frame,
                           (center_x, center_y),
                           color=(0, 0, 255),  # red
                           markerType=cv2.MARKER_CROSS,
                           markerSize=20,
                           thickness=2)

            # If a reference pose was successfully obtained, draw the 3D grid
            if rvec_ref is not None and tvec_ref is not None:
                draw_3d_grid_on_image(
                    img_to_draw_on=undistorted_frame,
                    rvec_ref=rvec_ref,
                    tvec_ref=tvec_ref,
                    camera_matrix_for_projection=newcameramtx_live,  # Use live view's undistorted matrix
                    pattern_size_chessboard=chessboard_pattern_size,
                    square_size_cm=chessboard_square_size_cm,
                    grid_expansion_cm=(3, 3),  # Extend grid 3cm around the 8x10cm board area
                    line_color=(0, 255, 0),  # Green grid lines
                    line_thickness=1
                )

            # Add labels to views
            cv2.putText(frame, "Original Live View", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)  # Green text
            cv2.putText(undistorted_frame, "Undistorted Live View with 1cm Grid", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)  # Green text

            # Display the original and undistorted (with grid) views
            cv2.imshow('Original Live View', frame)
            cv2.imshow('Undistorted Live View with 1cm Grid', undistorted_frame)

            # Exit on ESC or 'q'
            key = cv2.waitKey(1) & 0xFF  # Masking for 64-bit systems compatibility
            if key == 27 or key == ord('q'):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
        print("Camera and resources released.")


def true_camPointXY(point, cameraX_offset):
    x = point[0] + cameraX_offset
    y = point[1]
    deg = math.degrees(math.atan(y / x))
    print(deg)

    x1 = cameraX_offset * math.cos(math.radians(deg))
    y1 = cameraX_offset * math.sin(math.radians(deg))
    print("x1:", x1)
    print("y1:", y1)
    x2 = x - x1
    y2 = y - y1

    print("x2:", x2)
    print("y2:", y2)
    point[0] = x2
    point[1] = y2
    return point

# Paths
calibration_dir = calibration_capture_folder + session_folder
image_path = os.path.join(calibration_dir, 'image001.jpg')
calibration_file = os.path.join(calibration_dir, 'calibration_data.npz')


# Verify file paths
print(f"Checking image path: {os.path.exists(image_path)}")
print(f"Checking calibration file path: {os.path.exists(calibration_file)}")
if not os.path.exists(calibration_file):
    print(f"Error: Calibration file {calibration_file} not found.")
    exit()

# Load calibration data
with np.load(calibration_file) as data:
    mtx = data['camera_matrix']
    dist = data['dist_coeffs']
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)

# Load image
img = cv2.imread(image_path)
if img is None:
    print(f"Error: Could not read image {image_path}")
    exit()

# Get image dimensions (correctly interpret height and width)
h, w = img.shape[:2]  # shape returns (height, width)
print(f"Original image dimensions: {w}x{h}")

# Get optimal new camera matrix with alpha=0 to retain all pixels
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
print(f"ROI (x, y, w, h): {roi}")

# Undistort the image with correct dimensions
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
print(f"Undistorted image dimensions: {dst.shape[:2]}")

# Initialize robot
dashboard, move, feed = ConnectRobot()
print("Starting robot enable...")
dashboard.EnableRobot()
print("Robot enabled.")
feed_thread = threading.Thread(target=GetFeed, args=(feed,))
feed_thread.setDaemon(True)
feed_thread.start()
feed_thread1 = threading.Thread(target=ClearRobotError, args=(dashboard,))
feed_thread1.setDaemon(True)
feed_thread1.start()
dashboard.SpeedFactor(10)

# setZ for camera calibration
setZ = 0
# setZ=0
cameraX_offset = 53
Y_offset = 1.5
X_offset = -7

point_a = true_camPointXY([300, 0, setZ, 0], cameraX_offset)

move.MovL(point_a[0], point_a[1], point_a[2], point_a[3])
WaitArrive(point_a)



# Check if dimensions are swapped and correct orientation if needed
if dst.shape[:2] != (h, w):
    print("Warning: Undistorted image dimensions are transposed. Rotating to match original orientation.")
    dst = cv2.rotate(dst, cv2.ROTATE_90_CLOCKWISE)  # Adjust rotation as needed
    print(f"Corrected undistorted image dimensions: {dst.shape[:2]}")

# Display the undistorted image
cv2.namedWindow('Undistorted Image', cv2.WINDOW_NORMAL)
cv2.imshow('Undistorted Image', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- Chessboard detection and measurement ---
# Define the dimensions of the chessboard (number of inner corners)
# For a 9x7 grid, there are 9-1=8 horizontal corners and 7-1=6 vertical corners
chessboard_size = (9, 7)
# Define the physical size of each square in cm
square_size_cm = 1.0

# Convert the undistorted image to grayscale
gray_undistorted = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

# Find the chessboard corners
# flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE is often helpful
ret, corners = cv2.findChessboardCorners(gray_undistorted, chessboard_size, None, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

# If corners are found, proceed with measurement
if ret:
    print("Chessboard corners found.")

    # Refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cv2.cornerSubPix(gray_undistorted, corners, (11, 11), (-1, -1), criteria)

    # Draw and display corners on the undistorted image
    img_with_corners = dst.copy()
    cv2.drawChessboardCorners(img_with_corners, chessboard_size, corners, ret)
    cv2.imshow('Undistorted Image with Corners', img_with_corners)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Calculate the physical dimensions of the view
    # The pattern covers (chessboard_size[0] - 1) * square_size_cm in width
    # and (chessboard_size[1] - 1) * square_size_cm in height.
    # In pixel terms, the pattern spans from corners[0] to corners[chessboard_size[0] - 1] horizontally
    # and from corners[0] to corners[(chessboard_size[1] - 1) * chessboard_size[0]] vertically.

    # Pixel coordinates of the top-left, top-right, and bottom-left corners of the *inner* pattern
    top_left_pixel = corners[0][0]
    top_right_pixel = corners[chessboard_size[0] - 1][0]
    bottom_left_pixel = corners[(chessboard_size[1] - 1) * chessboard_size[0]][0]

    # Calculate pixel distances covered by the known physical size
    pixel_width_of_pattern = np.linalg.norm(top_right_pixel - top_left_pixel)
    pixel_height_of_pattern = np.linalg.norm(bottom_left_pixel - top_left_pixel)

    # Known physical width and height covered by the pattern
    physical_width_of_pattern_cm = (chessboard_size[0] - 1) * square_size_cm # 8 * 1cm = 8cm
    physical_height_of_pattern_cm = (chessboard_size[1] - 1) * square_size_cm # 6 * 1cm = 6cm

    # Calculate the pixel per cm ratio in both directions
    pixels_per_cm_x = pixel_width_of_pattern / physical_width_of_pattern_cm
    pixels_per_cm_y = pixel_height_of_pattern / physical_height_of_pattern_cm

    # Get the dimensions of the undistorted image
    img_h, img_w = dst.shape[:2] # height, width

    # Calculate the total physical size of the view in cm
    view_width_cm = img_w / pixels_per_cm_x
    view_height_cm = img_h / pixels_per_cm_y




    print(f"\nEstimated physical size of the camera view at the calibration plane:")
    print(f"  Width: {view_width_cm:.2f} cm")
    print(f"  Height: {view_height_cm:.2f} cm")

    # Live undistorted view loop should be here


    # Define the mouse callback function
    def mouse_callback(event, u, v, flags, param):
        # Access the necessary variables (read-only)
        global view_width_cm, view_height_cm, pixels_per_cm_x, pixels_per_cm_y, img_w, img_h, dst
        if event == cv2.EVENT_LBUTTONDOWN:
            if pixels_per_cm_x is not None and pixels_per_cm_y is not None:
                # Calculate pixel offset from center (image center is at img_w/2, img_h/2)
                # u, v are pixel coordinates from top-left corner
                pixel_offset_u = u - img_w / 2.0
                pixel_offset_v = v - img_h / 2.0

                # Convert pixel offset to cm offset
                # User coordinate system: X+ top, X- bottom; Y+ left, Y- right
                # Image v increases down, so for robot X (vertical), we negate the v offset
                # Image u increases right, so for robot Y (horizontal), we negate the u offset
                robot_x_cm = -pixel_offset_v / pixels_per_cm_y
                robot_y_cm = -pixel_offset_u / pixels_per_cm_x

                # Display coordinates on the image
                # Create a temporary copy to draw on
                img_display = dst.copy()
                coord_text = f"Robot X: {robot_x_cm:.2f} cm, Robot Y: {robot_y_cm:.2f} cm"


                #point_b = true_camPointXY( [300 + robot_x_cm * 10, robot_y_cm * 10, setZ, 0], cameraX_offset)
                point_b = [300+cameraX_offset + robot_x_cm * 10 + X_offset, robot_y_cm * 10 + Y_offset, -190, 0]
                print(point_b[0])
                print(point_b[1])
                print(robot_x_cm)
                print(robot_y_cm)

                move.MovL(point_b[0], point_b[1], point_b[2], point_b[3])
                WaitArrive(point_b)
                cv2.putText(img_display, coord_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


                # Draw a crosshair at the image center
                center_x, center_y = img_w // 2, img_h // 2
                cv2.line(img_display, (center_x - 10, center_y), (center_x + 10, center_y), (0, 0, 255), 1)
                cv2.line(img_display, (center_x, center_y - 10), (center_x, center_y + 10), (0, 0, 255), 1)

                # Draw markers at the origin (center of the view)
                cv2.circle(img_display, (center_x, center_y), 5, (255, 0, 0), -1) # Blue circle at origin

                cv2.imshow('Undistorted Image with Coordinates', img_display)

        if event == cv2.EVENT_MOUSEMOVE:
            if pixels_per_cm_x is not None and pixels_per_cm_y is not None:
                # Calculate pixel offset from center (image center is at img_w/2, img_h/2)
                # u, v are pixel coordinates from top-left corner
                pixel_offset_u = u - img_w / 2.0
                pixel_offset_v = v - img_h / 2.0

                # Convert pixel offset to cm offset
                # User coordinate system: X+ top, X- bottom; Y+ left, Y- right
                # Image v increases down, so for robot X (vertical), we negate the v offset
                # Image u increases right, so for robot Y (horizontal), we negate the u offset
                robot_x_cm = -pixel_offset_v / pixels_per_cm_y
                robot_y_cm = -pixel_offset_u / pixels_per_cm_x

                # Display coordinates on the image
                # Create a temporary copy to draw on
                img_display = dst.copy()
                coord_text = f"Robot X: {robot_x_cm:.2f} cm, Robot Y: {robot_y_cm:.2f} cm"
                cv2.putText(img_display, coord_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


                # Draw a crosshair at the image center
                center_x, center_y = img_w // 2, img_h // 2
                cv2.line(img_display, (center_x - 10, center_y), (center_x + 10, center_y), (0, 0, 255), 1)
                cv2.line(img_display, (center_x, center_y - 10), (center_x, center_y + 10), (0, 0, 255), 1)

                # Draw markers at the origin (center of the view)
                cv2.circle(img_display, (center_x, center_y), 5, (255, 0, 0), -1) # Blue circle at origin

                cv2.imshow('Undistorted Image with Coordinates', img_display)

    # Create a window and set the mouse callback
    window_name = 'Undistorted Image with Coordinates'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    # Display the initial image (without coordinates until mouse moves)
    cv2.imshow(window_name, dst)

    print("Move the mouse over the image to see coordinates in cm.")

else:
    print("Could not find chessboard corners in the undistorted image.")

# Keep the window open until a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()