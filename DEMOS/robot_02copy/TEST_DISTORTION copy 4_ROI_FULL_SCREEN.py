import cv2
import numpy as np
import os
import threading
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from time import sleep
from robot_control_module import ConnectRobot, RunPoint, WaitArrive, GetFeed, ClearRobotError


from module.settings import session_folder, calibration_capture_folder

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
dashboard.Tool(8)
dashboard.SetTool(8, 53, 0, 0, 0)
dashboard.SpeedFactor(10)


point_a = [300, 0, 0, 0]
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

# Save the undistorted image
output_path = os.path.join(calibration_dir, 'undistorted_image001.jpg')
cv2.imwrite(output_path, dst)
print(f"Undistorted image saved to {output_path}. Open in an image viewer to inspect full resolution.")

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
              
                point_a = [300 + robot_x_cm * 10, 0 + robot_y_cm * 10, 0, 0]
                print(point_a[0])
                print(point_a[1])

                move.MovL(point_a[0], point_a[1], point_a[2], point_a[3])
                WaitArrive(point_a)
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