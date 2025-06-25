import cv2
import numpy as np
import os
from datetime import datetime

# Define session folder
session_folder = 'session_20250517_054446'

# Paths
calibration_dir = os.path.join('d:\Dobot\\Dobot_projektai\\DOBOT_MG400_povilas\\DEMOS\\robot_02copy\\calibration_images', session_folder)
image_path = os.path.join(calibration_dir, 'image001.jpg')
calibration_file = os.path.join(calibration_dir, 'calibration_data.npz')
log_file = os.path.join(calibration_dir, 'undistort_log.txt')
output_image_path = os.path.join(calibration_dir, 'undistorted_image001.jpg')
original_image_copy = os.path.join(calibration_dir, 'original_image001_copy.jpg')

# Initialize log
log_messages = [f"Undistortion Log - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]

def log(message):
    """Append message to log list and print to console."""
    print(message)
    log_messages.append(message)

# Verify file paths
log(f"Checking image path: {image_path}")
if not os.path.exists(image_path):
    log(f"Error: Image file {image_path} not found.")
    with open(log_file, 'w') as f:
        f.write('\n'.join(log_messages))
    exit()
log(f"Checking calibration file path: {calibration_file}")
if not os.path.exists(calibration_file):
    log(f"Error: Calibration file {calibration_file} not found.")
    with open(log_file, 'w') as f:
        f.write('\n'.join(log_messages))
    exit()

# Load calibration data
try:
    with np.load(calibration_file) as data:
        mtx = data['camera_matrix']
        dist = data['dist_coeffs']
    log("Calibration data loaded successfully.")
    log(f"Camera matrix:\n{mtx}")
    log(f"Distortion coefficients:\n{dist}")
except Exception as e:
    log(f"Error: Could not load calibration data: {e}")
    with open(log_file, 'w') as f:
        f.write('\n'.join(log_messages))
    exit()

# Load image
img = cv2.imread(image_path)
if img is None:
    log(f"Error: Could not read image {image_path}")
    with open(log_file, 'w') as f:
        f.write('\n'.join(log_messages))
    exit()

# Verify image dimensions
h, w = img.shape[:2]
log(f"Original image dimensions: {w}x{h}")
if (w, h) != (1920, 1080):
    log(f"Warning: Image dimensions are {w}x{h}, expected 1920x1080.")

# Save original image copy
cv2.imwrite(original_image_copy, img)
log(f"Original image copied to {original_image_copy}")

# Undistort using cv2.undistort
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))  # alpha=1 for full image
log(f"Optimal new camera matrix:\n{newcameramtx}")
log(f"ROI (x, y, w, h): {roi}")
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
log(f"Undistorted image dimensions (cv2.undistort): {dst.shape[:2]}")

# Alternative undistortion using cv2.remap
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), cv2.CV_32FC1)
dst_remap = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
log(f"Undistorted image dimensions (cv2.remap): {dst_remap.shape[:2]}")

# Save undistorted images
cv2.imwrite(output_image_path, dst)
cv2.imwrite(os.path.join(calibration_dir, 'undistorted_image001_remap.jpg'), dst_remap)
log(f"Undistorted image (cv2.undistort) saved to {output_image_path}")
log(f"Undistorted image (cv2.remap) saved to {calibration_dir}\\undistorted_image001_remap.jpg")

# Test chessboard corner detection
pattern_size = (9, 7)  # 9x7 internal corners (10x8 squares)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret_corners, corners = cv2.findChessboardCorners(gray, pattern_size, None)
if ret_corners:
    log("Chessboard corners detected successfully in image001.jpg with pattern (9x7).")
else:
    log("Error: Chessboard corners not detected in image001.jpg with pattern (9x7). Calibration may be incompatible.")

# Display original and undistorted images
cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Undistorted Image (cv2.undistort)', cv2.WINDOW_NORMAL)
cv2.namedWindow('Undistorted Image (cv2.remap)', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Original Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty('Undistorted Image (cv2.undistort)', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty('Undistorted Image (cv2.remap)', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow('Original Image', img)
cv2.imshow('Undistorted Image (cv2.undistort)', dst)
cv2.imshow('Undistorted Image (cv2.remap)', dst_remap)
log("Displaying original and undistorted images in fullscreen. Press any key to exit.")
cv2.waitKey(0)
cv2.destroyAllWindows()

# Write log file
with open(log_file, 'w') as f:
    f.write('\n'.join(log_messages))
log(f"Log file saved to {log_file}")