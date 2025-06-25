import cv2
import numpy as np
import os
from datetime import datetime

# Define session folder (replace if different)
session_folder = 'session_20250517_054446'

# Paths
calibration_dir = os.path.join(r'd:\\Dobot\\Dobot_projektai\\DOBOT_MG400_povilas\\DEMOS\\robot_02copy\\calibration_images', session_folder)
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

# Save a copy of the original image for reference
cv2.imwrite(original_image_copy, img)
log(f"Original image copied to {original_image_copy}")

# Undistort the image
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))  # alpha=1 for full image
log(f"Optimal new camera matrix:\n{newcameramtx}")
log(f"ROI (x, y, w, h): {roi}")
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
log(f"Undistorted image dimensions: {dst.shape[:2]}")

# Save the undistorted image
cv2.imwrite(output_image_path, dst)
log(f"Undistorted image saved to {output_image_path}")

# Display original and undistorted images
cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Undistorted Image', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Original Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty('Undistorted Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow('Original Image', img)
cv2.imshow('Undistorted Image', dst)
log("Displaying original and undistorted images in fullscreen. Press any key to exit.")
cv2.waitKey(0)
cv2.destroyAllWindows()

# Write log file
with open(log_file, 'w') as f:
    f.write('\n'.join(log_messages))
log(f"Log file saved to {log_file}")

# Optional: Test chessboard corner detection to verify calibration compatibility
pattern_size = (9, 7)  # Match original calibration (9x7 internal corners)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret_corners, corners = cv2.findChessboardCorners(gray, pattern_size, None)
if ret_corners:
    log("Chessboard corners detected successfully in image001.jpg.")
else:
    log("Error: Chessboard corners not detected in image001.jpg. Calibration may be incompatible.")