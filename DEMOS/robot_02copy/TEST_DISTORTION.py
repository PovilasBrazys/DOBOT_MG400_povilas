import cv2
import numpy as np
import os

# --- Configuration ---
# Define paths to calibration data and image
calibration_dir = 'd:\\Dobot\\Dobot_projektai\\DOBOT_MG400_povilas\\DEMOS\\robot_02copy\\calibration_images\\session_20250517_054446'
image_path = os.path.join(calibration_dir, 'image002.jpg')  # Path to distorted image
calibration_file = os.path.join(calibration_dir, 'calibration_data.npz')  # Camera calibration data

# --- 1. Load Calibration Data ---
# Check if calibration file exists
if not os.path.exists(calibration_file):
    raise FileNotFoundError(f"Calibration file not found: {calibration_file}")

# Load camera matrix and distortion coefficients from calibration file
calib = np.load(calibration_file)
mtx = calib['camera_matrix']  # 3x3 camera matrix (fx, fy, cx, cy)
dist = calib['dist_coeffs']   # Lens distortion coefficients (k1, k2, p1, p2, k3)
print("Loaded calibration matrix and distortion coefficients.")

# --- 2. Load Distorted Image ---
img = cv2.imread(image_path)
if img is None:
    raise IOError(f"Could not read image from {image_path}")

# Get image dimensions (height, width)
h, w = img.shape[:2]
print(f"Original image dimensions: {h} (height) x {w} (width)")

# --- 3. Compute Undistortion Map ---
# Calculate optimal new camera matrix and ROI (Region of Interest)
# alpha=1: Keep all pixels (may have black borders)
# alpha=0: Crop to only valid pixels (no borders)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha=1, newImgSize=(w,h))

# Create undistortion maps (for remapping pixels)
# mapx/mapy are 2D arrays that tell where each pixel should move during undistortion
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), cv2.CV_32FC1)

# Apply undistortion using the maps
undistorted = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# --- 4. Optional: Crop valid ROI ---
# roi contains (x, y, width, height) of the valid region after undistortion
x, y, w, h = roi  # Note: w,h here are NEW dimensions, overwriting original w,h
print(f"Valid region after undistortion: {h} (height) x {w} (width)")

# Crop to the valid region (removes black borders)
undistorted_cropped = undistorted[y:y + h, x:x + w]  # Array slicing for cropping

# --- 5. Show Result ---
# Display all versions for comparison
cv2.imshow("Original Image", img)  # Original distorted image
cv2.imshow("Undistorted Image", undistorted)  # Full undistorted image (may have borders)
cv2.imshow("Undistorted + Cropped", undistorted_cropped)  # Clean cropped version
cv2.waitKey(0)  # Wait for key press
cv2.destroyAllWindows()  # Close all windows