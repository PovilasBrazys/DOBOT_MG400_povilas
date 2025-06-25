import cv2
import numpy as np
import os

from module.settings import session_folder, calibration_capture_folder

calibration_dir = calibration_capture_folder
image_path = os.path.join(calibration_dir, 'image001.jpg')
calibration_file = os.path.join(calibration_dir, 'calibration_data.npz')

if not os.path.exists(calibration_file) or not os.path.exists(image_path):
    print("Error: File not found.")
    exit()

with np.load(calibration_file) as data:
    mtx = data['camera_matrix']
    dist = data['dist_coeffs']
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)

img = cv2.imread(image_path)
if img is None:
    print(f"Error: Could not read image {image_path}")
    exit()

h, w = img.shape[:2]
print(f"Original image dimensions: {w}x{h}")

newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
print(f"Undistorted image dimensions: {dst.shape[:2]}")

cv2.namedWindow('Undistorted Image', cv2.WINDOW_NORMAL)
cv2.imshow('Undistorted Image', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

output_path = os.path.join(calibration_dir, 'undistorted_image001.jpg')
cv2.imwrite(output_path, dst)
print(f"Saved to {output_path}.")