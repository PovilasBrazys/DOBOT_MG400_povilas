import cv2
import numpy as np
import glob
import os

from module.settings import session_folder

# Define chessboard parameters
# Make sure these match the chessboard used in your photos
chessboard_width = 9  # number of internal corners in width
chessboard_height = 7 # number of internal corners in height
square_size = 10.0 # size of a chessboard square in your chosen units (e.g., mm)
                   # This is primarily needed for estimating real-world poses (rvecs, tvecs).
                   # For just undistorting the image, the exact size doesn't affect mtx or dist.

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(chessboard_width-1, chessboard_height-1,0)
# These are the 3D points of the chessboard corners in the board's coordinate system (units are square_size)
objp = np.zeros((chessboard_height * chessboard_width, 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_width, 0:chessboard_height].T.reshape(-1, 2) * square_size

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space (the ideal corner locations)
imgpoints = [] # 2d points in image plane (the detected corner locations)

# --- IMPORTANT: SET YOUR IMAGE DIRECTORY HERE ---
# Replace this path with the actual path to your calibration photos.
# Example: 'd:\\Povilas\\GIT_DOBOT\\DOBOT_MG400_povilas\\DEMOS\\robot_02\\calibration_images\\session_20250623_203239'
image_dir = 'd:\\Povilas\\GIT_DOBOT\\DOBOT_MG400_povilas\\DEMOS\\robot_02\\calibration_images\\' + session_folder
# ------------------------------------------------

# Get list of images
# Ensure you are filtering for image files correctly (e.g., '.jpg', '.png')
image_files = glob.glob(os.path.join(image_dir, '*.jpg')) # Change *.jpg if needed

print(f"Looking for calibration images in: {image_dir}")

if not image_files:
    print("Error: No images found matching the pattern *.jpg. Please check the 'image_dir' path and file extension.")
    exit()

# Criteria for sub-pixel corner refinement
# Refines the detected corner locations for better accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

found_images_count = 0
# Store successful image URIs to use one for undistortion example later
successful_image_uris = []

print(f"Processing {len(image_files)} images...")

for fname in image_files:
    img = cv2.imread(fname)
    if img is None:
        print(f"Warning: Could not read image {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    # The last parameter flags can improve detection. CALIB_CB_ADAPTIVE_THRESH is often useful.
    ret, corners = cv2.findChessboardCorners(gray, (chessboard_width, chessboard_height), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK)

    # If found, add object points, then refine and add image points
    if ret:
        objpoints.append(objp)

        # Refine the corner locations to sub-pixel accuracy
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        found_images_count += 1
        successful_image_uris.append(fname) # Store the URI of the image where corners were found
        print(f"Found corners in {os.path.basename(fname)}")
    #else:
        #print(f"Did not find corners in {os.path.basename(fname)}") # Uncomment to see which images failed

print(f"Successfully found corners in {found_images_count} out of {len(image_files)} images processed.")

# Check if enough images were processed
min_successful_images = 10 # A common guideline
if found_images_count < min_successful_images:
    print(f"Warning: Only {found_images_count} images with detectable corners were found. Calibration may be inaccurate. At least {min_successful_images} are generally recommended.")
    if found_images_count == 0:
        print("Error: No corners found in any image. Calibration aborted.")
        exit()


# Perform camera calibration
# gray.shape[::-1] gives the image size (width, height)
# mtx: Camera matrix (intrinsic parameters)
# dist: Distortion coefficients
# rvecs: Rotation vectors for each image (extrinsic parameters)
# tvecs: Translation vectors for each image (extrinsic parameters)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

if ret:
    print("\nCamera calibration successful!")
    print("-----------------------------")
    print("Camera matrix (mtx):\n", mtx)
    print("\nDistortion coefficients (dist):\n", dist)

    # --- Calculate and report Re-projection Error ---
    # This is the average distance between the detected image points and the
    # projected 3D object points after calibration. A lower value is better.
    mean_error = 0
    for i in range(len(objpoints)):
        # Project 3D points back to image plane using the found calibration parameters
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        # Calculate the error (distance) between the re-projected points and the original detected points
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error

    mean_error /= len(objpoints)
    print(f"\nTotal re-projection error: {mean_error:.4f} pixels")
    print("A low error (ideally < 0.5-1.0 pixels) indicates good calibration.")
    print("-----------------------------")


    # --- Save the camera matrix and distortion coefficients ---
    # Saving in .npz format (NumPy) and .yaml format (OpenCV)
    output_npz_file = os.path.join(image_dir, 'calibration_data.npz')
    np.savez(output_npz_file, camera_matrix=mtx, dist_coeffs=dist)
    print(f"\nCalibration data saved to {output_npz_file}")

    # Save as YAML using OpenCV for compatibility with other OpenCV applications (e.g., C++)
    # This requires pyyaml to be installed (pip install pyyaml)
    try:
        output_yaml_file = os.path.join(image_dir, 'calibration_data.yaml')
        fs = cv2.FileStorage(output_yaml_file, cv2.FILE_STORAGE_WRITE)
        fs.write("camera_matrix", mtx)
        fs.write("dist_coeffs", dist)
        fs.release()
        print(f"Calibration data also saved to {output_yaml_file}")
    except ImportError:
        print("\nWarning: PyYAML not found. Skipping saving to .yaml format. Install with 'pip install pyyaml' to enable.")
    except Exception as e:
        print(f"\nWarning: Could not save calibration data to .yaml: {e}")


    # --- Demonstrate Undistortion ---
    if successful_image_uris:
        print("\nDemonstrating undistortion on a sample image...")
        # Choose the first image where corners were successfully found
        sample_image_path = successful_image_uris[0]
        img = cv2.imread(sample_image_path)

        if img is not None:
            h, w = img.shape[:2]

            # Get the optimal new camera matrix and ROI (Region of Interest)
            # This is useful to crop the undistorted image and remove black borders
            # alpha=0 means all pixels are retained, even if remapped from outside the original image
            # alpha=1 means only pixels mapping to the original image area are retained
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

            # Undistort the image using the new camera matrix
            # You can use cv2.undistort() which is simpler but doesn't give ROI
            # Here we use cv2.initUndistortRectifyMap and cv2.remap for more control and potentially higher performance
            mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5) # 5 for CV_32FC1
            dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

            # Crop the image based on the ROI
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]

            output_undistorted_path = os.path.join(image_dir, 'undistorted_example.jpg')
            cv2.imwrite(output_undistorted_path, dst)
            print(f"Undistorted example image saved to {output_undistorted_path}")

            # Optional: Display original and undistorted images
            # cv2.imshow('Original Image', img)
            # cv2.imshow('Undistorted Image', dst)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        else:
            print(f"Warning: Could not read the sample image for undistortion: {sample_image_path}")
    else:
         print("\nCould not demonstrate undistortion as no images with corners were successfully processed.")

else:
    print("\nError: Camera calibration failed. Please check the input images and parameters.")
