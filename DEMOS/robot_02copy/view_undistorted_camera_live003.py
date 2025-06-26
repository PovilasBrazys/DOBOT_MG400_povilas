import cv2
import numpy as np
import os
from module.settings import session_folder, calibration_capture_folder

def get_reference_pose(mtx, dist, chessboard_image_path, pattern_size=(7, 9), square_size_cm=1.0):
    """
    Calculates the rotation and translation vectors (pose) of a chessboard from a static image.
    This pose defines where the Z=0 plane of the grid will be in the world.

    Args:
        mtx (np.ndarray): Camera intrinsic matrix from calibration.
        dist (np.ndarray): Distortion coefficients from calibration.
        chessboard_image_path (str): Path to the image of the chessboard.
        pattern_size (tuple): Number of internal corners (cols, rows), e.g., (7,9) for an 8x10 square board.
        square_size_cm (float): The size of a chessboard square in cm.

    Returns:
        tuple: (rvec, tvec) if pose is successfully found, otherwise (None, None).
               rvec and tvec describe the transformation from the object coordinate system
               (where the chessboard corners are defined in cm) to the camera coordinate system.
    """
    if not os.path.exists(chessboard_image_path):
        print(f"Error: Reference chessboard image not found at '{chessboard_image_path}'")
        return None, None

    img = cv2.imread(chessboard_image_path)
    if img is None:
        print(f"Error: Could not read reference chessboard image from '{chessboard_image_path}'")
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Prepare object points in 3D, in real-world units (cm).
    # For a pattern_size of (7,9), this creates points like:
    # (0,0,0), (1,0,0), ..., (6,0,0)  (first row of internal corners)
    # ...
    # (0,8,0), (1,8,0), ..., (6,8,0)  (last row of internal corners)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size_cm

    # Find the chessboard corners in the reference image
    ret_corners, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret_corners:
        # Refine corner positions for better accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Calculate rotation and translation vectors (rvec, tvec) using solvePnP.
        # These vectors relate the object points (objp) to the image points (corners_refined)
        # using the camera's intrinsic parameters (mtx, dist).
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
    The grid is aligned with the coordinate system of the object used for solvePnP (e.g., chessboard).
    One grid cell corresponds to square_size_cm. The grid will cover the extent of the
    physical chessboard (e.g. an 8x10 board for a 7x9 pattern) plus the specified expansion.

    Args:
        img_to_draw_on: The image (e.g., undistorted camera frame) to draw on.
        rvec_ref: Rotation vector from solvePnP of the reference object.
        tvec_ref: Translation vector from solvePnP of the reference object.
        camera_matrix_for_projection: The camera matrix for cv2.projectPoints (e.g., newcameramtx for undistorted view).
        pattern_size_chessboard: (cols, rows) of internal corners of the reference chessboard (e.g., (7,9)).
        square_size_cm: The size of one square on the reference chessboard in cm (defines grid scale).
        grid_expansion_cm: (extra_x, extra_y) Tuple of how many cm to extend the grid
                           beyond the reference chessboard's physical boundaries.
        line_color: Color of the grid lines.
        line_thickness: Thickness of the grid lines.
    """
    if rvec_ref is None or tvec_ref is None:
        return

    # The objp for solvePnP used internal corners from (0,0,0) to ((cols-1)*sq_cm, (rows-1)*sq_cm, 0).
    # For a standard chessboard, the physical board extends one square beyond these internal corners.
    # E.g., for a 7x9 pattern (8x10 squares), if (0,0,0) is the top-left internal corner,
    # the physical board's top-left corner is at (-1*sq_cm, -1*sq_cm, 0).
    # The board spans:
    # X: from -1*sq_cm to (pattern_size_chessboard[0])*sq_cm
    # Y: from -1*sq_cm to (pattern_size_chessboard[1])*sq_cm

    grid_min_x = -1.0 * square_size_cm - grid_expansion_cm[0]
    grid_max_x = (pattern_size_chessboard[0]) * square_size_cm + grid_expansion_cm[
        0]  # pattern_size[0] is 7, so this goes up to 7*sq_cm (edge of 8th square)

    grid_min_y = -1.0 * square_size_cm - grid_expansion_cm[1]
    grid_max_y = (pattern_size_chessboard[1]) * square_size_cm + grid_expansion_cm[
        1]  # pattern_size[1] is 9, so this goes up to 9*sq_cm (edge of 10th square)

    z_plane = 0.0  # Grid lies on the Z=0 plane of the reference object's coordinate system

    points_3d_lines = []

    # Vertical lines (constant x, y varies)
    current_x = grid_min_x
    while current_x <= grid_max_x + 1e-9:  # Add epsilon for float comparison
        points_3d_lines.append(
            np.array([[current_x, grid_min_y, z_plane], [current_x, grid_max_y, z_plane]], dtype=np.float32))
        current_x += square_size_cm

    # Horizontal lines (constant y, x varies)
    current_y = grid_min_y
    while current_y <= grid_max_y + 1e-9:  # Add epsilon for float comparison
        points_3d_lines.append(
            np.array([[grid_min_x, current_y, z_plane], [grid_max_x, current_y, z_plane]], dtype=np.float32))
        current_y += square_size_cm

    # Project and draw each line segment
    for line_seg_3d in points_3d_lines:
        # Project 3D points to 2D image plane.
        # We pass None for distortion coefficients because camera_matrix_for_projection (newcameramtx_live)
        # already defines an undistorted image space.
        points_2d_segment, _ = cv2.projectPoints(line_seg_3d, rvec_ref, tvec_ref, camera_matrix_for_projection, None)

        if points_2d_segment is not None:
            # Ensure points are integers for drawing
            pt1 = (int(round(points_2d_segment[0][0][0])), int(round(points_2d_segment[0][0][1])))
            pt2 = (int(round(points_2d_segment[1][0][0])), int(round(points_2d_segment[1][0][1])))

            cv2.line(img_to_draw_on, pt1, pt2, line_color, line_thickness)


def main():
    # Construct paths relative to the script's directory, including the session folder
    # NOTE: Update 'session_20250623_203239' if you are using a different calibration session
    script_dir = os.path.dirname(os.path.abspath(__file__))
    calibration_npz_file = os.path.join(script_dir, calibration_capture_folder, session_folder, 'calibration_data.npz')
    reference_chessboard_image_path = os.path.join(script_dir, calibration_capture_folder, session_folder, 'image001.jpg')

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


if __name__ == "__main__":
    # A note on paths:
    # 'calibration_data.npz' is expected in the same directory where this script is run.
    # '../calibration_photos/image1.jpg' means the 'calibration_photos' directory is one level
    # UP from where the script is run, and then down into 'calibration_photos'.
    # Example: If script is in 'project/my_code/', then 'calibration_photos' should be 'project/calibration_photos/'.
    # Adjust the 'reference_chessboard_image_path' if your directory structure is different.

    main()