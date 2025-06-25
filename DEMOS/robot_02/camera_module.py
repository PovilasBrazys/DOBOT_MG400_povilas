import cv2
import numpy as np
import os
from module.settings import session_folder

class CameraModule:
    def __init__(self, session_folder=session_folder):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.session_folder = session_folder
        self.calibration_npz_file = os.path.join(self.script_dir, 'calibration_images', self.session_folder, 'calibration_data.npz')
        self.reference_chessboard_image_path = os.path.join(self.script_dir, 'calibration_images', self.session_folder, 'image001.jpg')
        self.cam = None
        self.frame_width = None
        self.frame_height = None
        self.center_x = None
        self.center_y = None
        self.mtx_calib = None
        self.dist_calib = None
        self.mapx_live = None
        self.mapy_live = None
        self.newcameramtx_live = None
        self.rvec_ref = None
        self.tvec_ref = None

    def initialize_camera(self):
        """Initialize the camera with specified settings."""
        self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_V4L2 on Linux if needed
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cam.set(cv2.CAP_PROP_FPS, 5)
        self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cam.isOpened():
            print("FATAL ERROR: Could not open camera.")
            return False
        self.frame_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2
        print(f"Live camera resolution: {self.frame_width}x{self.frame_height}")
        return True

    def load_calibration_data(self):
        """Load camera calibration data from NPZ file."""
        if not os.path.exists(self.calibration_npz_file):
            print(f"Warning: Calibration file '{self.calibration_npz_file}' not found.")
            return False
        try:
            calib_data = np.load(self.calibration_npz_file)
            self.mtx_calib, self.dist_calib = calib_data['camera_matrix'], calib_data['dist_coeffs']
            print("Camera calibration data loaded.")
            return True
        except Exception as e:
            print(f"Warning: Could not load calibration data: {e}")
            return False

    def setup_undistortion(self):
        """Prepare undistortion maps for live frames."""
        if self.mtx_calib is not None and self.dist_calib is not None:
            self.newcameramtx_live, _ = cv2.getOptimalNewCameraMatrix(
                self.mtx_calib, self.dist_calib, (self.frame_width, self.frame_height), 0
            )
            self.mapx_live, self.mapy_live = cv2.initUndistortRectifyMap(
                self.mtx_calib, self.dist_calib, None, self.newcameramtx_live, (self.frame_width, self.frame_height), cv2.CV_32FC1
            )
            print("Undistortion maps generated.")
            return True
        print("Skipping undistortion setup (calibration data not available).")
        return False

    def get_reference_pose(self, pattern_size=(7, 9), square_size_cm=1.0):
        """Calculate reference pose for grid drawing."""
        if not os.path.exists(self.reference_chessboard_image_path):
            print(f"Error: Reference chessboard image not found at '{self.reference_chessboard_image_path}'")
            return False
        img = cv2.imread(self.reference_chessboard_image_path)
        if img is None:
            print(f"Error: Could not read reference chessboard image.")
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size_cm
        ret_corners, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret_corners:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            ret_pnp, self.rvec_ref, self.tvec_ref = cv2.solvePnP(objp, corners_refined, self.mtx_calib, self.dist_calib)
            if ret_pnp:
                print("Reference pose acquired successfully.")
                return True
            print("Error: solvePnP failed.")
            return False
        print("Error: Chessboard corners not found.")
        return False

    def capture_frame(self):
        """Capture and return a frame from the camera."""
        ret, frame = self.cam.read()
        if not ret:
            print("Failed to capture frame from camera.")
            return None
        return frame

    def undistort_frame(self, frame):
        """Apply undistortion to the frame if available."""
        if self.mapx_live is not None and self.mapy_live is not None:
            return cv2.remap(frame, self.mapx_live, self.mapy_live, cv2.INTER_LINEAR)
        return frame

    def draw_visual_elements(self, frame, calibration_points_pixel, calibration_state, robot_calib_z):
        """Draw center marker, calibration points, and grid on the frame."""
        # Draw center marker
        cv2.drawMarker(frame, (self.center_x, self.center_y), color=(0, 0, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        # Draw calibration points
        colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0)]
        for i, (u, v) in enumerate(calibration_points_pixel):
            color = colors[i] if i < len(colors) else (255, 255, 255)
            cv2.circle(frame, (u, v), 8, color, -1)
            cv2.putText(frame, f"P{i+1}", (u + 15, v + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        # Draw 3D grid if reference pose is available
        if self.rvec_ref is not None and self.tvec_ref is not None and self.newcameramtx_live is not None:
            self.draw_3d_grid(frame)
        # Add instruction text
        instruction_text = {
            1: "Step 1/4: Move robot to point 1. Press SPACE.",
            2: "Step 1/4: Click on point 1 in the image.",
            3: "Step 2/4: Move robot to point 2. Press SPACE.",
            4: "Step 2/4: Click on point 2 in the image.",
            5: "Step 3/4: Move robot to point 3. Press SPACE.",
            6: "Step 3/4: Click on point 3 in the image.",
            7: "Step 4/4: Move robot to point 4. Press SPACE.",
            8: "Step 4/4: Click on point 4 in the image.",
            9: "Calculating Transformation...",
            10: f"Calibration Complete. Click pixel to move robot to Z={robot_calib_z:.2f}mm" if robot_calib_z else "Calibration Complete.",
            0: "Initializing Robot/Camera..."
        }.get(calibration_state, "Robot-Camera Calibration")
        cv2.putText(frame, instruction_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        return frame

    def draw_3d_grid(self, img_to_draw_on, pattern_size=(7, 9), square_size_cm=1.0, grid_expansion_cm=(5, 5)):
        """Draw a 3D grid on the image."""
        if self.rvec_ref is None or self.tvec_ref is None:
            return
        grid_min_x = -1.0 * square_size_cm - grid_expansion_cm[0]
        grid_max_x = pattern_size[0] * square_size_cm + grid_expansion_cm[0]
        grid_min_y = -1.0 * square_size_cm - grid_expansion_cm[1]
        grid_max_y = pattern_size[1] * square_size_cm + grid_expansion_cm[1]
        z_plane = 0.0
        points_3d_lines = []
        current_x = grid_min_x
        while current_x <= grid_max_x + 1e-9:
            points_3d_lines.append(np.array([[current_x, grid_min_y, z_plane], [current_x, grid_max_y, z_plane]], dtype=np.float32))
            current_x += square_size_cm
        current_y = grid_min_y
        while current_y <= grid_max_y + 1e-9:
            points_3d_lines.append(np.array([[grid_min_x, current_y, z_plane], [grid_max_x, current_y, z_plane]], dtype=np.float32))
            current_y += square_size_cm
        for line_seg_3d in points_3d_lines:
            points_2d_segment, _ = cv2.projectPoints(line_seg_3d, self.rvec_ref, self.tvec_ref, self.newcameramtx_live, None)
            if points_2d_segment is not None:
                pt1 = (int(round(points_2d_segment[0][0][0])), int(round(points_2d_segment[0][0][1])))
                pt2 = (int(round(points_2d_segment[1][0][0])), int(round(points_2d_segment[1][0][1])))
                cv2.line(img_to_draw_on, pt1, pt2, (0, 255, 0), 1)

    def setup_display(self, window_name='Undistorted Live View - Calibration'):
        """Setup the display window."""
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        return window_name

    def cleanup(self):
        """Release camera and destroy windows."""
        if self.cam is not None:
            self.cam.release()
        cv2.destroyAllWindows()
        print("Camera cleanup complete.")