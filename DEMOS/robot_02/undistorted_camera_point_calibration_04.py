from camera_module import CameraModule
from calibration_module import CalibrationModule
from robot_control_module import RobotControlModule
import cv2
import time

def main():
    # Initialize modules
    robot_control = RobotControlModule()
    if not robot_control.initialize_robot():
        return
    camera = CameraModule()
    if not camera.initialize_camera():
        robot_control.cleanup()
        return
    camera.load_calibration_data()
    camera.setup_undistortion()
    if camera.mtx_calib is not None and camera.dist_calib is not None:
        camera.get_reference_pose()
    calibration = CalibrationModule(robot_control)
    window_name = camera.setup_display()
    cv2.setMouseCallback(window_name, calibration.mouse_callback, camera)
    print("\n--- Four-Point Robot-Camera Calibration ---")
    print("Calibration Step 1: Move robot to first point. Press SPACE.")
    calibration.calibration_state = 1

    # Main loop
    try:
        while True:
            frame = camera.capture_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            undistorted_frame = camera.undistort_frame(frame)
            undistorted_frame = camera.draw_visual_elements(
                undistorted_frame,
                calibration.calibration_points_pixel,
                calibration.calibration_state,
                calibration.robot_calib_z
            )
            cv2.imshow(window_name, undistorted_frame)
            key = cv2.waitKey(1) & 0xFF
            if not calibration.handle_key_press(key):
                break
            if calibration.calibration_state == 10:
                break
    finally:
        camera.cleanup()
        robot_control.cleanup()

if __name__ == "__main__":
    main()