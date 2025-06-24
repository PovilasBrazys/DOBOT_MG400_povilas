import cv2
import time
from camera_module import CameraModule
from calibration_module import CalibrationModule
from robot_control_module import RobotControlModule
from module.settings import session_folder

def main():
    robot_control = RobotControlModule()
    if not robot_control.initialize_robot():
        print("Failed to initialize robot. Exiting.")
        return
    camera = CameraModule(session_folder)
    if not camera.initialize_camera():
        robot_control.cleanup()
        print("Failed to initialize camera. Exiting.")
        return
    camera.load_calibration_data()
    camera.setup_undistortion()
    if camera.mtx_calib is not None and camera.dist_calib is not None:
        camera.get_reference_pose()
    calibration = CalibrationModule(robot_control)
    if not calibration.load_calibration_data():
        print("Failed to load calibration data. Run calibration script first.")
        camera.cleanup()
        robot_control.cleanup()
        return
    calibration.calibration_state = 10
    window_name = camera.setup_display(window_name='Undistorted Live View - Interface')
    cv2.setMouseCallback(window_name, calibration.mouse_callback, param=camera)
    print("\n--- Robot-Camera Interface ---")
    print(f"Calibration loaded. Click on the image to move the robot to Z={calibration.robot_calib_z:.2f}mm.")
    print("Test clicks along Y-axis (e.g., 6 grid places apart) to diagnose Y-alignment.")
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
            if key == 27 or key == ord('q'):
                print("Exiting...")
                break
    finally:
        camera.cleanup()
        robot_control.cleanup()

if __name__ == "__main__":
    main()