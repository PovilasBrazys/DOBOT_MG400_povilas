import cv2
import numpy as np
import time
import json
import os

class CalibrationModule:
    def __init__(self, robot_control):
        self.robot_control = robot_control
        self.calibration_state = 0
        self.calibration_points_pixel = []
        self.calibration_points_robot = []
        self.robot_calib_z = None
        self.robot_calib_r = None
        self.pixel_to_robot_transform = None
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

    def mouse_callback(self, event, u, v, flags, param):
        """Handle mouse clicks for calibration and robot movement."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        print(f"Clicked pixel: ({u}, {v})")
        if self.calibration_state in [2, 4, 6, 8]:
            time.sleep(0.1)  # Delay to ensure robot has settled
            pose = self.robot_control.get_pose()
            if pose is not None:
                self.calibration_points_robot.append(pose[:3])
                self.calibration_points_pixel.append((u, v))
                if self.calibration_state == 2:
                    self.robot_calib_z = pose[2]
                    self.robot_calib_r = pose[3]
                print(f"Point {len(self.calibration_points_pixel)} recorded: Robot ({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}) -> Pixel ({u}, {v})")
                self.transition_to_next_state()
            else:
                print("Could not get robot pose.")
        elif self.calibration_state == 10:
            if self.pixel_to_robot_transform is not None and self.robot_calib_z is not None and self.robot_calib_r is not None:
                target_x, target_y = self.apply_pixel_to_robot_transform(u, v)
                if target_x is not None and target_y is not None:
                    target_pose = [target_x, target_y, self.robot_calib_z, self.robot_calib_r]
                    print(f"Moving robot to: {target_pose}")
                    try:
                        self.robot_control.move_to_pose(target_pose)
                        print("Robot arrived at target.")
                    except Exception as e:
                        print(f"Error commanding robot movement: {e}")
                else:
                    print("Transformation failed.")
            else:
                print("Calibration incomplete.")

    def transition_to_next_state(self):
        """Transition to the next calibration state."""
        if len(self.calibration_points_pixel) == 1:
            self.calibration_state = 3
            print("Calibration Step 2: Move robot to the second point.")
        elif len(self.calibration_points_pixel) == 2:
            self.calibration_state = 5
            print("Calibration Step 3: Move robot to the third point.")
        elif len(self.calibration_points_pixel) == 3:
            self.calibration_state = 7
            print("Calibration Step 4: Move robot to the fourth point.")
        elif len(self.calibration_points_pixel) == 4:
            z_heights = [p[2] for p in self.calibration_points_robot]
            max_z_diff = max([abs(z - self.robot_calib_z) for z in z_heights]) if z_heights else 0
            if max_z_diff > 3.0:
                print(f"Warning: Z height difference is {max_z_diff:.2f}mm.")
            self.calibration_state = 9
            print("Calculating transformation...")
            self.calculate_pixel_to_robot_transform()
            if self.pixel_to_robot_transform is not None:
                self.save_calibration_data()
                self.calibration_state = 10
                print("Calibration complete. Click to move robot.")
            else:
                self.calibration_state = 0
                self.calibration_points_pixel = []
                self.calibration_points_robot = []
                print("Failed to calculate transformation matrix.")

    def calculate_pixel_to_robot_transform(self):
        """Calculate the perspective transformation matrix."""
        if len(self.calibration_points_pixel) != 4 or len(self.calibration_points_robot) != 4:
            print("Error: Need exactly four pairs of calibration points.")
            self.pixel_to_robot_transform = None
            return
        src_pts = np.array(self.calibration_points_pixel, dtype=np.float32)
        dst_pts = np.array([p[:2] for p in self.calibration_points_robot], dtype=np.float32)
        M, _ = cv2.findHomography(src_pts, dst_pts)
        if M is not None:
            self.pixel_to_robot_transform = M
            print("Calculated Pixel to Robot Transformation Matrix:")
            print(self.pixel_to_robot_transform)
        else:
            print("Error: cv2.findHomography failed.")
            self.pixel_to_robot_transform = None

    def apply_pixel_to_robot_transform(self, u, v):
        """Apply transformation to convert pixel coordinates to robot coordinates."""
        if self.pixel_to_robot_transform is None:
            print("Error: Transformation matrix not calculated.")
            return None, None
        src_point = np.array([[[u, v]]], dtype=np.float32)
        dst_point = cv2.perspectiveTransform(src_point, self.pixel_to_robot_transform)
        if dst_point is not None and dst_point.shape == (1, 1, 2):
            return dst_point[0, 0, 0], dst_point[0, 0, 1]
        print("Error applying perspective transform.")
        return None, None

    def save_calibration_data(self):
        """Save calibration data to a JSON file."""
        data = {
            'calibration_points_pixel': self.calibration_points_pixel,
            'calibration_points_robot': self.calibration_points_robot,
            'pixel_to_robot_transform': self.pixel_to_robot_transform.tolist() if self.pixel_to_robot_transform is not None else None,
            'robot_calib_z': self.robot_calib_z,
            'robot_calib_r': self.robot_calib_r
        }
        output_file = os.path.join(self.script_dir, 'calibration_data.json')
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Calibration data saved to {output_file}")

    def load_calibration_data(self):
        """Load calibration data from a JSON file."""
        input_file = os.path.join(self.script_dir, 'calibration_data.json')
        if not os.path.exists(input_file):
            print(f"Error: Calibration data file '{input_file}' not found.")
            return False
        with open(input_file, 'r') as f:
            data = json.load(f)
        self.calibration_points_pixel = [tuple(p) for p in data['calibration_points_pixel']]
        self.calibration_points_robot = data['calibration_points_robot']
        self.pixel_to_robot_transform = np.array(data['pixel_to_robot_transform']) if data['pixel_to_robot_transform'] else None
        self.robot_calib_z = data['robot_calib_z']
        self.robot_calib_r = data['robot_calib_r']
        print(f"Calibration data loaded from {input_file}")
        return True

    def handle_key_press(self, key):
        """Handle key presses for state transitions."""
        if key == 27 or key == ord('q'):
            print("Exiting...")
            return False
        elif key == ord(' ') and self.calibration_state in [1, 3, 5, 7]:
            self.calibration_state += 1
            print(f"Robot at Point {(self.calibration_state-1)//2} confirmed. Ready for pixel click.")
        return True