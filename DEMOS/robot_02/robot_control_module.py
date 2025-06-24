import sys
import threading
import re
import time
import os

# Add parent directory to sys.path to import robot_control
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from robot_control import ConnectRobot, RunPoint, WaitArrive, GetFeed, ClearRobotError

class RobotControlModule:
    def __init__(self):
        self.dashboard = None
        self.move = None
        self.feed = None

    def initialize_robot(self):
        """Initialize and connect to the robot."""
        print("Connecting to robot...")
        try:
            self.dashboard, self.move, self.feed = ConnectRobot()
            print("Robot connected.")
            self.dashboard.EnableRobot()
            print("Robot enabled!")
            self.move.MovL(330, 0, 0, 0)
            feed_thread = threading.Thread(target=GetFeed, args=(self.feed,))
            feed_thread.setDaemon(True)
            feed_thread.start()
            print("Robot feedback thread started.")
            error_thread = threading.Thread(target=ClearRobotError, args=(self.dashboard,))
            error_thread.setDaemon(True)
            error_thread.start()
            print("Robot error clearing thread started.")
            self.dashboard.Tool(8)
            self.dashboard.SetTool(8, 53, 0, 0, 0)
            self.dashboard.User(0)
            self.dashboard.SpeedFactor(10)
            return True
        except Exception as e:
            print(f"FATAL ERROR: Could not connect or enable robot: {e}")
            return False

    def get_pose(self):
        """Get the current robot pose in User 0, Tool 8."""
        time.sleep(0.1)  # Ensure robot has settled
        pose_response_str = self.dashboard.GetPoseInFrame(0, 8)
        print(f"Robot pose response: {pose_response_str}")
        return self.parse_pose_string(pose_response_str)

    def parse_pose_string(self, pose_str):
        """Parse robot pose string into [X, Y, Z, R, J1, J2]."""
        try:
            match = re.search(r"\{([^}]+)\}", pose_str)
            if match:
                pose_values_str = match.group(1).split(',')
                numbers = [float(n) for n in pose_values_str]
                if len(numbers) >= 6:
                    return numbers[:6]
            else:
                numbers = [float(n) for n in pose_str.split(',')]
                if len(numbers) >= 6:
                    return numbers[:6]
            print(f"Warning: Could not parse pose values from string: {pose_str}")
            return None
        except Exception as e:
            print(f"Error parsing pose string '{pose_str}': {e}")
            return None

    def move_to_pose(self, pose):
        """Move the robot to the specified pose."""
        self.dashboard.User(0)
        self.dashboard.Tool(8)
        RunPoint(self.move, pose)
        WaitArrive(pose)

    def cleanup(self):
        """Disable the robot and perform cleanup."""
        if self.dashboard is not None:
            try:
                self.dashboard.DisableRobot()
                print("Robot disabled.")
            except Exception as e:
                print(f"Error disabling robot: {e}")
        print("Robot cleanup complete.")