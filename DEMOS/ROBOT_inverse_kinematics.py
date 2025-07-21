import sys
import numpy as np
import json
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGridLayout
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ArmControlPanel(QWidget):
    def __init__(self):
        super().__init__()

        self.L1 = 175  # mm
        self.L2 = 241  # mm
        self.tool_offset = 48  # mm (horizontal X offset)

        # Miscalibration parameters
        self.j1_home_offset = 95  # Actual J1 is 95° when display shows 0°
        self.j2_home_offset = -95  # Actual J2 is -95° when display shows 0°

        # Expected angles (what control panel thinks)
        self.theta1_expected = 0
        self.theta2_expected = 0

        # Actual angles (physical reality)
        self.theta1_actual = self.j1_home_offset
        self.theta2_actual = self.j2_home_offset

        # Positions (tracking offset endpoints)
        self.expected_offset_x = 0
        self.expected_offset_z = 0
        self.actual_offset_x = 0
        self.actual_offset_z = 0

        # Trace of actual offset positions
        self.trace_points = []
        self.max_trace_points = 100

        self.elbow_up = False
        self.initUI()
        self.update_kinematics()

    def initUI(self):
        self.setWindowTitle("2-Axis Robot Control (X Tool Offset)")

        main_layout = QHBoxLayout(self)

        # Left panel - controls and status
        control_panel = QWidget()
        layout = QVBoxLayout(control_panel)

        # --- Control buttons ---
        grid = QGridLayout()
        grid.addWidget(QLabel("Joint Control:"), 0, 0, 1, 2)
        grid.addWidget(QPushButton("+J1", clicked=self.increase_j1), 1, 0)
        grid.addWidget(QPushButton("-J1", clicked=self.decrease_j1), 1, 1)
        grid.addWidget(QPushButton("+J2", clicked=self.increase_j2), 2, 0)
        grid.addWidget(QPushButton("-J2", clicked=self.decrease_j2), 2, 1)

        grid.addWidget(QLabel("Cartesian Control:"), 3, 0, 1, 2)
        grid.addWidget(QPushButton("+X", clicked=self.increase_x), 4, 0)
        grid.addWidget(QPushButton("-X", clicked=self.decrease_x), 4, 1)
        grid.addWidget(QPushButton("+Z", clicked=self.increase_z), 5, 0)
        grid.addWidget(QPushButton("-Z", clicked=self.decrease_z), 5, 1)

        grid.addWidget(QPushButton("Flip Elbow", clicked=self.flip_elbow), 6, 0, 1, 2)
        grid.addWidget(QPushButton("Clear Trace", clicked=self.clear_trace), 7, 0, 1, 2)
        grid.addWidget(QPushButton("Generate Z-Scan", clicked=self.generate_z_scan), 8, 0, 1, 2)
        layout.addLayout(grid)

        # --- Status displays ---
        layout.addWidget(QLabel("\nExpected (Panel):"))
        self.label_expected_angles = QLabel()
        self.label_expected_position = QLabel()

        layout.addWidget(QLabel("\nActual (Robot):"))
        self.label_actual_angles = QLabel()
        self.label_actual_position = QLabel()
        self.label_position_error = QLabel()

        layout.addWidget(QLabel("\nMiscalibration:"))
        self.label_miscalibration = QLabel(f"J1 offset: {90 - self.j1_home_offset}° from vertical")
        self.label_tool_offset = QLabel(f"Tool offset: {self.tool_offset}mm in X")

        layout.addWidget(self.label_expected_angles)
        layout.addWidget(self.label_expected_position)
        layout.addWidget(self.label_actual_angles)
        layout.addWidget(self.label_actual_position)
        layout.addWidget(self.label_position_error)
        layout.addWidget(self.label_miscalibration)
        layout.addWidget(self.label_tool_offset)
        layout.addStretch()

        # Right panel - visualization
        vis_panel = QWidget()
        vis_layout = QVBoxLayout(vis_panel)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        vis_layout.addWidget(self.canvas)

        main_layout.addWidget(control_panel, 1)
        main_layout.addWidget(vis_panel, 2)

    def generate_z_scan(self):
        """Generate coordinate array from Z=300 to Z=-150 in 5mm steps"""
        try:
            z_values = np.arange(200.0, -100.0, -5.0)  # From 300 to -150 in -5mm steps
            coordinates = []

            # Get current X position (of the offset point)
            current_x = float(self.actual_offset_x)

            for z in z_values:
                try:
                    # Target is the tool point (offset point minus offset)
                    target_x = current_x - self.tool_offset
                    target_z = float(z)

                    # Calculate inverse kinematics for this position
                    r2 = target_x ** 2 + target_z ** 2
                    r = np.sqrt(r2)

                    if r > (self.L1 + self.L2) or r < abs(self.L1 - self.L2):
                        print(f"Position Z={z} is out of reach")
                        continue

                    cos_theta2 = (r2 - self.L1 ** 2 - self.L2 ** 2) / (2 * self.L1 * self.L2)
                    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
                    theta2 = np.arccos(cos_theta2)

                    if not self.elbow_up:
                        theta2 = -theta2

                    k1 = self.L1 + self.L2 * np.cos(theta2)
                    k2 = self.L2 * np.sin(theta2)
                    theta1 = np.arctan2(target_z, target_x) - np.arctan2(k2, k1)

                    # Calculate actual position with miscalibration
                    theta1_robot = float(np.degrees(theta1)) + self.j1_home_offset
                    theta2_robot = float(np.degrees(theta2)) + self.j2_home_offset

                    # Forward kinematics to get actual position
                    t1 = np.radians(theta1_robot)
                    t2 = np.radians(theta2_robot)
                    actual_x2 = self.L1 * np.cos(t1) + self.L2 * np.cos(t1 + t2)
                    actual_z2 = self.L1 * np.sin(t1) + self.L2 * np.sin(t1 + t2)
                    actual_offset_x = float(actual_x2 + self.tool_offset)
                    actual_offset_z = float(actual_z2)

                    coordinates.append({
                        'commanded_z': float(z),
                        'actual_x': actual_offset_x,
                        'actual_z': actual_offset_z,
                        'theta1': theta1_robot,
                        'theta2': theta2_robot
                    })
                except Exception as e:
                    print(f"Error processing Z={z}: {str(e)}")
                    continue

            # Save to JSON file
            with open('z_scan_coordinates.json', 'w') as f:
                json.dump(coordinates, f, indent=2, ensure_ascii=False)

            print(f"Successfully saved {len(coordinates)} points to z_scan_coordinates.json")
            return coordinates

        except Exception as e:
            print(f"Failed to generate Z-scan: {str(e)}")
            return []

    def clear_trace(self):
        self.trace_points = []
        self.plot_arm()

    def flip_elbow(self):
        self.elbow_up = not self.elbow_up
        self.update_kinematics()

    def update_labels(self):
        self.label_expected_angles.setText(
            f"J1: {self.theta1_expected:.1f}°  J2: {self.theta2_expected:.1f}°"
        )
        self.label_expected_position.setText(
            f"Expected Position: X = {self.expected_offset_x:.1f} mm, Z = {self.expected_offset_z:.1f} mm"
        )

        self.label_actual_angles.setText(
            f"J1: {self.theta1_actual:.1f}°  J2: {self.theta2_actual:.1f}°"
        )
        self.label_actual_position.setText(
            f"Actual Position: X = {self.actual_offset_x:.1f} mm, Z = {self.actual_offset_z:.1f} mm"
        )

        error_x = self.actual_offset_x - self.expected_offset_x
        error_z = self.actual_offset_z - self.expected_offset_z
        self.label_position_error.setText(
            f"Position Error: ΔX = {error_x:.1f} mm, ΔZ = {error_z:.1f} mm"
        )

    def plot_arm(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if len(self.trace_points) > 1:
            trace_x, trace_z = zip(*self.trace_points)
            ax.plot(trace_x, trace_z, 'g:', linewidth=1, alpha=0.5, label='Tool Path')

        # Calculate expected arm positions
        exp_t1 = np.radians(self.theta1_expected + 90)
        exp_t2 = np.radians(self.theta2_expected - 90)
        exp_x1 = self.L1 * np.cos(exp_t1)
        exp_z1 = self.L1 * np.sin(exp_t1)
        exp_x2 = exp_x1 + self.L2 * np.cos(exp_t1 + exp_t2)
        exp_z2 = exp_z1 + self.L2 * np.sin(exp_t1 + exp_t2)

        # Expected offset line (horizontal X)
        exp_offset_x = exp_x2 + self.tool_offset
        exp_offset_z = exp_z2
        ax.plot([exp_x2, exp_offset_x], [exp_z2, exp_offset_z], 'm--', linewidth=2, label='Expected Offset')

        # Plot expected arm
        ax.plot([0, exp_x1, exp_x2], [0, exp_z1, exp_z2], 'r--', linewidth=2, label='Expected Arm')
        ax.plot(exp_offset_x, exp_offset_z, 'mo', markersize=6)

        # Calculate actual arm positions
        act_t1 = np.radians(self.theta1_actual)
        act_t2 = np.radians(self.theta2_actual)
        act_x1 = self.L1 * np.cos(act_t1)
        act_z1 = self.L1 * np.sin(act_t1)
        act_x2 = act_x1 + self.L2 * np.cos(act_t1 + act_t2)
        act_z2 = act_z1 + self.L2 * np.sin(act_t1 + act_t2)

        # Actual offset line (horizontal X)
        act_offset_x = act_x2 + self.tool_offset
        act_offset_z = act_z2
        ax.plot([act_x2, act_offset_x], [act_z2, act_offset_z], 'c-', linewidth=2, label='Actual Offset')

        # Plot actual arm
        ax.plot([0, act_x1, act_x2], [0, act_z1, act_z2], 'b-', linewidth=3, label='Actual Arm')
        ax.plot(act_offset_x, act_offset_z, 'co', markersize=8)

        ax.set_xlim(-450, 450)
        ax.set_ylim(-450, 450)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend()
        ax.set_title(f"Robot Arm with {self.tool_offset}mm X Tool Offset (J1 offset: {90 - self.j1_home_offset}°)")
        self.canvas.draw()

    def update_kinematics(self):
        # Calculate expected positions
        exp_t1 = np.radians(self.theta1_expected + 90)
        exp_t2 = np.radians(self.theta2_expected - 90)
        exp_x2 = self.L1 * np.cos(exp_t1) + self.L2 * np.cos(exp_t1 + exp_t2)
        exp_z2 = self.L1 * np.sin(exp_t1) + self.L2 * np.sin(exp_t1 + exp_t2)
        self.expected_offset_x = exp_x2 + self.tool_offset
        self.expected_offset_z = exp_z2

        # Calculate actual positions
        act_t1 = np.radians(self.theta1_actual)
        act_t2 = np.radians(self.theta2_actual)
        act_x2 = self.L1 * np.cos(act_t1) + self.L2 * np.cos(act_t1 + act_t2)
        act_z2 = self.L1 * np.sin(act_t1) + self.L2 * np.sin(act_t1 + act_t2)
        self.actual_offset_x = act_x2 + self.tool_offset
        self.actual_offset_z = act_z2

        # Add current offset position to trace
        self.trace_points.append((self.actual_offset_x, self.actual_offset_z))
        if len(self.trace_points) > self.max_trace_points:
            self.trace_points.pop(0)

        self.update_labels()
        self.plot_arm()

    def update_inverse_kinematics(self, dx=0, dz=0):
        self.expected_offset_x += dx
        self.expected_offset_z += dz

        target_x = self.expected_offset_x - self.tool_offset
        target_z = self.expected_offset_z

        r2 = target_x ** 2 + target_z ** 2
        r = np.sqrt(r2)

        if r > (self.L1 + self.L2) or r < abs(self.L1 - self.L2):
            print("Target out of reach")
            return

        cos_theta2 = (r2 - self.L1 ** 2 - self.L2 ** 2) / (2 * self.L1 * self.L2)
        cos_theta2 = np.clip(cos_theta2, -1, 1)
        theta2 = np.arccos(cos_theta2)

        if not self.elbow_up:
            theta2 = -theta2

        k1 = self.L1 + self.L2 * np.cos(theta2)
        k2 = self.L2 * np.sin(theta2)
        theta1 = np.arctan2(target_z, target_x) - np.arctan2(k2, k1)

        self.theta1_expected = np.degrees(theta1) - 90
        self.theta2_expected = np.degrees(theta2) + 90

        self.theta1_actual = self.theta1_expected + self.j1_home_offset
        self.theta2_actual = self.theta2_expected + self.j2_home_offset

        self.update_kinematics()

    # --- Button handlers ---
    def increase_j1(self):
        self.theta1_expected += 1
        self.theta1_actual = self.theta1_expected + self.j1_home_offset
        self.update_kinematics()

    def decrease_j1(self):
        self.theta1_expected -= 1
        self.theta1_actual = self.theta1_expected + self.j1_home_offset
        self.update_kinematics()

    def increase_j2(self):
        self.theta2_expected += 1
        self.theta2_actual = self.theta2_expected + self.j2_home_offset
        self.update_kinematics()

    def decrease_j2(self):
        self.theta2_expected -= 1
        self.theta2_actual = self.theta2_expected + self.j2_home_offset
        self.update_kinematics()

    def increase_x(self):
        self.update_inverse_kinematics(dx=5, dz=0)

    def decrease_x(self):
        self.update_inverse_kinematics(dx=-5, dz=0)

    def increase_z(self):
        self.update_inverse_kinematics(dx=0, dz=5)

    def decrease_z(self):
        self.update_inverse_kinematics(dx=0, dz=-5)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ArmControlPanel()
    window.resize(1000, 600)
    window.show()
    sys.exit(app.exec_())