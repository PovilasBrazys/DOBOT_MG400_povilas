import sys
import os
# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

from ui import RobotUI

robot_ui = RobotUI()

robot_ui.pack()
robot_ui.mainloop()