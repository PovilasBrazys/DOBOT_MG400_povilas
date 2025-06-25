import threading
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from time import sleep
from robot_control_module import ConnectRobot, RunPoint, WaitArrive, GetFeed, ClearRobotError



if __name__ == '__main__':
    dashboard, move, feed = ConnectRobot()
    print("开始使能...")
    dashboard.EnableRobot()
    print("完成使能:)")
    feed_thread = threading.Thread(target=GetFeed, args=(feed,))
    feed_thread.setDaemon(True)
    feed_thread.start()
    feed_thread1 = threading.Thread(target=ClearRobotError, args=(dashboard,))
    feed_thread1.setDaemon(True)
    feed_thread1.start()
    dashboard.Tool(8)
    dashboard.SetTool(8, 53, 0, 0, 0)
    print("循环执行...")
    point_a = [300, 0, 0, 0]
    dashboard.SpeedFactor(10)
    while True:
        move.MovL(point_a[0], point_a[1], point_a[2], point_a[3])
        WaitArrive(point_a)
        
        print("GetPose")
        angle_data = dashboard.GetPose()
        print(angle_data)
        
 