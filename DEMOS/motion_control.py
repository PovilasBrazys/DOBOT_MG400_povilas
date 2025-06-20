import threading
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from time import sleep
from robot_control import ConnectRobot, RunPoint, WaitArrive, GetFeed, ClearRobotError

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
    print("循环执行...")
    point_a = [350, 50, 0, 200]
    point_b = [350, -50, 0, 170]
    while True:
        RunPoint(move, point_a)
        WaitArrive(point_a)
        RunPoint(move, point_b)
        WaitArrive(point_b)
