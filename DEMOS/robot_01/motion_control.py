import threading
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from time import sleep
from robot_control import ConnectRobot, RunPoint, WaitArrive, GetFeed, ClearRobotError

def GrabPart(dashboard):
    print("Aktyvuojamas DO 1 (griebtuvas užsidaro)...")
    dashboard.DO(1, 1)  # Įjungiam DO1
    sleep(1)
    dashboard.DO(1, 0)  # Išjungiam DO1
    print("DO 1 išjungtas.")

def ReleasePart(dashboard):
    print("Aktyvuojamas DO 2 (griebtuvas atsidaro)...")
    dashboard.DO(2, 1)  # Įjungiam DO2 (kompresorius)
    sleep(0.5)
    dashboard.DO(2, 0)  # Išjungiam DO2
    print("DO 2 išjungtas.")


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
    
    point_a = [300, 100, -150, 200]
    point_b = [300, -100, -150, 170]
    dashboard.SpeedFactor(10)
    while True:
        move.MovL(point_a[0], point_a[1], point_a[2], point_a[3])
        WaitArrive(point_a)
        GrabPart(dashboard)
        
        print("GetAngle")
        joint_angles = dashboard.GetAngle()
        print(joint_angles)
        

        move.MovL(point_b[0], point_b[1], point_b[2], point_b[3])
        WaitArrive(point_b)
        ReleasePart(dashboard)

        print("GetPose")
        angle_data = dashboard.GetPose()
        print(angle_data)
  