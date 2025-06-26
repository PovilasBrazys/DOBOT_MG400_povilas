import threading
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from time import sleep
from robot_control import ConnectRobot, RunPoint, WaitArrive, GetFeed, ClearRobotError
import math

def true_camPointXY(point, cameraX_offset):
    x = point[0] + cameraX_offset
    y = point[1]
    deg = math.degrees(math.atan(y / x))
    print(deg)

    x1 = cameraX_offset * math.cos(math.radians(deg))
    y1 = cameraX_offset * math.sin(math.radians(deg))
    print("x1:", x1)
    print("y1:", y1)
    x2 = x - x1
    y2 = y - y1

    print("x2:", x2)
    print("y2:", y2)
    point[0] = x2
    point[1] = y2
    return point


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

    #setZ for camera calibration
    setZ=-190
    #setZ=0
    cameraX_offset = 53

    point_a = [348.5+cameraX_offset, 0, setZ, 0]
    #point_a = [361.41+cameraX_offset, 0, setZ, 0]
    #point_a = [371.41+cameraX_offset, 0, setZ, 0]
    dashboard.SpeedFactor(10)


    x = point_a[0] + cameraX_offset
    y = 50
    deg = math.degrees(math.atan(y / x))
    print(deg)

    x1 = cameraX_offset * math.cos(math.radians(deg))
    y1 = cameraX_offset * math.sin(math.radians(deg))
    print("x1:", x1)
    print("y1:", y1)
    x2 = x - x1
    y2 = y - y1

    print("x2:", x2)
    print("y2:", y2)

    point_b = true_camPointXY( [point_a[0], 50, setZ, 0], cameraX_offset)
    point_c = true_camPointXY([point_a[0], -50, setZ, 0], cameraX_offset)

    while True:
        move.MovL(point_a[0], point_a[1], point_a[2], point_a[3])
        WaitArrive(point_a)
        print("GetPoseA", dashboard.GetPose())
        sleep(3)

        move.MovL(point_b[0], point_b[1], point_b[2], point_b[3])
        WaitArrive(point_a)
        print("GetPoseB", dashboard.GetPose())
        sleep(3)

        move.MovL(point_a[0], point_a[1], point_a[2], point_a[3])
        WaitArrive(point_a)
        print("GetPoseA", dashboard.GetPose())
        sleep(3)

        move.MovL(point_c[0], point_c[1], point_c[2], point_c[3])
        WaitArrive(point_b)
        print("GetPoseC", dashboard.GetPose())
 