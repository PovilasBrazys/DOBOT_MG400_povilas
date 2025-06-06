import threading
from dobot_api import DobotApiDashboard, DobotApi, DobotApiMove, MyType,alarmAlarmJsonFile
from time import sleep
import numpy as np
import re

# 全局变量(当前坐标)
current_actual = None
algorithm_queue = None
enableStatus_robot = None
robotErrorState = False
globalLockValue = threading.Lock()

def ConnectRobot():
    try:
        ip = "192.168.1.6"
        dashboardPort = 29999
        movePort = 30003
        feedPort = 30004
        print("正在建立连接...")
        dashboard = DobotApiDashboard(ip, dashboardPort)
        move = DobotApiMove(ip, movePort)
        feed = DobotApi(ip, feedPort)
        print(">.<连接成功>!<")
        return dashboard, move, feed
    except Exception as e:
        print(":(连接失败:(")
        raise e

def RunPoint(move: DobotApiMove, point_list: list):
    move.MovL(point_list[0], point_list[1], point_list[2], point_list[3])

def GetFeed(feed: DobotApi):
    global current_actual
    global algorithm_queue
    global enableStatus_robot
    global robotErrorState
    hasRead = 0
    while True:
        data = bytes()
        while hasRead < 1440:
            temp = feed.socket_dobot.recv(1440 - hasRead)
            if len(temp) > 0:
                hasRead += len(temp)
                data += temp
        hasRead = 0
        feedInfo = np.frombuffer(data, dtype=MyType)
        if hex((feedInfo['test_value'][0])) == '0x123456789abcdef':
            globalLockValue.acquire()
            # Refresh Properties
            current_actual = feedInfo["tool_vector_actual"][0]
            algorithm_queue = feedInfo['isRunQueuedCmd'][0]
            enableStatus_robot=feedInfo['EnableStatus'][0]
            robotErrorState= feedInfo['ErrorStatus'][0]
            globalLockValue.release()
        sleep(0.001)

def WaitArrive(point_list):
    while True:
        is_arrive = True
        globalLockValue.acquire()
        if current_actual is not None:
            for index in range(4):
                if (abs(current_actual[index] - point_list[index]) > 1):
                    is_arrive = False
            if is_arrive :
                globalLockValue.release()
                return
        globalLockValue.release()  
        sleep(0.001)

def ClearRobotError(dashboard: DobotApiDashboard):
    global robotErrorState
    dataController,dataServo =alarmAlarmJsonFile()    # 读取控制器和伺服告警码
    while True:
      globalLockValue.acquire()
      if robotErrorState:
                numbers = re.findall(r'-?\d+', dashboard.GetErrorID())
                numbers= [int(num) for num in numbers]
                if (numbers[0] == 0):
                  if (len(numbers)>1):
                    for i in numbers[1:]:
                      alarmState=False
                      if i==-2:
                          print("机器告警 机器碰撞 ",i)
                          alarmState=True
                      if alarmState:
                          continue                
                      for item in dataController:
                        if  i==item["id"]:
                            print("机器告警 Controller errorid",i,item["zh_CN"]["description"])
                            alarmState=True
                            break 
                      if alarmState:
                          continue
                      for item in dataServo:
                        if  i==item["id"]:
                            print("机器告警 Servo errorid",i,item["zh_CN"]["description"])
                            break  
                       
                    print("尝试清除错误...")
                    dashboard.ClearError()
                    sleep(0.01)
                    dashboard.Continue()

      else:  
         if int(enableStatus_robot[0])==1 and int(algorithm_queue[0])==0:
            dashboard.Continue()
      globalLockValue.release()
      sleep(5)

def move_parts(matrix1_start, matrix2_start, move, dashboard):
    # 定义矩阵间距
    matrix_spacing = -32

    parts_moved = 0
    for row in range(3):
        for col in range(3):
            if parts_moved >= 9:
                break

            # 计算第一个矩阵的抓取位置
            pick_location = [
                matrix1_start[0] + col * matrix_spacing,
                matrix1_start[1] + row * matrix_spacing,
                matrix1_start[2],
                matrix1_start[3]
            ]

            # 计算第二个矩阵的放置位置
            place_location = [
                matrix2_start[0] + col * matrix_spacing,
                matrix2_start[1] + row * matrix_spacing,
                matrix2_start[2],
                matrix2_start[3]
            ]

            # 抬高 jump_height
            jump_height = 50
            jump_location = [pick_location[0], pick_location[1], pick_location[2] + jump_height, pick_location[3]]
            RunPoint(move, jump_location)
            WaitArrive(jump_location)

            print(f"Picking from {pick_location}")
            RunPoint(move, pick_location)
            WaitArrive(pick_location)
            
            # 开启吸盘 (DO1 ON)
            print("DO1 ON (开启吸盘)")
            dashboard.DO(index=1, status=1)
            sleep(1)  # 吸附 1 秒
            print("DO1 OFF (关闭吸盘)")
            dashboard.DO(index=1, status=0)

            # 抬高 jump_height
            RunPoint(move, jump_location)
            WaitArrive(jump_location)

            # 移动到放置位置上方
            jump_location = [place_location[0], place_location[1], place_location[2] + jump_height, place_location[3]]
            RunPoint(move, jump_location)
            WaitArrive(jump_location)

            print(f"Placing at {place_location}")
            RunPoint(move, place_location)
            WaitArrive(place_location)

            # 释放气压 (DO2 ON)
            print("DO2 ON (释放气压)")
            dashboard.DO(index=2, status=1)
            sleep(0.2)  # 释放 0.1 秒
            print("DO2 OFF (关闭气压)")
            dashboard.DO(index=2, status=0)

            # 抬高 jump_height
            RunPoint(move, jump_location)
            WaitArrive(jump_location)

            parts_moved += 1

    print("移动完成！")
       
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

    print("开始执行 Pick & Place...")

    # 定义第一个矩阵的起始位置
    matrix1_start = [314, 41, -150, 0]

    # 定义第二个矩阵的起始位置
    matrix2_start = [314, -100, -150, 0]

    while True:
        print("Moving parts from Matrix 1 to Matrix 2...")
        move_parts(matrix1_start, matrix2_start, move, dashboard)
        print("Moving parts from Matrix 2 to Matrix 1...")
        move_parts(matrix2_start, matrix1_start, move, dashboard)
