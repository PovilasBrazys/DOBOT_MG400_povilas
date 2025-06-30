import threading
import sys
import os
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from time import sleep
from robot_control import ConnectRobot, RunPoint, WaitArrive, GetFeed, ClearRobotError, true_camPointXY

def camera_thread(dashboard, move, camera_ready_event, stop_event):
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return

    window_name = "Camera Feed"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Signal that camera is ready
    camera_ready_event.set()

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        height, width, _ = frame.shape
        center_x = width // 2
        center_y = height // 2

        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Red dot

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            stop_event.set()  # Signal all threads to stop
            break

    cap.release()
    cv2.destroyAllWindows()

def robot_movement_thread(dashboard, move, camera_ready_event, stop_event):
    # Wait for camera to be ready
    camera_ready_event.wait()

    setZ = -190
    cameraX_offset = 53
    point_a = [347, 0, setZ, 0]
    dashboard.SpeedFactor(10)

    point_b = true_camPointXY([point_a[0], 50, setZ, 0], cameraX_offset)
    point_c = true_camPointXY([point_a[0], -50, setZ, 0], cameraX_offset)

    while not stop_event.is_set():
        move.MovL(point_a[0], point_a[1], point_a[2], point_a[3])
        WaitArrive(point_a)
        print("GetPoseA", dashboard.GetPose())
        sleep(3)

        if stop_event.is_set():
            break

        move.MovL(point_b[0], point_b[1], point_b[2], point_b[3])
        WaitArrive(point_b)
        print("GetPoseB", dashboard.GetPose())
        sleep(3)

        if stop_event.is_set():
            break

        move.MovL(point_a[0], point_a[1], point_a[2], point_a[3])
        WaitArrive(point_a)
        print("GetPoseA", dashboard.GetPose())
        sleep(3)

        if stop_event.is_set():
            break

        move.MovL(point_c[0], point_c[1], point_c[2], point_c[3])
        WaitArrive(point_c)
        print("GetPoseC", dashboard.GetPose())
        sleep(3)

if __name__ == '__main__':
    dashboard, move, feed = ConnectRobot()
    dashboard.EnableRobot()

    # Create events for camera readiness and program termination
    camera_ready_event = threading.Event()
    stop_event = threading.Event()

    # Start feed and error clearing threads
    feed_thread = threading.Thread(target=GetFeed, args=(feed,))
    feed_thread.daemon = True
    feed_thread.start()

    feed_thread1 = threading.Thread(target=ClearRobotError, args=(dashboard,))
    feed_thread1.daemon = True
    feed_thread1.start()

    # Start camera thread
    camera_thread1 = threading.Thread(target=camera_thread, args=(dashboard, move, camera_ready_event, stop_event))
    camera_thread1.daemon = True
    camera_thread1.start()

    # Start robot movement thread
    robot_thread = threading.Thread(target=robot_movement_thread, args=(dashboard, move, camera_ready_event, stop_event))
    robot_thread.daemon = True
    robot_thread.start()

    # Keep main thread running
    try:
        while not stop_event.is_set():
            sleep(1)
    except KeyboardInterrupt:
        print("Program terminated by user")
        stop_event.set()
    
    # Ensure clean shutdown
    dashboard.DisableRobot()
    cv2.destroyAllWindows()
    sys.exit(0)