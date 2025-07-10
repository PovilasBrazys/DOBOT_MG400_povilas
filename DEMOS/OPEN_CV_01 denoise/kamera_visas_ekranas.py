import cv2
import numpy as np

def nothing(x):
    pass

def create_trackbars():
    cv2.namedWindow('Nustatymai')
    # Raudona
    cv2.createTrackbar('Red H min', 'Nustatymai', 0, 179, nothing)
    cv2.createTrackbar('Red H max', 'Nustatymai', 10, 179, nothing)
    cv2.createTrackbar('Red S min', 'Nustatymai', 100, 255, nothing)
    cv2.createTrackbar('Red S max', 'Nustatymai', 255, 255, nothing)
    cv2.createTrackbar('Red V min', 'Nustatymai', 100, 255, nothing)
    cv2.createTrackbar('Red V max', 'Nustatymai', 255, 255, nothing)
    # Žalia
    cv2.createTrackbar('Green H min', 'Nustatymai', 40, 179, nothing)
    cv2.createTrackbar('Green H max', 'Nustatymai', 80, 179, nothing)
    cv2.createTrackbar('Green S min', 'Nustatymai', 70, 255, nothing)
    cv2.createTrackbar('Green S max', 'Nustatymai', 255, 255, nothing)
    cv2.createTrackbar('Green V min', 'Nustatymai', 70, 255, nothing)
    cv2.createTrackbar('Green V max', 'Nustatymai', 255, 255, nothing)
    # Mėlyna
    cv2.createTrackbar('Blue H min', 'Nustatymai', 100, 179, nothing)
    cv2.createTrackbar('Blue H max', 'Nustatymai', 140, 179, nothing)
    cv2.createTrackbar('Blue S min', 'Nustatymai', 150, 255, nothing)
    cv2.createTrackbar('Blue S max', 'Nustatymai', 255, 255, nothing)
    cv2.createTrackbar('Blue V min', 'Nustatymai', 0, 255, nothing)
    cv2.createTrackbar('Blue V max', 'Nustatymai', 255, 255, nothing)
    # Kontrastas
    cv2.createTrackbar('Kontrastas', 'Nustatymai', 10, 30, nothing) # 1.0 - 3.0

def get_trackbar_values():
    # Raudona
    r_h_min = cv2.getTrackbarPos('Red H min', 'Nustatymai')
    r_h_max = cv2.getTrackbarPos('Red H max', 'Nustatymai')
    r_s_min = cv2.getTrackbarPos('Red S min', 'Nustatymai')
    r_s_max = cv2.getTrackbarPos('Red S max', 'Nustatymai')
    r_v_min = cv2.getTrackbarPos('Red V min', 'Nustatymai')
    r_v_max = cv2.getTrackbarPos('Red V max', 'Nustatymai')
    # Žalia
    g_h_min = cv2.getTrackbarPos('Green H min', 'Nustatymai')
    g_h_max = cv2.getTrackbarPos('Green H max', 'Nustatymai')
    g_s_min = cv2.getTrackbarPos('Green S min', 'Nustatymai')
    g_s_max = cv2.getTrackbarPos('Green S max', 'Nustatymai')
    g_v_min = cv2.getTrackbarPos('Green V min', 'Nustatymai')
    g_v_max = cv2.getTrackbarPos('Green V max', 'Nustatymai')
    # Mėlyna
    b_h_min = cv2.getTrackbarPos('Blue H min', 'Nustatymai')
    b_h_max = cv2.getTrackbarPos('Blue H max', 'Nustatymai')
    b_s_min = cv2.getTrackbarPos('Blue S min', 'Nustatymai')
    b_s_max = cv2.getTrackbarPos('Blue S max', 'Nustatymai')
    b_v_min = cv2.getTrackbarPos('Blue V min', 'Nustatymai')
    b_v_max = cv2.getTrackbarPos('Blue V max', 'Nustatymai')
    # Kontrastas
    kontrastas = cv2.getTrackbarPos('Kontrastas', 'Nustatymai') / 10.0
    return {
        'red':   (np.array([r_h_min, r_s_min, r_v_min]), np.array([r_h_max, r_s_max, r_v_max])),
        'red2':  (np.array([160, r_s_min, r_v_min]), np.array([179, r_s_max, r_v_max])), # antra raudonos sritis
        'green': (np.array([g_h_min, g_s_min, g_v_min]), np.array([g_h_max, g_s_max, g_v_max])),
        'blue':  (np.array([b_h_min, b_s_min, b_v_min]), np.array([b_h_max, b_s_max, b_v_max])),
        'kontrastas': kontrastas
    }

def adjust_contrast(frame, alpha):
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=0)

def detect_and_draw_circles(frame, hsv_ranges):
    # Kontrastas
    frame = adjust_contrast(frame, hsv_ranges['kontrastas'])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Kaukės
    mask_red1 = cv2.inRange(hsv, hsv_ranges['red'][0], hsv_ranges['red'][1])
    mask_red2 = cv2.inRange(hsv, hsv_ranges['red2'][0], hsv_ranges['red2'][1])
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_green = cv2.inRange(hsv, hsv_ranges['green'][0], hsv_ranges['green'][1])
    mask_blue = cv2.inRange(hsv, hsv_ranges['blue'][0], hsv_ranges['blue'][1])

    # Aptikti apskritimus kiekvienoje kaukėje
    for mask, color in [(mask_red, (0,0,255)), (mask_green, (0,255,0)), (mask_blue, (255,0,0))]:
        blurred = cv2.GaussianBlur(mask, (9,9), 2)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                   param1=50, param2=30, minRadius=10, maxRadius=200)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(frame, (i[0], i[1]), i[2], color, 4)
                cv2.circle(frame, (i[0], i[1]), 2, color, 3)
    return frame

def show_fullscreen_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera nerasta.")
        return

    create_trackbars()
    window_name = "Kameros vaizdas"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Nepavyko nuskaityti vaizdo.")
            break

        hsv_ranges = get_trackbar_values()
        frame = detect_and_draw_circles(frame, hsv_ranges)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC mygtukas uždaro langą
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_fullscreen_camera()