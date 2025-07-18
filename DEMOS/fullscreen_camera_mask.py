import cv2
import numpy as np

# Callback funkcija trackbar'ams
def nothing(x):
    pass

# Sukuriamas langas
cv2.namedWindow("Camera Feed", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Sukuriami trackbar'ai HSV spalvos nustatymams
cv2.namedWindow("Settings")
cv2.createTrackbar("Lower Hue", "Settings", 0, 179, nothing)
cv2.createTrackbar("Upper Hue", "Settings", 179, 179, nothing)
cv2.createTrackbar("Lower Saturation", "Settings", 104, 255, nothing)
cv2.createTrackbar("Upper Saturation", "Settings", 213, 255, nothing)
cv2.createTrackbar("Lower Value", "Settings", 177, 255, nothing)
cv2.createTrackbar("Upper Value", "Settings", 255, 255, nothing)


# Atidaroma kamera (0 - numatytoji kamera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Klaida: Nepavyko atidaryti kameros.")
    exit()

while True:
    # Nuskaitomas kadras iš kameros
    ret, frame = cap.read()
    if not ret:
        print("Klaida: Nepavyko nuskaityti kadro.")
        break

    # Konvertuojamas kadras į HSV spalvų erdvę
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Gaunami HSV reikšmių nustatymai iš trackbar'ų
    lower_hue = cv2.getTrackbarPos("Lower Hue", "Settings")
    upper_hue = cv2.getTrackbarPos("Upper Hue", "Settings")
    lower_saturation = cv2.getTrackbarPos("Lower Saturation", "Settings")
    upper_saturation = cv2.getTrackbarPos("Upper Saturation", "Settings")
    lower_value = cv2.getTrackbarPos("Lower Value", "Settings")
    upper_value = cv2.getTrackbarPos("Upper Value", "Settings")

    # Nustatomi mėlynos spalvos rėžiai HSV formatu
    lower_blue = np.array([lower_hue, lower_saturation, lower_value])
    upper_blue = np.array([upper_hue, upper_saturation, upper_value])

    # Sukuriamas maskas, kuris išskiria tik mėlyną spalvą
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Primenamas maskas ant originalaus kadro
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # --- Mask refinement using morphological operations ---
    kernel = np.ones((5, 5), np.uint8)
    mask_erosion = cv2.erode(mask, kernel, iterations=1)
    mask_dilation = cv2.dilate(mask_erosion, kernel, iterations=1)


    # Parodomas originalus kadras ir rezultatas
    cv2.imshow("Camera Feed", frame)
    cv2.imshow("Blue Detection", result)
    cv2.imshow("Mask", mask_dilation)

    # Laukiama klavišo paspaudimo (1 ms)
    key = cv2.waitKey(1)

    # Jei paspaudžiamas 'q' klavišas, išeinama iš ciklo
    if key == ord('q'):
        break

# Atlaisvinami resursai
cap.release()
cv2.destroyAllWindows()
# Modified to include blue color detection with HSV filter
