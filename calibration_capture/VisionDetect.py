import cv2
import numpy as np


def main():
    # Load calibration data
    try:
        data = np.load('calibration_data.npz')
        mtx, dist = data['mtx'], data['dist']
        print("Calibration data loaded successfully")
    except Exception as e:
        print(f"Error loading calibration data: {e}")
        return

    # Initialize camera
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cam.set(cv2.CAP_PROP_FPS, 5)
    cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    if not cam.isOpened():
        print("Error: Could not open camera")
        return

    # Get actual frame size
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    center_x = width // 2
    center_y = height // 2
    print(f"Camera resolution: {width}x{height}")

    # Undistortion mapping
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, (width, height), cv2.CV_16SC2)

    # Create control window with trackbars
    cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Controls', 600, 200)
    cv2.createTrackbar('Threshold', 'Controls', 60, 255, lambda x: None)
    cv2.createTrackbar('Min Radius', 'Controls', 5, 500, lambda x: None)
    cv2.createTrackbar('Max Radius', 'Controls', 300, 500, lambda x: None)
    cv2.createTrackbar('Mean Threshold', 'Controls', 80, 255, lambda x: None)

    # Main window
    cv2.namedWindow('Undistorted View', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Undistorted View', 1280, 720)

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Failed to capture frame")
                continue

            # Undistort image
            undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

            # Get current trackbar positions
            thresh_val = cv2.getTrackbarPos('Threshold', 'Controls')
            min_radius = cv2.getTrackbarPos('Min Radius', 'Controls')
            max_radius = cv2.getTrackbarPos('Max Radius', 'Controls')
            mean_thresh = cv2.getTrackbarPos('Mean Threshold', 'Controls')

            # Ensure valid radius range
            current_min = min(min_radius, max_radius)
            current_max = max(min_radius, max_radius)

            # Convert to grayscale and process
            gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)

            # Adaptive thresholding for better contrast handling
            _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)

            # Morphological operations to reduce noise
            kernel = np.ones((3, 3), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

            # Find contours
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 100:
                    continue

                # Improved circularity check
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = (4 * np.pi * area) / (perimeter ** 2)

                if circularity < 0.7:
                    continue

                # Ellipse fitting for better shape verification
                if len(cnt) < 5:
                    continue
                ellipse = cv2.fitEllipse(cnt)
                (xe, ye), (ma, MA), angle = ellipse
                axis_ratio = min(ma, MA) / max(ma, MA)
                if axis_ratio < 0.7:
                    continue

                (x, y), radius = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                radius = int(radius)

                if radius < current_min or radius > current_max:
                    continue

                # Intensity check with dynamic threshold
                mask = np.zeros_like(gray)
                cv2.circle(mask, center, radius, 255, -1)
                mean_val = cv2.mean(gray, mask=mask)[0]
                if mean_val > mean_thresh:
                    continue

                # Calculate position relative to center
                dx = int(x - center_x)
                dy = int(y - center_y)

                # Draw detection markers
                cv2.circle(undistorted, center, radius, (0, 255, 0), 2)
                cv2.circle(undistorted, center, 3, (0, 0, 255), -1)
                cv2.ellipse(undistorted, ellipse, (255, 0, 0), 2)

                # Position text with background for readability
                label = f"dx: {dx}, dy: {dy}"
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(undistorted,
                              (center[0] + 10, center[1] - text_height - 10),
                              (center[0] + 10 + text_width, center[1] - 10),
                              (0, 0, 0), -1)
                cv2.putText(undistorted, label, (center[0] + 10, center[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Draw center markers
            cv2.drawMarker(undistorted, (center_x, center_y), (255, 0, 255),
                           cv2.MARKER_CROSS, 20, 2)
            cv2.line(undistorted, (center_x, 0), (center_x, height), (255, 0, 255), 1)
            cv2.line(undistorted, (0, center_y), (width, center_y), (255, 0, 255), 1)

            # Show parameter overlay
            param_text = f"Thresh: {thresh_val} | Min R: {current_min} | Max R: {current_max} | Mean: {mean_thresh}"
            cv2.putText(undistorted, param_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Undistorted View', undistorted)
            cv2.imshow('Threshold View', cleaned)

            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break

    finally:
        cam.release()
        cv2.destroyAllWindows()
        print("Resources released")


if __name__ == "__main__":
    main()