from __future__ import print_function
import sys
import numpy as np
import cv2 as cv
import argparse

PY3 = sys.version_info[0] == 3
if PY3:
    xrange = range


class ColorDetectionApp(object):
    def __init__(self, video_src=0):
        # Initialize camera
        self.cam = cv.VideoCapture(video_src)
        if not self.cam.isOpened():
            print("Error: Could not open camera")
            sys.exit(1)

        # Set camera properties for better quality
        self.cam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        self.cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        self.cam.set(cv.CAP_PROP_FPS, 30)

        # Read first frame
        ret, self.frame = self.cam.read()
        if not ret:
            print("Error: Could not read from camera")
            sys.exit(1)

        # Create windows
        cv.namedWindow('Color Detection', cv.WINDOW_NORMAL)
        cv.namedWindow('Controls', cv.WINDOW_NORMAL)
        cv.namedWindow('Histogram', cv.WINDOW_NORMAL)
        cv.namedWindow('Mask', cv.WINDOW_NORMAL)

        # Set mouse callback
        cv.setMouseCallback('Color Detection', self.onmouse)

        # Initialize variables
        self.selection = None
        self.drag_start = None
        self.track_window = None
        self.hist = None

        # Color detection parameters
        self.contrast = 50  # 0-100
        self.brightness = 50  # 0-100
        self.hue_range = 15  # How much hue variation to allow
        self.sat_min = 50  # Minimum saturation
        self.val_min = 50  # Minimum value
        self.blur_kernel = 5  # Blur kernel size
        self.morph_kernel = 5  # Morphological operations kernel size

        # Color groups for robust detection
        self.color_ranges = {
            'red': [(0, 10), (170, 180)],  # Red wraps around in HSV
            'orange': [(10, 25)],
            'yellow': [(25, 35)],
            'green': [(35, 85)],
            'cyan': [(85, 95)],
            'blue': [(95, 125)],
            'purple': [(125, 145)],
            'pink': [(145, 170)]
        }

        self.selected_color = None
        self.current_color_range = None

        # Create trackbars
        self.create_trackbars()

    def create_trackbars(self):
        """Create control trackbars"""
        cv.createTrackbar('Contrast', 'Controls', self.contrast, 100, self.on_contrast_change)
        cv.createTrackbar('Brightness', 'Controls', self.brightness, 100, self.on_brightness_change)
        cv.createTrackbar('Hue Range', 'Controls', self.hue_range, 50, self.on_hue_range_change)
        cv.createTrackbar('Min Saturation', 'Controls', self.sat_min, 255, self.on_sat_min_change)
        cv.createTrackbar('Min Value', 'Controls', self.val_min, 255, self.on_val_min_change)
        cv.createTrackbar('Blur Kernel', 'Controls', self.blur_kernel, 15, self.on_blur_change)
        cv.createTrackbar('Morph Kernel', 'Controls', self.morph_kernel, 15, self.on_morph_change)

    def on_contrast_change(self, val):
        self.contrast = val

    def on_brightness_change(self, val):
        self.brightness = val

    def on_hue_range_change(self, val):
        self.hue_range = val

    def on_sat_min_change(self, val):
        self.sat_min = val

    def on_val_min_change(self, val):
        self.val_min = val

    def on_blur_change(self, val):
        self.blur_kernel = max(1, val)
        if self.blur_kernel % 2 == 0:
            self.blur_kernel += 1

    def on_morph_change(self, val):
        self.morph_kernel = max(1, val)

    def onmouse(self, event, x, y, flags, param):
        """Handle mouse events for region selection"""
        if event == cv.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.track_window = None
            self.selection = None

        elif event == cv.EVENT_MOUSEMOVE and self.drag_start:
            xmin = min(x, self.drag_start[0])
            ymin = min(y, self.drag_start[1])
            xmax = max(x, self.drag_start[0])
            ymax = max(y, self.drag_start[1])
            self.selection = (xmin, ymin, xmax, ymax)

        elif event == cv.EVENT_LBUTTONUP and self.drag_start:
            xmin = min(x, self.drag_start[0])
            ymin = min(y, self.drag_start[1])
            xmax = max(x, self.drag_start[0])
            ymax = max(y, self.drag_start[1])
            self.track_window = (xmin, ymin, xmax - xmin, ymax - ymin)
            self.drag_start = None

            # Analyze selected region to determine color
            if self.track_window[2] > 10 and self.track_window[3] > 10:
                self.analyze_selected_region()

    def analyze_selected_region(self):
        """Analyze the selected region to determine dominant color"""
        if not self.track_window:
            return

        x, y, w, h = self.track_window
        hsv = cv.cvtColor(self.frame, cv.COLOR_BGR2HSV)
        roi = hsv[y:y + h, x:x + w]

        # Calculate histogram of hue channel
        hist = cv.calcHist([roi], [0], None, [180], [0, 180])

        # Find dominant hue
        dominant_hue = np.argmax(hist)

        # Determine color group
        self.selected_color = self.get_color_name(dominant_hue)
        self.current_color_range = self.get_color_range(dominant_hue)

        print(f"Detected color: {self.selected_color} (Hue: {dominant_hue})")

    def get_color_name(self, hue):
        """Get color name from hue value"""
        for color, ranges in self.color_ranges.items():
            for hue_range in ranges:
                if hue_range[0] <= hue <= hue_range[1]:
                    return color
        return "unknown"

    def get_color_range(self, hue):
        """Get the appropriate hue range for detection"""
        # Create range around the dominant hue
        lower_hue = max(0, hue - self.hue_range)
        upper_hue = min(179, hue + self.hue_range)

        # Handle red color wrap-around
        if hue < self.hue_range:
            return [(0, upper_hue), (180 - (self.hue_range - hue), 179)]
        elif hue > 179 - self.hue_range:
            return [(lower_hue, 179), (0, self.hue_range - (179 - hue))]
        else:
            return [(lower_hue, upper_hue)]

    def adjust_contrast_brightness(self, img):
        """Apply contrast and brightness adjustments"""
        alpha = self.contrast / 50.0  # Contrast control (1.0-3.0)
        beta = (self.brightness - 50) * 2  # Brightness control (-100 to 100)

        adjusted = cv.convertScaleAbs(img, alpha=alpha, beta=beta)
        return adjusted

    def create_mask(self, hsv):
        """Create mask for color detection with improved robustness"""
        if not self.current_color_range:
            return np.zeros(hsv.shape[:2], dtype=np.uint8)

        # Create mask for each hue range
        masks = []
        for hue_range in self.current_color_range:
            lower = np.array([hue_range[0], self.sat_min, self.val_min])
            upper = np.array([hue_range[1], 255, 255])
            mask = cv.inRange(hsv, lower, upper)
            masks.append(mask)

        # Combine masks
        final_mask = masks[0]
        for mask in masks[1:]:
            final_mask = cv.bitwise_or(final_mask, mask)

        # Apply morphological operations to clean up the mask
        if self.blur_kernel > 1:
            final_mask = cv.medianBlur(final_mask, self.blur_kernel)

        if self.morph_kernel > 1:
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,
                                              (self.morph_kernel, self.morph_kernel))
            final_mask = cv.morphologyEx(final_mask, cv.MORPH_OPEN, kernel)
            final_mask = cv.morphologyEx(final_mask, cv.MORPH_CLOSE, kernel)

        return final_mask

    def show_histogram(self, hsv, mask):
        """Display 1D histogram of hue values"""
        if mask is None:
            return

        # Calculate histogram
        hist = cv.calcHist([hsv], [0], mask, [180], [0, 180])

        # Create histogram image
        hist_img = np.zeros((300, 720, 3), dtype=np.uint8)

        # Normalize histogram
        cv.normalize(hist, hist, 0, 250, cv.NORM_MINMAX)

        # Draw histogram bars
        for i in range(180):
            val = int(hist[i])
            color = cv.cvtColor(np.uint8([[[i, 255, 255]]]), cv.COLOR_HSV2BGR)[0][0]
            cv.rectangle(hist_img,
                         (i * 4, 300),
                         (i * 4 + 4, 300 - val),
                         (int(color[0]), int(color[1]), int(color[2])),
                         -1)

        # Add labels
        cv.putText(hist_img, 'Hue Histogram', (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if self.selected_color:
            cv.putText(hist_img, f'Tracking: {self.selected_color}', (10, 70),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv.imshow('Histogram', hist_img)

    def detect_objects(self, frame, mask):
        """Detect and draw bounding boxes around objects"""
        # Find contours
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Filter contours by area
        min_area = 500
        valid_contours = [cnt for cnt in contours if cv.contourArea(cnt) > min_area]

        # Draw bounding boxes and labels
        for i, contour in enumerate(valid_contours):
            # Get bounding rectangle
            x, y, w, h = cv.boundingRect(contour)

            # Draw bounding box
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Calculate area and add label
            area = cv.contourArea(contour)
            label = f'{self.selected_color or "Object"} #{i + 1} ({int(area)}px)'
            cv.putText(frame, label, (x, y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return len(valid_contours)

    def run(self):
        """Main loop"""
        print("Color Detection App")
        print("Instructions:")
        print("- Click and drag to select a region with the color you want to track")
        print("- Use trackbars to adjust detection parameters")
        print("- Press 'r' to reset selection")
        print("- Press 'q' or ESC to quit")

        while True:
            ret, self.frame = self.cam.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Apply contrast and brightness adjustments
            adjusted_frame = self.adjust_contrast_brightness(self.frame)

            # Convert to HSV
            hsv = cv.cvtColor(adjusted_frame, cv.COLOR_BGR2HSV)

            # Create visualization frame
            vis = adjusted_frame.copy()

            # Create mask if we have a color selected
            mask = None
            if self.current_color_range:
                mask = self.create_mask(hsv)

                # Detect and draw objects
                num_objects = self.detect_objects(vis, mask)

                # Show mask
                cv.imshow('Mask', mask)

                # Show histogram
                self.show_histogram(hsv, mask)

                # Add info text
                info_text = f'Objects detected: {num_objects}'
                cv.putText(vis, info_text, (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Draw selection rectangle
            if self.selection:
                x0, y0, x1, y1 = self.selection
                cv.rectangle(vis, (x0, y0), (x1, y1), (255, 0, 0), 2)

            # Show main frame
            cv.imshow('Color Detection', vis)

            # Handle key presses
            key = cv.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('r'):  # Reset
                self.selection = None
                self.track_window = None
                self.selected_color = None
                self.current_color_range = None
                print("Selection reset")

        # Cleanup
        self.cam.release()
        cv.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Color Detection App')
    parser.add_argument('--source', type=str, default='0',
                        help='Video source (0 for webcam, or path to video file)')
    args = parser.parse_args()

    # Convert source to int if it's a digit
    video_src = int(args.source) if args.source.isdigit() else args.source

    app = ColorDetectionApp(video_src)
    app.run()


if __name__ == '__main__':
    main()