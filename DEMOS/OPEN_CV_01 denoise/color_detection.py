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

        # Create windows (Controls, Histogram, Mask unchanged)
        cv.namedWindow('Color Detection', cv.WINDOW_NORMAL)
        cv.namedWindow('Controls', cv.WINDOW_NORMAL)
        cv.namedWindow('Histogram', cv.WINDOW_NORMAL)
        cv.namedWindow('Mask', cv.WINDOW_NORMAL)

        # Set fullscreen for 'Color Detection' window (borderless fullscreen)
        cv.setWindowProperty('Color Detection', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

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
        self.hue_range = 20  # How much hue variation to allow
        self.sat_min = 40  # Minimum saturation
        self.val_min = 30  # Minimum value (lowered for shadows)
        self.blur_kernel = 5  # Blur kernel size
        self.morph_kernel = 5  # Morphological operations kernel size

        # Enhanced color groups with tighter ranges to prevent false positives
        self.color_ranges = {
            'red': [(0, 12), (168, 180)],  # Tighter red range
            'orange': [(12, 25)],  # Clear boundaries
            'yellow': [(25, 35)],
            'green': [(35, 80)],  # Tighter green range
            'cyan': [(80, 95)],  # Clear cyan boundary
            'blue': [(100, 125)],  # Tighter, more specific blue range
            'purple': [(125, 145)],  # Clear purple range
            'pink': [(145, 168)]  # Clear pink range
        }

        # Advanced detection parameters
        self.shadow_detection = True  # Detect darker tones of same color
        self.adaptive_range = True  # Adapt range based on lighting
        self.false_positive_filter = True  # Enhanced filtering for false positives
        self.shape_completion = True  # Fill gaps in 3D object detection
        self.multi_exposure_analysis = False  # Analyze multiple exposure levels

        # 3D object detection parameters
        self.shadow_tolerance = 0.3  # How much darker shadows can be (30% of original value)
        self.hue_shift_tolerance = 8  # Hue shift tolerance for shadows (degrees)
        self.gradient_threshold = 30  # Edge gradient threshold for shape completion

        self.selected_color = None
        self.current_color_range = None

        # Create trackbars
        self.create_trackbars()

    def show_fullscreen_camera_feed(self):
        """
        Show camera feed on the 'Color Detection' window in borderless fullscreen mode,
        draw a red dot cursor at the center and overlay RGB values of center pixel.
        """
        while True:
            ret, frame = self.cam.read()
            if not ret:
                print("Error: Could not read frame")
                break

            height, width, _ = frame.shape
            center_x = width // 2
            center_y = height // 2

            # Get BGR values at center pixel
            b, g, r = frame[center_y, center_x]
            rgb_text = f"RGB: ({r}, {g}, {b})"

            # Overlay RGB text
            cv.putText(
                frame,
                rgb_text,
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv.LINE_AA
            )

            # Draw red dot at center
            cv.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            # Show the frame in 'Color Detection' window (fullscreen)
            cv.imshow('Color Detection', frame)

            # Exit on ESC key press
            if cv.waitKey(1) & 0xFF == 27:
                break

        self.cam.release()
        cv.destroyAllWindows()

    def create_trackbars(self):
        """Create control trackbars"""
        cv.createTrackbar('Contrast', 'Controls', self.contrast, 100, self.on_contrast_change)
        cv.createTrackbar('Brightness', 'Controls', self.brightness, 100, self.on_brightness_change)
        cv.createTrackbar('Hue Range', 'Controls', self.hue_range, 50, self.on_hue_range_change)
        cv.createTrackbar('Min Saturation', 'Controls', self.sat_min, 255, self.on_sat_min_change)
        cv.createTrackbar('Min Value', 'Controls', self.val_min, 255, self.on_val_min_change)
        cv.createTrackbar('Shadow Tolerance', 'Controls', int(self.shadow_tolerance * 100), 100, self.on_shadow_change)
        cv.createTrackbar('Hue Shift Tolerance', 'Controls', self.hue_shift_tolerance, 30, self.on_hue_shift_change)
        cv.createTrackbar('Shape Completion', 'Controls', 1 if self.shape_completion else 0, 1, self.on_shape_change)
        cv.createTrackbar('False Positive Filter', 'Controls', 1 if self.false_positive_filter else 0, 1,
                          self.on_fp_change)

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

    def on_shadow_change(self, val):
        self.shadow_tolerance = val / 100.0

    def on_hue_shift_change(self, val):
        self.hue_shift_tolerance = val

    def on_shape_change(self, val):
        self.shape_completion = bool(val)

    def on_fp_change(self, val):
        self.false_positive_filter = bool(val)

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
        """Advanced analysis with false positive filtering and shadow detection"""
        if not self.track_window:
            return

        x, y, w, h = self.track_window
        hsv = cv.cvtColor(self.frame, cv.COLOR_BGR2HSV)
        roi = hsv[y:y + h, x:x + w]

        # Calculate detailed histogram analysis
        hist_h = cv.calcHist([roi], [0], None, [180], [0, 180])
        hist_s = cv.calcHist([roi], [1], None, [256], [0, 256])
        hist_v = cv.calcHist([roi], [2], None, [256], [0, 256])

        # Find dominant hue with confidence scoring
        dominant_hue = np.argmax(hist_h)
        hue_confidence = hist_h[dominant_hue] / np.sum(hist_h)

        # Find dominant saturation and value for better thresholding
        dominant_sat = np.argmax(hist_s)
        dominant_val = np.argmax(hist_v)

        # Analyze color distribution to avoid false positives
        significant_hues = []
        for i in range(180):
            if hist_h[i] > np.max(hist_h) * 0.1:  # At least 10% of max
                significant_hues.append(i)

        # Check for color purity (avoid mixed/ambiguous colors)
        if len(significant_hues) > 20:  # Too many significant hues = noisy/mixed color
            print("Warning: Selected region contains mixed colors. Try selecting a more uniform area.")
            return

        # Enhanced dominant hue calculation considering shadows
        hue_candidates = []
        for hue in significant_hues:
            weight = hist_h[hue][0]
            hue_candidates.append((hue, weight))

        # Sort by weight and analyze top candidates
        hue_candidates.sort(key=lambda x: x[1], reverse=True)

        if len(hue_candidates) > 1:
            # Check if secondary peaks are likely shadows of main color
            main_hue = hue_candidates[0][0]
            for candidate_hue, weight in hue_candidates[1:]:
                hue_diff = abs(candidate_hue - main_hue)
                if hue_diff > 90:  # Handle wraparound
                    hue_diff = 180 - hue_diff

                if hue_diff <= self.hue_shift_tolerance and weight > hue_candidates[0][1] * 0.3:
                    print(f"Detected shadow variant at hue {candidate_hue}")

        dominant_hue = hue_candidates[0][0]

        # Store additional color properties for better detection
        self.dominant_sat = dominant_sat
        self.dominant_val = dominant_val
        self.hue_confidence = hue_confidence[0]

        # Determine color group with stricter criteria
        self.selected_color = self.get_color_name(dominant_hue)
        self.current_color_range = self.get_color_range(dominant_hue)

        print(f"Detected color: {self.selected_color} (Hue: {dominant_hue}, Confidence: {self.hue_confidence:.2f})")
        print(f"Dominant S/V: {dominant_sat}/{dominant_val}")
        print(f"Detection range: {self.current_color_range}")

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
        """Simple contrast and brightness adjustment"""
        alpha = self.contrast / 50.0  # Contrast control (1.0-3.0)
        beta = (self.brightness - 50) * 2  # Brightness control (-100 to 100)

        adjusted = cv.convertScaleAbs(img, alpha=alpha, beta=beta)
        return adjusted

    def create_mask(self, hsv):
        """Advanced mask creation for 3D objects with shadow detection and false positive filtering"""
        if not self.current_color_range:
            return np.zeros(hsv.shape[:2], dtype=np.uint8)

        # Create base mask with primary color ranges
        base_masks = []
        for hue_range in self.current_color_range:
            lower = np.array([hue_range[0], self.sat_min, self.val_min])
            upper = np.array([hue_range[1], 255, 255])
            mask = cv.inRange(hsv, lower, upper)
            base_masks.append(mask)

        # Combine base masks
        primary_mask = base_masks[0]
        for mask in base_masks[1:]:
            primary_mask = cv.bitwise_or(primary_mask, mask)

        # Create shadow/darker tone masks for 3D object detection
        shadow_masks = []
        if self.shadow_detection and hasattr(self, 'dominant_val'):
            for hue_range in self.current_color_range:
                # Calculate shadow value threshold
                shadow_val_min = max(10, int(self.dominant_val * self.shadow_tolerance))

                # Allow slight hue shift in shadows
                shadow_hue_min = max(0, hue_range[0] - self.hue_shift_tolerance)
                shadow_hue_max = min(179, hue_range[1] + self.hue_shift_tolerance)

                # More lenient saturation for shadows
                shadow_sat_min = max(10, self.sat_min // 2)

                lower_shadow = np.array([shadow_hue_min, shadow_sat_min, shadow_val_min])
                upper_shadow = np.array([shadow_hue_max, 255, self.val_min + 20])

                shadow_mask = cv.inRange(hsv, lower_shadow, upper_shadow)
                shadow_masks.append(shadow_mask)

        # Combine shadow masks
        if shadow_masks:
            combined_shadow_mask = shadow_masks[0]
            for mask in shadow_masks[1:]:
                combined_shadow_mask = cv.bitwise_or(combined_shadow_mask, mask)

            # Only include shadow pixels that are connected to primary color regions
            shadow_mask_filtered = self.filter_shadow_mask(primary_mask, combined_shadow_mask)
            final_mask = cv.bitwise_or(primary_mask, shadow_mask_filtered)
        else:
            final_mask = primary_mask

        # False positive filtering
        if self.false_positive_filter:
            final_mask = self.apply_false_positive_filter(final_mask, hsv)

        # Shape completion for 3D objects
        if self.shape_completion:
            final_mask = self.complete_object_shape(final_mask)

        # Standard morphological cleanup
        if self.blur_kernel > 1:
            final_mask = cv.medianBlur(final_mask, self.blur_kernel)

        if self.morph_kernel > 1:
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,
                                              (self.morph_kernel, self.morph_kernel))
            final_mask = cv.morphologyEx(final_mask, cv.MORPH_CLOSE, kernel)
            final_mask = cv.morphologyEx(final_mask, cv.MORPH_OPEN, kernel)

        return final_mask

    def filter_shadow_mask(self, primary_mask, shadow_mask):
        """Filter shadow mask to only include regions connected to primary color"""
        # Dilate primary mask to create connection regions
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
        dilated_primary = cv.dilate(primary_mask, kernel, iterations=2)

        # Only keep shadow regions that overlap with dilated primary regions
        connected_shadows = cv.bitwise_and(shadow_mask, dilated_primary)

        return connected_shadows

    def apply_false_positive_filter(self, mask, hsv):
        """Apply advanced filtering to remove false positives"""
        # Remove small isolated regions (likely false positives)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Calculate size statistics
        if len(contours) == 0:
            return mask

        areas = [cv.contourArea(cnt) for cnt in contours]
        if len(areas) == 0:
            return mask

        # Remove regions significantly smaller than the largest
        max_area = max(areas)
        min_significant_area = max(100, max_area * 0.1)  # At least 10% of largest or 100 pixels

        filtered_mask = np.zeros_like(mask)
        for contour in contours:
            if cv.contourArea(contour) >= min_significant_area:
                cv.fillPoly(filtered_mask, [contour], 255)

        # Additional spatial coherence filtering
        # Remove regions that don't have similar color neighbors
        filtered_mask = self.spatial_coherence_filter(filtered_mask, hsv)

        return filtered_mask

    def spatial_coherence_filter(self, mask, hsv):
        """Remove isolated regions that don't have color-similar neighbors"""
        # Create a copy for modification
        coherent_mask = mask.copy()

        # Find contours
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv.boundingRect(contour)

            # Expand search area
            search_x = max(0, x - w // 2)
            search_y = max(0, y - h // 2)
            search_w = min(hsv.shape[1] - search_x, w * 2)
            search_h = min(hsv.shape[0] - search_y, h * 2)

            # Analyze neighborhood
            neighborhood = hsv[search_y:search_y + search_h, search_x:search_x + search_w]
            region = hsv[y:y + h, x:x + w]

            if neighborhood.size > 0 and region.size > 0:
                # Calculate color similarity in neighborhood
                region_mean_hue = np.mean(region[:, :, 0])
                neighborhood_similar = np.abs(neighborhood[:, :, 0] - region_mean_hue) < self.hue_range
                similarity_ratio = np.sum(neighborhood_similar) / neighborhood_similar.size

                # Remove region if it's too isolated (less than 20% similar colors in neighborhood)
                if similarity_ratio < 0.2:
                    cv.fillPoly(coherent_mask, [contour], 0)

        return coherent_mask

    def complete_object_shape(self, mask):
        """Fill gaps and complete object shapes for better 3D object detection"""
        # Use more aggressive closing to connect object parts
        kernel_large = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
        completed = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel_large)

        # Fill convex hull of large objects to complete shape
        contours, _ = cv.findContours(completed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        shape_completed = completed.copy()
        for contour in contours:
            area = cv.contourArea(contour)
            if area > 1000:  # Only for reasonably large objects
                # Create convex hull
                hull = cv.convexHull(contour)
                cv.fillPoly(shape_completed, [hull], 255)

        return shape_completed

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

            adjusted_frame = self.adjust_contrast_brightness(self.frame)
            hsv = cv.cvtColor(adjusted_frame, cv.COLOR_BGR2HSV)
            vis = adjusted_frame.copy()

            # Draw red dot at center of frame
            height, width = vis.shape[:2]
            center_x = width // 2
            center_y = height // 2
            cv.circle(vis, (center_x, center_y), 5, (0, 0, 255), -1)  # Red dot

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