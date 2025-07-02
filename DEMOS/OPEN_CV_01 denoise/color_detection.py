from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2 as cv

# local module
import video
from video import presets


class App(object):
    def __init__(self, video_src):
        self.cam = video.create_capture(video_src, presets['cube'])
        _ret, self.frame = self.cam.read()
        cv.namedWindow('camshift')
        cv.setMouseCallback('camshift', self.onmouse)

        self.selection = None
        self.drag_start = None
        self.show_backproj = False
        self.track_window = None

    def onmouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.track_window = None
        if self.drag_start:
            xmin = min(x, self.drag_start[0])
            ymin = min(y, self.drag_start[1])
            xmax = max(x, self.drag_start[0])
            ymax = max(y, self.drag_start[1])
            self.selection = (xmin, ymin, xmax, ymax)
        if event == cv.EVENT_LBUTTONUP:
            self.drag_start = None
            self.track_window = (xmin, ymin, xmax - xmin, ymax - ymin)

    def show_hist(self):
        bin_count = self.hist.shape[0]
        bin_w = 24
        img = np.zeros((256, bin_count*bin_w, 3), np.uint8)
        for i in xrange(bin_count):
            h = int(self.hist[i])
            cv.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h),
                         (int(180.0*i/bin_count), 255, 255), -1)
        img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
        cv.imshow('hist', img)

    def preprocess(self, frame):
        # Apply bilateral filter to reduce noise but keep edges
        filtered = cv.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
        hsv = cv.cvtColor(filtered, cv.COLOR_BGR2HSV)

        h, s, v = cv.split(hsv)

        # Adaptive thresholds for S and V with some margin for lighting variation
        # Here we set saturation threshold moderately high but allow low saturation pixels if hue is strong
        s_thresh_low = 40
        v_thresh_low = 40

        # Create masks with thresholds
        mask_s = cv.inRange(s, s_thresh_low, 255)
        mask_v = cv.inRange(v, v_thresh_low, 255)

        # Combine masks - keep pixels with sufficient saturation and brightness
        mask = cv.bitwise_and(mask_s, mask_v)

        # Morphological operations to clean up noise
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

        return hsv, mask

    def run(self):
        while True:
            _ret, self.frame = self.cam.read()
            vis = self.frame.copy()

            hsv, mask = self.preprocess(self.frame)

            if self.selection:
                x0, y0, x1, y1 = self.selection
                hsv_roi = hsv[y0:y1, x0:x1]
                mask_roi = mask[y0:y1, x0:x1]
                hist = cv.calcHist([hsv_roi], [0], mask_roi, [16], [0, 180])
                cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)
                self.hist = hist.reshape(-1)
                self.show_hist()

                vis_roi = vis[y0:y1, x0:x1]
                cv.bitwise_not(vis_roi, vis_roi)
                vis[mask == 0] = 0

            if self.track_window and self.track_window[2] > 0 and self.track_window[3] > 0:
                self.selection = None
                prob = cv.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
                prob &= mask
                term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
                track_box, self.track_window = cv.CamShift(prob, self.track_window, term_crit)

                if self.show_backproj:
                    vis[:] = prob[..., np.newaxis]
                try:
                    cv.ellipse(vis, track_box, (0, 0, 255), 2)
                except Exception as e:
                    print('CamShift ellipse error:', e)

            cv.imshow('camshift', vis)

            ch = cv.waitKey(5)
            if ch == 27:
                break
            if ch == ord('b'):
                self.show_backproj = not self.show_backproj

        cv.destroyAllWindows()


if __name__ == '__main__':
    print(__doc__)
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0
    App(video_src).run()
