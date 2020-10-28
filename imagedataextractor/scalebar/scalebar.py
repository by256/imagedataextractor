import re
import cv2
import numpy as np
from .ocr import OCR
from .textdetection import TextDetector


class ScalebarDetector:

    def __init__(self):
        self.text_detector = TextDetector()
        self.ocr = OCR()
        self.conversion_dict = {'nm': 1e-9, 
                                'um': 1e-6, 
                                'mm': 1e-3}

    # def filter_lines(self, lines, image_w):
    #     filtered_lines = []
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         line_w = x2 - x1
    #         if line_w <= 0.8*image_w and line_w != 0:
    #             filtered_lines.append(line[0])
        
    #     # sort lines by length
    #     line_lengths = np.array([x[2]-x[0] for x in filtered_lines])
    #     sort_idx = np.argsort(line_lengths)[::-1]
    #     filtered_lines = np.array(filtered_lines)[sort_idx]
        
    #     return filtered_lines

    # def get_scalebar_line(self, roi, resize_factor=3):
    #     h, w = roi.shape[:2]
    #     if len(roi.shape) == 2:
    #         roi = np.stack([roi]*3, axis=-1)
    #     roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #     roi_0 = cv2.resize(roi, (w*resize_factor, h*resize_factor), interpolation=cv2.INTER_CUBIC)
    #     roi = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5)).apply(roi_0.astype(np.uint8))
    #     import matplotlib.pyplot as plt
    #     # fig, axes = plt.subplots(1, 2)
    #     # axes[0].imshow(roi_0, cmap='gray')
    #     # axes[1].imshow(roi, cmap='gray')
    #     # plt.show()
    #     # gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #     edges = cv2.Canny(roi, 50, 150, apertureSize=3, L2gradient=True)

    #     lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40, 100, 5)
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         cv2.line(roi, (x1,y1), (x2,y2), (0,255,0), thickness=2)
    #     plt.imshow(roi)
    #     plt.show()
    #     if (lines is not None) and (len(lines) > 0):
    #         filtered_lines = self.filter_lines(lines, w)
    #     if len(filtered_lines) > 0:
    #         return (filtered_lines[0]/resize_factor).astype(int)
    #     else:
    #         return None

    def filter_scalebar_contours(self, contours, size_ratio=4, min_height=5):

        valid_contours = []
        valid_contour_widths = []

        for contour in contours:
            _, _, w, h = cv2.boundingRect(contour)
            if (w > size_ratio*h) and (h > min_height):
                valid_contours.append(contour)
                valid_contour_widths.append(w)
        
        if len(valid_contours) == 0:
            scalebar_contour = None
            scalebar_width = None
        else:
            min_idx = np.argmin(valid_contour_widths)
            scalebar_contour = valid_contours[min_idx]
            scalebar_width = valid_contour_widths[min_idx]
        return scalebar_contour, scalebar_width

    def get_scalebar_line(self, roi, resize_factor=3):
        h, w = roi.shape[:2]
        if len(roi.shape) == 2:
            roi = np.stack([roi]*3, axis=-1)
        roi = cv2.resize(roi, (w*resize_factor, h*resize_factor), interpolation=cv2.INTER_CUBIC)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # roi = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5)).apply(roi.astype(np.uint8))
        _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # maybe invert here, if there are more black pixels than white pixels
        # this is to ensure background is white and scalebar is black
        n_white = np.sum((roi == 255).astype(int))
        n_black = np.sum((roi == 0).astype(int))
        if n_black > n_white:
            roi = 255 - roi
        contours, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        scalebar_contour, scalebar_width = self.filter_scalebar_contours(contours)
        if scalebar_contour is not None:
            import matplotlib.pyplot as plt
            # for i, contour in enumerate(contours):
            
            roi = np.stack([roi, roi, roi], axis=-1)
            x, y, w, h = cv2.boundingRect(scalebar_contour)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
            plt.imshow(roi)#, cmap='gray')
            plt.title('scalebar = {} pixels'.format(scalebar_width))
            plt.show()

        return None

    def parse_scalebar_text(self, text):
        value = float(re.split(str('[^\d+]'), text)[0])
        unit = re.split(str('\d+\s*'), text)[-1]
        return value, unit

    def detect(self, image):
        
        rois = self.text_detector.get_text_rois(image)
        text, best_idx = self.ocr.get_text_from_rois(rois)
        if text is not None:
            print('rois[best_idx]', rois[best_idx].shape)
            line = self.get_scalebar_line(rois[best_idx])
            im = rois[best_idx]
            if line is not None:
                x1, y1, x2, y2 = line
                cv2.line(im, (x1,y1), (x2,y2), (0,255,0), thickness=2)
            import matplotlib.pyplot as plt
            plt.imshow(im)
            plt.show()
            if line is not None:
                print('line', line)
                scale_bar_len = line[2] - line[0]
                print('scale_bar_len', scale_bar_len)
                value, units = self.parse_scalebar_text(text)
                conversion = value / scale_bar_len
            else:
                conversion = None
            return text, conversion, line
        else:
            return None, None, None