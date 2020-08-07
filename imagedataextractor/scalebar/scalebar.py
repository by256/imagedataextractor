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

    def filter_lines(self, lines, image_w):
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_w = x2 - x1
            if line_w <= 0.8*image_w:
                filtered_lines.append(line[0])
        
        # sort lines by length
        line_lengths = np.array([x[2]-x[0] for x in filtered_lines])
        sort_idx = np.argsort(line_lengths)[::-1]
        filtered_lines = np.array(filtered_lines)[sort_idx]
        
        return filtered_lines

    def get_scalebar_line(self, roi):
        h, w = roi.shape[:2]
        if len(roi.shape) == 2:
            roi = np.stack([roi]*3, axis=-1)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3, L2gradient=True)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40, 100, 5)
        if (lines is not None) and (len(lines) > 0):
            lines = self.filter_lines(lines, w)
        return lines[0]

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
            scale_bar_len = line[2] - line[0]
            
            value, units = self.parse_scalebar_text(text)
            conversion = value / scale_bar_len
            return text, conversion, line
        else:
            return None, None, None