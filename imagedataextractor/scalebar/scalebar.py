# -*- coding: utf-8 -*-
"""
Module for performing scalebar detection.

.. codeauthor:: Batuhan Yildirim <by256@cam.ac.uk>
"""

import re
import cv2
import numpy as np
from .ocr import OCR
from .textdetection import TextDetector
from .utils import get_contours


class Scalebar:

    def __init__(self, text=None, units=None, conversion=None, contour=None):
        """Main data object in which scalebar data is stored."""
        self.text = text
        self.units = units
        self.conversion = conversion
        self.contour = contour

    def __repr__(self):
        if self.text is not None:
            repr_ =  'Scalebar: {} {}    {} {} per pixel.'.format(self.text, self.units, self.conversion, self.units)
        else:
            repr_ = 'Scalebar: Not detected.'

    @property
    def data(self):
        return self.text, self.units, self.conversion, self.contour


class ScalebarDetector:

    def __init__(self):
        """Scalebar detector."""
        self.text_detector = TextDetector()
        self.ocr = OCR()
        self.conversion_dict = {'nm': 1e-9, 
                                'um': 1e-6, 
                                'mm': 1e-3}

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
        """
        Obtains scalebar line by finding rectangular contours.

        Parameters
        ----------
        roi: np.ndarray
            image region-of-interest from which to obtain scalebar line.
        resize_factor: int
            factor by which to scale ROI for more accurate scalebar line localization.
        """

        h, w = roi.shape[:2]
        if len(roi.shape) == 2:
            roi = np.stack([roi]*3, axis=-1)
        roi = cv2.resize(roi, (w*resize_factor, h*resize_factor), interpolation=cv2.INTER_CUBIC)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # if there are more black pixels than white pixels, invert (helps with sb contour finding).
        n_white = np.sum((roi == 255).astype(int))
        n_black = np.sum((roi == 0).astype(int))
        if n_black > n_white:
            roi = 255 - roi
        contours, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours = get_contours(roi)
        scalebar_contour, scalebar_width = self.filter_scalebar_contours(contours)
        
        if scalebar_contour is not None:
            scalebar_contour = (scalebar_contour / resize_factor).astype(int)
            scalebar_width = scalebar_width / resize_factor
        return scalebar_contour, scalebar_width

    def parse_scalebar_text(self, text):
        """Parse and split scalebar text into value and units."""
        value = float(re.split(str('[^\d+]'), text)[0])
        unit = re.split(str('\d+\s*'), text)[-1]
        return value, unit

    def detect(self, image):
        """Detect and return scalebar from an input microscopy image."""
        
        # initialise scalebar
        scalebar = Scalebar()

        rois, roi_locs = self.text_detector.get_text_rois(image)
        text, best_idx = self.ocr.get_text_from_rois(rois)

        scalebar_contour = None
        conversion = None
        units = None

        if text is not None:
            scalebar_contour, scalebar_width = self.get_scalebar_line(rois[best_idx])
        if scalebar_contour is not None:
            value, units = self.parse_scalebar_text(text)
            # move scalebar rect to pos in original image
            scalebar_contour = scalebar_contour + np.array([roi_locs[best_idx][0], roi_locs[best_idx][1]])
            # convert pixels to meters
            conversion = value / scalebar_width
            conversion = conversion * self.conversion_dict[units]
        
        scalebar.text = text
        scalebar.units = units
        scalebar.conversion = conversion
        scalebar.scalebar_contour = scalebar_contour

        return scalebar
