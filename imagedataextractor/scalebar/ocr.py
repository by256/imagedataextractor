import re
import cv2
import numpy as np
import pytesseract
from PIL import Image

from .textdetection import TextDetector

"""
This file will contain all code required to perform OCR.

"""


class OCR:

    def __init__(self):
        self.text_detector = TextDetector()
        self.valid_text_expression = r'[1-9]\d+\s+(μ|n)m'

    def perform_ocr(self, image: np.ndarray, custom_config: str=None) -> str:
        h, w = image.shape[:2]
        image = Image.fromarray(image).resize((int(w*2), int(h*2)), resample=Image.BICUBIC)
        image = np.array(image)
        text = pytesseract.image_to_string(image, config=custom_config)
        print('ocr text', text)
        match = re.search(self.valid_text_expression, text)
        if match:
            text = match[0]
        else:
            text = None
        return text

    def get_scalebar_text(self, image: np.ndarray):
        valid_match = False
        rois = self.text_detector.get_text_rois(image)
        for roi in rois:
            text = self.perform_ocr(roi)
            if text is not None:
                valid_match = True
            import matplotlib.pyplot as plt
            plt.imshow(roi)
            plt.title(text)
            plt.show()
            print(text)
            if valid_match:
                break
