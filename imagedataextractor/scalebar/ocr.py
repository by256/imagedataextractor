import cv2
import numpy as np
import pytesseract
from PIL import Image

"""
This file will contain all code required to perform OCR.

tesseract documentation - https://github.com/tesseract-ocr/tessdoc

"""


def perform_ocr(image: np.ndarray, custom_config: str=None) -> str:
    h, w = image.shape[:2]
    image = Image.fromarray(image).resize((int(w*2.5), int(h*2.5)), resample=Image.BICUBIC)
    image = np.array(image)
    text = pytesseract.image_to_string(image, config=custom_config)
    return text