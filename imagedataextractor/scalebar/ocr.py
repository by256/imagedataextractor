import cv2
import numpy as np
import pytesseract

"""
This file will contain all code required to perform OCR.

tesseract documentation - https://github.com/tesseract-ocr/tessdoc

"""


def detect_bounding_boxes(image):
    # data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    custom_config = '--oem 1'
    # boxes = pytesseract.image_to_boxes(image, config=custom_config)
    # boxes = boxes.splitlines()
    # boxes = [box.split(' ')[1:-1] for box in boxes]
    # boxes = list(map(lambda x: list(map(int, x)), boxes))  # convert nested list of str to int

    boxes = pytesseract.image_to_string(image, config=custom_config)

    return boxes

def perform_ocr(x):
    pass