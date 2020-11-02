import cv2
import numpy as np


def aspect_ratio(contour):
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w/h
    return aspect_ratio
