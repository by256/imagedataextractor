import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from scalebar.ocr import *


image = Image.open('../examples/10nm.png')
image = np.array(image)
h, w = image.shape[:2]

detector = TextDetector()
boxes = detector.detect_text(image)
