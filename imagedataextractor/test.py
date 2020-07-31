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

for x1, y1, x2, y2 in boxes:
    print(x1, y1, x2, y2)
    box_w = x2 - x1
    box_h = y2 - y1
    print(box_w, box_h)
    roi_image = image[y1-box_h:y2+box_h, x1-box_w:x2+box_w]
    plt.imshow(roi_image)
    plt.show()
