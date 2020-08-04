import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from scalebar.ocr import *
from scalebar.textdetection import TextDetector


image = Image.open('../examples/10nm.png')
image = np.array(image)
h, w = image.shape[:2]

detector = TextDetector()
print('image', type(image))
# boxes = detector.detect_text(image)

rois = detector.get_text_rois(image)
for roi in rois:
    plt.imshow(roi)
    plt.show()