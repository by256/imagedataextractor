import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from scalebar.ocr import *


image = Image.open('../examples/10nm.png')
image = np.array(image)
h, w = image.shape[:2]

image = image[h - h//5:, w - w//5:]
h, w = image.shape[:2]
image = Image.fromarray(image).resize((w*3, h*3), resample=Image.BICUBIC)


boxes = detect_bounding_boxes(image)

# a, b, c, d = boxes[0]

# print(a)
# print(h - b)
# print(c)
# print(h - d)

# image = cv2.rectangle(image, (w-a, h-b, c, d), (0, 255, 0), 2)
plt.imshow(image)
plt.show()

print(boxes)