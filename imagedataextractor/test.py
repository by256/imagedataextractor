import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from scalebar.ocr import OCR
from scalebar.textdetection import TextDetector


# image = Image.open('../examples/10 nm.png')  # y
image = Image.open('../examples/50 nm.png')  # n
# image = Image.open('../examples/200 nm.png')  # y
# image = Image.open('../examples/20 nm.png')  # n
# image = Image.open('../examples/20 nm(2).png')  # n


image = image.resize((512, 512), resample=Image.BICUBIC)
image = np.array(image)
h, w = image.shape[:2]

ocr = OCR()
ocr.get_scalebar_text(image)

# detector = TextDetector()
# print('image', type(image))
# # boxes = detector.detect_text(image)

# rois = detector.get_text_rois(image)
# for roi in rois:
#     valid_match = False
#     text = perform_ocr(roi)
#     if text:
#         print(text, len(text))
#         plt.imshow(roi)
#         plt.title(text)
#         plt.show()
#         # break