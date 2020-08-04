import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from scalebar.ocr import OCR
from scalebar.textdetection import TextDetector


# image = Image.open('../examples/10 nm.png')  # y (n with resize)
# image = Image.open('../examples/50 nm.png')  # y (with --psm 6)
# image = Image.open('../examples/200 nm.png')  # y
# image = Image.open('../examples/20 nm.png')  # n
# image = Image.open('../examples/20 nm(2).png')  # y (with thresholding)
# image = Image.open('../examples/100 nm.png')  # n
# image = Image.open('../examples/200 nm(2).png')  # y 
# image = Image.open('../examples/500 nm.png')  # y (somehow)
# image = Image.open('../examples/50 nm(2).png')  # y (easy)
# image = Image.open('../examples/2 um.png')  # y
# image = Image.open('../examples/1 um.png')  # n
# image = Image.open('../examples/5 um.png')  # n
# image = Image.open('../examples/100 um.png')  # n
# image = Image.open('../examples/200 um.png')  # n
image = Image.open('../examples/1 um(2).png')  #

print('original image', np.array(image).shape)
h, w = np.array(image).shape[:2]
new_h = int(h - (h % 32))
new_w = int(w - (w % 32))
print('new image', new_h, new_w)


image = image.resize((new_w, new_h), resample=Image.BICUBIC)
image = np.array(image)
plt.imshow(image)
plt.show()
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