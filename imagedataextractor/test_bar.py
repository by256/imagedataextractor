import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from scalebar import ScalebarDetector
from scalebar.textdetection import TextDetector


def filter_lines(lines, image_w):
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        line_w = x2 - x1
        if line_w <= 0.8*image_w:
            filtered_lines.append(line[0])
    
    # sort lines by length
    line_lengths = np.array([x[2]-x[0] for x in filtered_lines])
    sort_idx = np.argsort(line_lengths)[::-1]
    filtered_lines = np.array(filtered_lines)[sort_idx]
    
    return filtered_lines


base_path = '../examples/'
image_paths = [base_path + x for x in os.listdir(base_path) if x.endswith('.png')][2:]

# text_detector = TextDetector()

# for path in image_paths:
#     image = cv2.imread(path)
#     rois = text_detector.get_text_rois(image)
#     if rois:
#         image = rois[0]
#         h, w = image.shape[:2]
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         edges = cv2.Canny(gray, 50, 150, apertureSize=3, L2gradient=True)

#         lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40, 100, 5)
#         if (lines is not None) and (len(lines) > 0):
#             lines = filter_lines(lines, w)
#             print('line', lines[0][2] - lines[0][0])
#             print(lines[0])

#             x1, y1, x2, y2  = lines[0]
#             cv2.line(image, (x1,y1), (x2,y2), (0,255,0), 2)

#             plt.imshow(image)
#             plt.show()
    # break

scalebar_detector = ScalebarDetector()

for i, path in enumerate(image_paths):
    image = cv2.imread(path)
    text, conversion, line = scalebar_detector.detect(image)
    print(text)
    # break