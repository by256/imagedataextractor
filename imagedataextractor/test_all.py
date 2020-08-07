import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from scalebar import ScalebarDetector
from scalebar.ocr import OCR
from segment.model import ParticleSegmentation
from analysis import particle_size_hist


base_path = '../examples/'
image_paths = [base_path + x for x in os.listdir(base_path) if x.endswith('.png')][4:]

for i, path in enumerate(image_paths):
    image = cv2.imread(path)

    # image = np.array(Image.open('../examples/20 nm(2).png'))

    ocr = OCR()
    scalebar_detector = ScalebarDetector()
    model = ParticleSegmentation(device='cpu')

    # segment
    pred = model.segment(image)

    #scalebar
    text, conversion, line = scalebar_detector.detect(image)
    if (text or conversion or line):
        x1, y1, x2, y2  = line
        cv2.line(image, (x1,y1), (x2,y2), (0,255,0), 2)

        # analysis
        sizes = particle_size_hist(pred, conversion)

        fig, axes = plt.subplots(1, 3)
        axes[0].imshow(image)
        axes[1].imshow(pred, cmap='tab20')
        axes[2].hist(sizes, bins=20)
        plt.suptitle(text)

        # x_ticks = axes[0].get_xticks()
        # print('x', axes[0].get_xticks())
        # print('y', axes[0].get_yticks())

        # y_min, y_max = axes[0].get_ylim()
        # x_min, x_max = axes[0].get_xlim()
        # axes[0].set_ylim(y_min*conversion, y_max*conversion)
        # axes[0].set_xlim(x_min*conversion, x_max*conversion)

        plt.show()
