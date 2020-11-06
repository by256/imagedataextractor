import cv2
import numpy as np


def get_contours(x):
    contours = cv2.findContours(x.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 2:
        contours = contours[0][0]
    elif len(contours) == 3:
        contours = contours[1][0]
    return contours

def shuffle_segmap(x):
    insts = np.unique(x)
    shuffle_idx = np.arange(1, len(insts))
    np.random.shuffle(shuffle_idx)
    shuffled_segmap = np.zeros_like(x)
    for i, idx in enumerate(shuffle_idx):
        inst_mask = x == idx
        shuffled_segmap[inst_mask] = i + 1
        
    return shuffled_segmap