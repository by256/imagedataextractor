# -*- coding: utf-8 -*-
"""
General utils.

.. codeauthor:: Batuhan Yildirim <by256@cam.ac.uk>
"""

import os
import cv2
import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


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

def visualize_seg(data, save_to):
    seg = data.segmentation.astype(float)

    seg_cmap = copy.copy(matplotlib.cm.tab20)
    seg_cmap.set_bad(color='k')  # black

    if len(np.unique(seg)) > 1:
        seg[seg == 0.0] = np.nan
        plt.imsave(os.path.join(save_to, '{}-seg.png'.format(data.fn)), seg, cmap=seg_cmap)
    else:
        plt.imsave(os.path.join(save_to, '{}-seg.png'.format(data.fn)), seg, cmap='gray')
