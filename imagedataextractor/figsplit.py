# -*- coding: utf-8 -*-
"""
Figure splitting procedures.

.. codeauthor:: Batuhan Yildirim <by256@cam.ac.uk>
"""

import numpy as np
from skimage.color import rgb2gray
from skimage.measure import label, regionprops


def figsplit(image, min_edge=0.05, min_fill=0.8, t=0.9):
    """
    Extracts constituent images from a panel of images typically 
    found in figures in scientific literature. This was adapted 
    from code written by Matt Swain.

    Parameters
    ----------
    image: np.ndarray
        Image to perform extraction on.
    min_edge: float
        Minimum edge (default is 0.05).
    min_fill: float
        Minimum fill (default is 0.8).
    t: float
        Binarization threshold (default is 0.9).
    """
    images = []

    # convert to grayscale \in [0, 1]
    gray = rgb2gray(image)
    # Binarize with high thresh (default 0.9)
    binary = (gray > t).astype(np.uint8)

    min_length_px = np.around(min_edge * min(binary.shape))

    label_image = label(1 - binary)
    for region in regionprops(label_image):
        # Check if region is large enough
        top, left, bottom, right = region.bbox
        if right - left > min_length_px and bottom - top > min_length_px:

            # Check if region is 'filled' enough
            if region.extent > min_fill:
                images.append(image[top:bottom, left:right])

    return images
