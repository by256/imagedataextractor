import cv2
import numpy as np


def aspect_ratio(contour):
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w/h
    return aspect_ratio

def particle_size_hist(segmap, conversion, nbins=10, density=False):
    instances = np.unique(segmap)
    print(instances)
    instances = instances[instances > 0]

    sizes = []

    for instance in instances:
        area = np.sum((segmap == instance).astype(int))
        radius = np.sqrt(area/np.pi)  # assume circle (for now)
        radius = radius * conversion
        sizes.append(radius)
    
    return sizes
    # print(sizes)

    # hist, bins = np.histogram(sizes, bins=nbins, density=density)
    # return hist, bins
