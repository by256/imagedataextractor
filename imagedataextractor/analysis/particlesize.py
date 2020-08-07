import numpy as np


def particle_size_histogram(segmap, bins=10):
    instances = np.unique(segmap)
    instances = instances[instances > 0]

    sizes = []

    for instance in instances:
        area = np.sum((segmap == instance).astype(int))
        radius = np.sqrt(area/np.pi)  # assume circle (for now)
        sizes.append(radius)

    hist, bins = np.histogram(sizes, bins=bins)
    return hist, bins
