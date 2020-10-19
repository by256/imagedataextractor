import numpy as np


def edge_filter(x):
    """
    Removes particle segmentations that exist on the edge of the image.
    """
    h, w = x.shape
    inst_ids = np.unique(x)
    inst_ids = inst_ids[inst_ids > 0]
    
    filtered = np.zeros_like(x)
    
    for inst_id in inst_ids:
        inst_mask = x == inst_id
        coords = np.stack(np.where(inst_mask), axis=1)
        edge_cond = (0 in coords) & (h-1 in coords[:, 0]) & (w-1 in coords[:, 1])
        if not edge_cond:
            filtered[inst_mask] = inst_id
    return filtered
 