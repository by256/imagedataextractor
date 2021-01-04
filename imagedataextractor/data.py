import numpy as np
import pandas as pd
from rdfpy import rdf2d


class EMData:

    def __init__(self):
        self.data = {'idx': [], 
                     'center': [], 
                     'edge': [],
                     'contours': [],
                     'area (pixels^2)': [],
                     'area': [], 
                     'area_units': [],
                     'aspect_ratio': [], 
                     'shape_estimate': [],
                     'diameter': [], 
                     'diameter_units': [],
                     'original_units': [], 
                     'uncertainty': []}
        self.segmentation = None
        self.uncertainty = None
        self.scalebar = None
        self.rdf = {'g_r': None, 'radii': None}

    def __repr__(self):
        pass

    def __len__(self):
        return len(self.data['idx'])

    def to_pandas(self):
        pass

    def save(self, path):
        pass

    def compute_sizehist(self, bins=None):
        pass

    def compute_rdf(self, dr=None):
        if not dr:
            dr = np.sqrt(np.mean(self.data['area (pixels^2)'])) / 4
        
        # compute rdf
        centers = np.array(list(self.data['center']))
        g_r, radii = rdf2d(centers, dr)
        self.rdf['g_r'] = g_r
        self.rdf['radii'] = radii
        
        return g_r, radii

    @property
    def rdf(self):
        return self.rdf['g_r'], self.rdf['radii']
