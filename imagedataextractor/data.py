# -*- coding: utf-8 -*-
"""
Electron microscopy data class.

.. codeauthor:: Batuhan Yildirim <by256@cam.ac.uk>
"""

import os
import warnings
import numpy as np
import pandas as pd
from rdfpy import rdf2d


class EMData:
    """
    Main ImageDataExtractor data object that contains data once it has been extracted.
    Also includes methods to compute radial distribution function and particle size 
    histogram from extracted data.
    """

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
        self.fn = None
        self.image = None
        self.segmentation = None
        self.uncertainty = None
        self.scalebar = None
        self.rdf = {'g_r': None, 'radii': None}

    def __repr__(self):
        return 'EMData({} particles.)'.format(len(self.data['idx']))

    def __len__(self):
        return len(self.data['idx'])

    def to_pandas(self):
        return pd.DataFrame(self.data)

    def to_csv(self, path):
        if not path.endswith('.csv'):
            path = os.path.splitext(path)[0]
            path = os.path.join(path, '.csv')
        self.to_pandas().to_csv(path, index=False)

    @property
    def valid_sizes(self):
        valid_idx = np.bitwise_not(self.data['edge'])
        return np.array(self.data['area'])[valid_idx]

    def compute_sizehist(self, bins=None):
        if self.__len__() < 15:
            warnings.warn('Less than 15 particles were extracted. Resulting histogram is likely to be incorrect.')

        if not bins:
            bins = 'rice'
        counts, bin_edges = np.histogram(self.valid_sizes, bins=bins)
        return counts, bin_edges

    @property
    def coords(self):
        return np.array(self.data['center'])

    @property
    def valid_coords(self):
        valid_idx = np.bitwise_not(self.data['edge'])
        return np.array(self.data['center'])[valid_idx]

    def compute_rdf(self, dr=None, **rdf_kwargs):
        if self.__len__() < 40:
            warnings.warn('Less than 40 particles were extracted. Resulting RDF is likely to be incorrect.')
        if not dr:
            dr = np.sqrt(np.mean(self.data['area (pixels^2)'])) / 4
        # compute rdf
        g_r, radii = rdf2d(self.valid_coords, dr)
        self.rdf['g_r'] = g_r
        self.rdf['radii'] = radii
        
        return g_r, radii
