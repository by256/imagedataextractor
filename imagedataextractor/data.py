import pandas as pd
from rdfpy import rdf2d


class EMData:

    def __init__(self):
        self.data = {}
        self.segmentation = None
        self.uncertainty = None
        self.scalebar = None

    def __repr__(self):
        pass

    def to_pandas(self):
        pass

    def save(self, path):
        pass

    def compute_sizehist(self, bins=None):
        pass

    def compute_rdf(self, dr=None):
        pass
