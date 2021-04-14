import os
import unittest
import logging
import numpy as np
from PIL import Image
import imagedataextractor as ide
from imagedataextractor.scalebar import ScalebarDetector


class TestExtraction(unittest.TestCase):
    """Test full pipeline for any errors/exceptions."""
    def test_extraction(self):
        test_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/full-pipeline-image.png')
        
        # test in Bayesian mode.
        data = ide.extract(test_image_path, 
                   seg_bayesian=True,  # Bayesian mode
                   seg_tu=0.0125,  # uncertainty threshold beyond which to filter FPs
                   seg_n_samples=50,  # number of monte carlo samples for Bayesian inference
                   seg_device='cpu'  # set to 'cuda' to utilise GPU.
                  )
        # test in discriminative mode.
        data = ide.extract(test_image_path, 
                   seg_bayesian=False,
                   seg_device='cpu'  # set to 'cuda' to utilise GPU.
                  )
