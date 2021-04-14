import os
import unittest
import logging
import numpy as np
from PIL import Image

from imagedataextractor.scalebar import ScalebarDetector


class TestScalebarDetection(unittest.TestCase):

    def test_scalebar_detection(self):
        test_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/testimage.png')
        test_image = np.array(Image.open(test_image_path))

        expected_text = '50 nm'
        expected_units = 'nm'

        scalebar_detector = ScalebarDetector()
        scalebar = scalebar_detector.detect(test_image)
        text, units, conversion, scalebar_contour = scalebar.data

        self.assertEqual(text, expected_text)
        self.assertEqual(units, expected_units)
