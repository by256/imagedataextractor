# -*- coding: utf-8 -*-
"""
ImageDataExtractor
Microscopy image quantification.
by256@cam.ac.uk
~~~~~~~~~~~~~~~
"""

import logging
from .extract import *
from .figsplit import figsplit

__title__ = 'ImageDataExtractor'
__version__ = '2.0.0'
__author__ = 'Batuhan Yildirim'
__email__ = 'by256@cam.ac.uk'
__license__ = 'MIT License'

logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')
