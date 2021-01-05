# -*- coding: utf-8 -*-
"""
Main extraction modules for imagedataextractor.

.. codeauthor:: Batuhan Yildirim <by256@cam.ac.uk>
"""

import os
import cv2
import torch
import imghdr
import logging
import numpy as np
import pandas as pd
from chemdataextractor import Document

from .data import EMData
from .figsplit import figsplit
from .analysis import ShapeDetector
from .analysis.particlesize import aspect_ratio
from .scalebar import ScalebarDetector
from .segment import ParticleSegmenter
from .utils import shuffle_segmap

log = logging.getLogger(__name__)


def extract(input_path, seg_bayesian=True, seg_n_samples=30, seg_tu=0.0125, seg_device='cpu'):
    """Extract from single image, single doc, directory of images, or directory of docs."""
    allowed_doc_exts = ['.html', '.xml', '.pdf']
    # single image
    if os.path.isfile(input_path):
        if imghdr.what(input_path) is not None:
            log.info('Input is an image of type {}.'.format(imghdr.what(input_path)))
            image = cv2.imread(input_path)
            images = figsplit(image)
            data = [_extract_image(im, seg_bayesian, seg_n_samples, seg_tu, seg_device) for im in images]
    # single document
    elif os.path.splitext(input_path)[1] in allowed_doc_exts:
        log.info('Input is a document.')
        extract_document()
    # directory of images or documents
    elif os.path.isdir(input_path):
        log.info('Input is a directory of images/documents.')
        data = []
        for f in os.listdir(input_path):
            file_path = os.path.join(input_path, f)
            file_ext = os.path.splitext(file_path)[-1]
            # image
            if imghdr.what(file_path) is not None:
                image = cv2.imread(file_path)
                images = figsplit(image)
                data.append([_extract_image(im, seg_bayesian, seg_n_samples, seg_tu, seg_device) for im in images])
            # document
            elif file_ext in allowed_doc_exts:
                extract_document()
        data = [item for sublist in data for item in sublist]  # flatten
    else:
        error_msg = 'Input is invalid. Provide a path to an image, a path to a document of type {}, or a path to a directory of images and/or documents.'.format(allowed_doc_exts[:2])
        raise TypeError(error_msg)
    return data

def _extract_image(image, seg_bayesian=True, seg_n_samples=30, seg_tu=0.0125, seg_device='cpu'):
    """
    Extract from a single image (not a panel).
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # initialise detectors
    sb_detector = ScalebarDetector()
    segmenter = ParticleSegmenter(bayesian=seg_bayesian, n_samples=seg_n_samples, tu=seg_tu, device=seg_device)
    shape_detector = ShapeDetector()

    # initialise EM data object
    em_data = EMData()
    if not seg_bayesian:
        del em_data.data['uncertainty']

    # detect scalebar
    scalebar = sb_detector.detect(image)
    em_data.scalebar = scalebar
    text, units, conversion, scalebar_contour = scalebar.data
    if scalebar_contour is not None:
        log.info('Scalebar detection successful.')
    else:
        log.info('Scalebar detection failed. Measurements will be given in units of pixels.')

    # segment particles
    segmentation, uncertainty, original = segmenter.segment(image)
    segmentation = shuffle_segmap(segmentation)
    em_data.segmentation = segmentation
    em_data.uncertainty = uncertainty
    if len(np.unique(segmentation)) > 1:
        log.info('Particle segmentation successful.')
    else:
        log.info('Particle segmentation was completed but no particles were found.')

    # extract particle measures
    for inst in np.unique(segmentation):
        if inst == 0.0:  # 0 is always background
            continue
        em_data.data['idx'].append(inst)
        em_data.data['original_units'].append(units)

        inst_mask = (segmentation == inst).astype(np.uint8)

        # area
        area = np.sum(inst_mask)  # pixels
        em_data.data['area (pixels^2)'].append(area)
        if conversion is not None:
            area = area * conversion**2  # meters^2
            em_data.data['area_units'].append('meters^2')
            em_data.data['diameter_units'].append('meters')
        else:
            em_data.data['area_units'].append('pixels^2')
            em_data.data['diameter_units'].append('pixels')
        em_data.data['area'].append(area)

        # center
        coords = np.argwhere(inst_mask == 1.0)
        center = coords.mean(axis=0, dtype=int)  # (y, x)
        em_data.data['center'].append(center)

        # edge
        edge_cond = (0 in coords) | (image.shape[0]-1 in coords[:, 0]) | (image.shape[1]-1 in coords[:, 1])
        em_data.data['edge'].append(edge_cond)

        # contours
        contours, _ = cv2.findContours(inst_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        em_data.data['contours'].append(contours)

        # aspect ratio
        ar = aspect_ratio(contours[0])
        em_data.data['aspect_ratio'].append(ar)

        # shape estimate
        shape_estimate, shape_distances = shape_detector.match_shapes(inst_mask)
        if shape_estimate == 'circle':
            em_data.data['shape_estimate'].append(shape_estimate)
            # diameter
            diameter = 2*np.sqrt(area/np.pi)
            em_data.data['diameter'].append(diameter)
        else:
            em_data.data['shape_estimate'].append(None)
            em_data.data['diameter'].append(None)

        # particle instance uncertainty
        if seg_bayesian:
            inst_uncertainty = np.mean(uncertainty[inst_mask])
            em_data.data['uncertainty'].append(inst_uncertainty)
        
    if len(em_data) > 0:
        log.info('Extraction successful - Found {} particles.'.format(len(em_data)))
    else:
        log.info('Extraction failed - no particles were found.')

    return em_data

def extract_document():
    """Extract from single document."""
    raise NotImplementedError('Extraction from documents will be implemented upon the release of CDE 2.0. Please use image extraction instead.')
