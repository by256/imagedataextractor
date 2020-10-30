import os
import imghdr
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from chemdataextractor import Document

from analysis import ShapeDetector
from analysis.filtering import edge_filter
from analysis.particlesize import aspect_ratio
from scalebar import ScalebarDetector
from segment import ParticleSegmenter


def extract(inputs, out_dir, bayesian=True, device='cpu'):
    
    # check wether inputs are images, or papers
    for inp in inputs:
        # if inputs are image paths
        if imghdr.what(inp) is not None:
            im_path = inp
            image = cv2.imread(im_path)

            file_ext = imghdr.what(im_path)
            fn = im_path.split('/')[-1].split('.'+file_ext)[0]
            target_dir = os.path.join(out_dir, fn)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            extract_image(image, target_dir=target_dir, bayesian=bayesian)
        else:
            # else documents, therefore use chemdataextractor
            cde_retrieve_images(inp)

def cde_retrieve_images(doc_path):
    raise NotImplementedError('This will be implemented upon the release of CDE 2.0.')

def extract_image(image, target_dir, bayesian=True, min_particles=10):

    

    # initialise detectors
    sb_detector = ScalebarDetector()
    segmenter = ParticleSegmenter(bayesian=bayesian)
    shape_detector = ShapeDetector()

    output_image = image.copy()

    # detect scalebar
    text, units, conversion, scalebar_contour = sb_detector.detect(image)
    # draw scalebar on output image
    if scalebar_contour is not None:
        x, y, w, h = cv2.boundingRect(scalebar_contour)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (24, 24, 205), 1)

    # segment particles
    particle_preds, uncertainty = segmenter.segment(image)
    particle_preds = particle_preds.astype(float)

    particles = []
    
    # compute individual particle measures
    for inst in np.unique(particle_preds):
        if inst == 0.0:  # 0 is always background
            continue
        # initialise particle records
        particle_data = {'idx': inst, 
                        'center': None, 
                        'edge': None,
                        'contours': None,
                        'area': None, 
                        'area_units': None,
                        'aspect_ratio': None, 
                        'shape_estimate': None,
                        'diameter': None, 
                        'diameter_units': None,
                        'original_units': units, 
                        'uncertainty': None}
        
        inst_mask = (particle_preds == inst).astype(np.uint8)

        # area
        area = np.sum(inst_mask)  # pixels
        if conversion is not None:
            area = area * conversion**2  # meters^2
            particle_data['area_units'] = 'meters^2'
            particle_data['diameter_units'] = 'meters'
        else:
            particle_data['area_units'] = 'pixels^2'
            particle_data['diameter_units'] = 'pixels'
        particle_data['area'] = area
        # center
        coords = np.argwhere(inst_mask == 1.0)
        center = coords.mean(axis=0, dtype=int)  # (y, x)
        particle_data['center'] = center
        # edge
        edge_cond = (0 in coords) | (image.shape[0]-1 in coords[:, 0]) | (image.shape[1]-1 in coords[:, 1])
        particle_data['edge'] = edge_cond
        # contours
        contours, _ = cv2.findContours(inst_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        particle_data['contours'] = contours
        output_image = cv2.drawContours(output_image, contours, -1, (60, 205, 24), 1)
        # aspect ratio
        ar = aspect_ratio(contours[0])
        particle_data['aspect_ratio'] = ar
        # shape estimate
        shape = shape_detector.detect(inst_mask)
        particle_data['shape_estimate'] = shape[0]
        # diameter
        if shape[0] == 'circle':
            diameter = 2*np.sqrt(area/np.pi)
            particle_data['diameter'] = diameter
        # particle instance uncertainty
        if bayesian:
            inst_uncertainty = np.mean(uncertainty[inst_mask])
            particle_data['uncertainty'] = inst_uncertainty

        particles.append(particle_data)

    # results DataFrame
    results_df = pd.DataFrame(particles)
    
    # particle size hist
    valid_particles_df = results_df[results_df['edge'] == False]
    particle_preds_filtered = edge_filter(particle_preds)
    N = len(np.unique(particle_preds_filtered)) - 1
    if N > min_particles:
        particle_sizes =  np.array(valid_particles_df['area'])

        counts, bins = np.histogram(particle_sizes, bins='rice')
        plt.bar(bins[:-1] + np.diff(bins) / 2, counts, np.diff(bins), color='k', edgecolor='k', alpha=0.6)
        plt.ylabel('Count')
        if units is not None:
            xlabel = 'Particle Size: ({}^2)'.format(units)
        else:
            xlabel = 'Particle Size: (Pixels^2)'
        plt.xlabel(xlabel)
        plt.savefig(os.path.join(target_dir, 'sizehist.png'), bbox_inches='tight', pad_inches=0.1)
        plt.close()

    
    # create and save outputs
    cv2.imwrite(os.path.join(target_dir, 'image.png'), output_image)

    seg_cmap = matplotlib.cm.tab20
    seg_cmap.set_bad(color='k')
    particle_preds[particle_preds == 0.0] = np.nan
    plt.imsave(os.path.join(target_dir, 'seg.png'), particle_preds, cmap=seg_cmap)
    particle_preds_filtered[particle_preds_filtered == 0.0] = np.nan
    plt.imsave(os.path.join(target_dir, 'seg_edgefiltered.png'), particle_preds_filtered, cmap=seg_cmap)
    if bayesian:
        plt.imsave(os.path.join(target_dir, 'uncertainty.png'), uncertainty, cmap='viridis')

    results_df.to_csv(os.path.join(target_dir, 'data.csv'), index=False)


#### tests ####

import os
import cv2
import random
import matplotlib.pyplot as plt

# test cde retreive

# doc_path = '../test/test_docs/b.html'
# cde_retrieve_images(doc_path)


# test extract_image

# base_path = '/home/by256/Documents/Projects/particle-seg-dataset/elsevier/processed-images/'
# im_paths = os.listdir(base_path)[10:]
# out_dir = '/home/by256/Documents/Projects/imagedataextractor/test/test_out'
# # random.shuffle(im_paths)
# im_paths = im_paths[10:]

# # sb_detector = ScalebarDetector()
# for i, im_path in enumerate(im_paths):
#     print(im_path)
#     # try:
#     sb_detector = ScalebarDetector()
#     # image = cv2.imread(base_path + im_path)
#     extract_image(base_path + im_path, out_dir=out_dir, bayesian=True)
#     # except Exception as e:
#     #     print(e)

#     # break

print(imghdr.what('/home/by256/Documents/Projects/imagedataextractor/test/test_docs/a.html'))