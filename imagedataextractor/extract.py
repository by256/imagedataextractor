import os
import cv2
import copy
import imghdr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdfpy import rdf2d
from chemdataextractor import Document

from .figsplit import figsplit
from .analysis import ShapeDetector
from .analysis.filtering import edge_filter
from .analysis.particlesize import aspect_ratio
from .scalebar import ScalebarDetector
from .segment import ParticleSegmenter
from .utils import get_contours, shuffle_segmap


def extract(input_path, out_dir,  seg_kws={'bayesian':True, 'n_samples':30, 'tu':0.0125, 'device':'cpu'}):
    """Extract from single image, single doc, directory of images, or directory of docs."""
    
    allowed_doc_exts = ['.html', '.xml', '.json', '.pdf']

    # single image
    if os.path.isfile(input_path):
        if imghdr.what(input_path) is not None:
            file_ext = os.path.splitext(input_path)[-1]
            fn = input_path.split('/')[-1].split(file_ext)[0]
            target_dir = os.path.join(out_dir, fn)
            image = cv2.imread(input_path)
            _figsplit_mkdir_and_extract(image, target_dir, seg_kws=seg_kws)
    # single document
    elif os.path.splitext(input_path)[1] in allowed_doc_exts:
        pass
    # directory of images or documents
    elif os.path.isdir(input_path):
        for f in os.listdir(input_path):
            file_path = os.path.join(input_path, f)
            file_ext = os.path.splitext(file_path)[-1]
            # image
            if imghdr.what(file_path) is not None:
                fn = f.split('/')[-1].split(file_ext)[0]
                target_dir = os.path.join(out_dir, fn)
                image = cv2.imread(file_path)
                _figsplit_mkdir_and_extract(image, target_dir, seg_kws=seg_kws)
            # document
            if file_ext in allowed_doc_exts:
                extract_document()

def _figsplit_mkdir_and_extract(image, target_dir, seg_kws):
    """Private function that combines figsplit, creation of output dir, and extract split images."""    
    images = figsplit(image)
    if len(images) == 1:
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        _extract_image(image, target_dir, seg_kws=seg_kws)
    elif len(images) > 1:
        for i, image in enumerate(images):
            split_target_dir = os.path.join(target_dir, str(i+1))
            if not os.path.exists(split_target_dir):
                os.makedirs(split_target_dir)
            _extract_image(image, split_target_dir, seg_kws=seg_kws)


def _extract_image(image, target_dir, min_particles=10, 
                  seg_kws={'bayesian':True, 'n_samples':30, 'tu':0.0125, 'device':'cpu'}):
    """
    Extract from a single image (not a panel).
    """

    # initialise detectors
    sb_detector = ScalebarDetector()
    segmenter = ParticleSegmenter(**seg_kws)
    shape_detector = ShapeDetector()

    output_image = image.copy()
    scalebar_image = image.copy()

    # detect scalebar
    text, units, conversion, scalebar_contour = sb_detector.detect(image)
    # draw scalebar on output image
    if scalebar_contour is not None:
        x, y, w, h = cv2.boundingRect(scalebar_contour)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (24, 24, 205), 1)
        cv2.putText(output_image, text, (image.shape[1]//2, image.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(24, 24, 205))
        cv2.rectangle(scalebar_image, (x, y), (x + w, y + h), (24, 24, 205), 1)
        cv2.putText(scalebar_image, text, (image.shape[1]//2, image.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(24, 24, 205))
    else:
        cv2.putText(scalebar_image, 'Scalebar not found.', (image.shape[1]//4, 3*image.shape[0]//4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=(24, 24, 205))

    # segment particles
    particle_preds, uncertainty, original = segmenter.segment(image)
    particle_preds = shuffle_segmap(particle_preds)  # for vis purposes
    particle_preds = particle_preds.astype(float)
    original = original.astype(float)

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
                        'area (pixels^2)': None,
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
        particle_data['area (pixels^2)'] = area
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
        # contours = get_contours(inst_mask)
        particle_data['contours'] = contours
        output_image = cv2.drawContours(output_image, contours, -1, (60, 205, 24), 1)
        # aspect ratio
        ar = aspect_ratio(contours[0])
        particle_data['aspect_ratio'] = ar
        # shape estimate
        shape_estimate, shape_distances = shape_detector.match_shapes(inst_mask)
        particle_data['shape_estimate'] = shape_estimate
        if shape_estimate == 'circle':
            # diameter
            diameter = 2*np.sqrt(area/np.pi)
            particle_data['diameter'] = diameter
        # particle instance uncertainty
        if seg_kws['bayesian']:
            inst_uncertainty = np.mean(uncertainty[inst_mask])
            particle_data['uncertainty'] = inst_uncertainty

        particles.append(particle_data)

    # results DataFrame
    results_df = pd.DataFrame(particles)
    
    if len(results_df) > 0:
        # particle size hist
        valid_particles_df = results_df[results_df['edge'] == False]
        particle_preds_filtered = edge_filter(particle_preds)
        N = len(np.unique(particle_preds_filtered)) - 1
        if N > min_particles:
            particle_sizes =  np.array(valid_particles_df['area'])

            counts, bins = np.histogram(particle_sizes, bins='rice')
            hist_fig = plt.figure()
            plt.bar(bins[:-1] + np.diff(bins) / 2, counts, np.diff(bins), color='k', edgecolor='k', alpha=0.6)
            plt.ylabel('Count')
            if units is not None:
                xlabel = 'Particle Size: ({}^2)'.format(units)
            else:
                xlabel = 'Particle Size: (Pixels^2)'
            plt.xlabel(xlabel)
            plt.savefig(os.path.join(target_dir, 'sizehist.png'), bbox_inches='tight', pad_inches=0.1)
            plt.close(hist_fig)

        # rdf
        if N > min_particles:
            center_coords = np.array(list(results_df['center']))
            dr = np.sqrt(np.mean(results_df['area (pixels^2)'])) / 4
            g_r, radii = rdf2d(center_coords, dr=dr)
            
            rdf_fig = plt.figure(figsize=(10, 6))
            plt.plot(radii, g_r, color='k')

            plt.ylabel('g(r)')
            plt.xlabel('r (nm)')
            plt.savefig(os.path.join(target_dir, 'rdf.png'), bbox_inches='tight', pad_inches=0.1)
            plt.close(rdf_fig)
            np.savetxt(os.path.join(target_dir, 'rdf.txt'), np.stack([radii, g_r], axis=-1))
        
        # create and save outputs
        cv2.imwrite(os.path.join(target_dir, 'det.png'), output_image)
        cv2.imwrite(os.path.join(target_dir, 'scalebar.png'), scalebar_image)

        seg_cmap = copy.copy(matplotlib.cm.tab20)
        seg_cmap.set_bad(color='k')
        original[original == 0.0] = np.nan
        plt.imsave(os.path.join(target_dir, 'pre.png'), original, cmap=seg_cmap)
        particle_preds[particle_preds == 0.0] = np.nan
        plt.imsave(os.path.join(target_dir, 'seg.png'), particle_preds, cmap=seg_cmap)
        particle_preds_filtered[particle_preds_filtered == 0.0] = np.nan
        plt.imsave(os.path.join(target_dir, 'seg_edgefiltered.png'), particle_preds_filtered, cmap=seg_cmap)
        if seg_kws['bayesian']:
            plt.imsave(os.path.join(target_dir, 'uncertainty.png'), uncertainty, cmap='viridis')

        results_df.to_csv(os.path.join(target_dir, 'data.csv'), index=False)
    else:
        with open('{}/{}.txt'.format(target_dir, 'result'), 'w+') as f:
            f.write('No particles found.')

def extract_document():
    """Extract from single document."""
    raise NotImplementedError('Extraction from documents will be implemented upon the release of CDE 2.0. Please use image extraction instead.')

#### tests ####

# import os
# import cv2
# import random
# import matplotlib.pyplot as plt

# test cde retreive

# doc_path = '../test/test_docs/b.html'
# cde_retrieve_images(doc_path)


# test extract_image

# base_path = '/home/by256/Documents/Projects/particle-seg-dataset/elsevier/processed-images/'
# im_paths = os.listdir(base_path)[10:]
# out_dir = '/home/by256/Documents/Projects/imagedataextractor/test/test_out3'
# random.shuffle(im_paths)
# im_paths = im_paths[3:]

# seg_kws = {'bayesian':True, 'n_samples':30, 'tu':0.0125, 'device':'cpu'}

# # sb_detector = ScalebarDetector()
# for i, im_path in enumerate(im_paths):
#     print(im_path)
#     # try:
#     sb_detector = ScalebarDetector()
    # image = cv2.imread(base_path + im_path)
    # extract_image(image, out_dir, seg_kws=seg_kws)
#     # except Exception as e:
#     #     print(e)

#     # break

# test extract (dir of images)

# base_path = '/home/by256/Documents/Projects/particle-seg-dataset/elsevier/processed-images/'
# im_paths = os.listdir(base_path)[:3]
# im_paths = [os.path.join(base_path, x) for x in im_paths]
# random.shuffle(im_paths)

# out_dir = '/home/by256/Documents/Projects/imagedataextractor/test/test_out4/'
# extract(base_path, out_dir)

# test extract (single images)
# im_path = '/home/by256/Documents/Projects/ideweb/ideweb/static/img/0_C6CE01551D_fig1_2.png'
# im_path = '/media/by256/128GBMaster/EM-images/images-c/10.1016.j.jpowsour.2016.10.028.gr1.png' # error!
# im_path = '/media/by256/128GBMaster/EM-images/images-c/10.1016.j.jssc.2017.09.004.gr8.png'
# out_dir = '/home/by256/Documents/Projects/imagedataextractor/test/test_out3/'
# extract(im_path, out_dir, seg_kws=seg_kws)