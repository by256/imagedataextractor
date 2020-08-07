import os
import sys
import argparse
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

sys.path.append('../../')
from scalebar.textdetection import TextDetector


def show_image(image):
    fig, ax = plt.subplots()
    ax.imshow(image)
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument('--images-dir', 
                    type=str, 
                    default='/home/by256/Documents/Projects/particle-seg-dataset/elsevier/norm-images/',
                    help='path to input image directory')
parser.add_argument('--target-dir', 
                    type=str, 
                    default='/home/by256/Documents/Projects/imagedataextractor/imagedataextractor/scalebar/finetune/data/',
                    help='path to target directory')

args = parser.parse_args()

image_paths = os.listdir(args.images_dir)
image_paths = [args.images_dir+x for x in image_paths if x.endswith('.png')]

images = []

text_detector = TextDetector()

print('Detecting scalebars in input images...\n')
for path in tqdm(image_paths):
    image = np.array(Image.open(path))
    rois = text_detector.get_text_rois(image, augment=False)
    for roi in rois:
        roi = np.array(roi)
        images.append(roi)
        break # only append the first one

image_count = 1000 # change this back to 1

for i, image in enumerate(images):
    plot_thread = multiprocessing.Process(target=show_image, args=(image,))
    plot_thread.start()

    user_input = input('Enter scalebar text: ')
    if user_input in ['skip', 's']:
        pass
    else:
        # save image and gt
        Image.fromarray(image).save('{}{}.tif'.format(args.target_dir, image_count))
        with open('{}{}.gt.txt'.format(args.target_dir, image_count), 'w+') as f:
            f.write(user_input)
        # save inverted image and gt
        Image.fromarray(255-image).save('{}{}_inv.tif'.format(args.target_dir, image_count))
        with open('{}{}_inv.gt.txt'.format(args.target_dir, image_count), 'w+') as f:
            f.write(user_input)

    plot_thread.kill()
    # plt.close()
    image_count += 1
    print('{}/{}'.format(i+1, len(images)), end='\r', flush=True)
    
    # if i == 10:
    #     break
        

        

    