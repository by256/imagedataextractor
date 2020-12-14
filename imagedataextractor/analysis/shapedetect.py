import os
import cv2
import numpy as np


class ShapeDetector:

    def __init__(self):
        self.shape_names = ['circle', 
                            'diamond', 
                            'ellipse', 
                            'hexagon', 
                            'rectangle', 
                            'rod', 
                            'square', 
                            'triangle']
        self.shapes_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shapes/')
        self.shapes_dict = None  # key - name, value - image
        self.populate_shape_dict()

    def match_shapes(self, mask):

        mask_contour = self.get_contours(mask)[0] # index 0, since there is only 1 cntr

        distances = []
        for shape_name in self.shape_names:
            shape_image = self.shapes_dict[shape_name]
            shape_contour = self.get_contours(shape_image)[0]
            d = cv2.matchShapes(mask_contour, shape_contour, method=1, parameter=0.0)
            distances.append(d)
            
        closest_match = self.shape_names[np.argmin(distances)]
        return closest_match, distances

    def get_contours(self, x):
        contours = cv2.findContours(x.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 2:
            contours = contours[0]
        elif len(contours) == 3:
            contours = contours[1]
        return contours

    def populate_shape_dict(self):
        shape_images = []
        for shape_name in self.shape_names:
            shape_image = cv2.imread(os.path.join(self.shapes_dir, '{}.png'.format(shape_name)))
            shape_image = cv2.cvtColor(shape_image, cv2.COLOR_BGR2GRAY)
            shape_images.append(shape_image)
        self.shapes_dict = dict(zip(self.shape_names, shape_images))
