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
        print('matches', dict(zip(self.shape_names, distances)))
        print('closest_match', closest_match, '\n')
        return closest_match, distances

    # def match_shapes(self, mask):

    #     # mask_contour = self.get_contours(mask)[0] # index 0, since there is only 1 cntr
    #     mask_hu_moments = self.compute_log_hu_moments(mask)

    #     distances = []
    #     for shape_name in self.shape_names:
    #         shape_image = self.shapes_dict[shape_name]
    #         # shape_contour = self.get_contours(shape_image)[0]
    #         shape_hu_moments = self.compute_log_hu_moments(shape_image)
    #         d = np.linalg.norm(mask_hu_moments-shape_hu_moments)
    #         distances.append(d)
        
    #     closest_match = self.shape_names[np.argmin(distances)]
    #     print('matches', dict(zip(self.shape_names, distances)))
    #     print('closest_match', closest_match, '\n')
    #     return closest_match, distances
    

    # def compute_log_hu_moments(self, x, eps=1e-12):
    #     moments = cv2.moments(x)
    #     hu_moments = cv2.HuMoments(moments) + 10
    #     for i in range(len(hu_moments)):
    #         hu_moments[i] =  -1.0 * np.copysign(1.0, hu_moments[i]) * np.log10(np.abs(hu_moments[i]))

    #     return hu_moments

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



# class ShapeDetector:

#     def __init__(self):
#         pass

#     def detect_circle(self, mask):

#         circle = False

#         contours = self.get_contours(mask)
#         if len(contours) >= 5:
#             ellipse = cv2.fitEllipse(contours)
#             w, h = ellipse[1]
#             aspect_ratio = w/h

#             ellipse_image = np.zeros_like(mask.copy(), dtype=np.uint8)
#             ellipse_image = cv2.ellipse(ellipse_image, ellipse, color=(255,255,255), thickness=-1)
#             ellipse_image = (ellipse_image > 0).astype(np.uint8)

#             intersection = np.sum(np.logical_and(mask, ellipse_image))
#             union = np.sum(np.logical_or(mask, ellipse_image))
#             iou = intersection / union

#             if aspect_ratio > 0.85 and iou > 0.95:
#                 circle = True
        
#         return circle

    # def get_contours(self, x):
    #     contours = cv2.findContours(x.copy(), cv2.RETR_EXTERNAL,
    #                                 cv2.CHAIN_APPROX_SIMPLE)
    #     if len(contours) == 2:
    #         contours = contours[0][0]
    #     elif len(contours) == 3:
    #         contours = contours[1][0]
    #     return contours

    # def logscale_moments(self, moments):
    #     return -1*np.copysign(1.0, moments)*np.log10(np.abs(moments))

    # def detect(self, mask):
    #     shapes = list(self.shape_dict.keys())
    #     shape_moments = list()

    #     distances = []

    #     for shape in shapes:
    #         shape_image = self.shape_dict[shape] / 255
    #         d = cv2.matchShapes(mask, shape_image, cv2.CONTOURS_MATCH_I1, 0)
    #         distances.append(d)

    # def create_shape_images(self):
    #     import matplotlib.pyplot as plt
    #     # square
    #     square_image = np.zeros(shape=(self.h, self.w), dtype=np.uint8)
    #     cv2.rectangle(square_image, (self.w//4, self.h//4), (self.w-(self.w//4), self.h-(self.h//4)), (255, 255, 255), thickness=-1)
    #     self.shape_dict['square'] = square_image

    #     # rectangle
    #     rect_image = np.zeros(shape=(self.h, self.w), dtype=np.uint8)
    #     cv2.rectangle(rect_image, (self.w//3, self.h//4), (self.w-(self.w//3), self.h-(self.h//4)), (255, 255, 255), thickness=-1)
    #     self.shape_dict['rectangle'] = rect_image

    #     # rod
    #     rod_image = np.zeros(shape=(self.h, self.w), dtype=np.uint8)
    #     cv2.ellipse(rod_image, (self.w//2, self.h//2), (self.w//3, self.h//10), 0, 0, 360, color=(255,255,255), thickness=-1, )
    #     self.shape_dict['rod'] = rod_image

    #     # triangle
    #     tri_image = np.zeros(shape=(self.h, self.w), dtype=np.uint8)
    #     points = np.array([[self.w//2, self.h//6],
    #                        [self.w//6, self.h-(self.h//6)], 
    #                        [self.w-(self.w//6), self.h-(self.h//6)]]).astype(int)
    #     cv2.drawContours(tri_image, [points], -1, (255,255,255), thickness=-1)
    #     self.shape_dict['triangle'] = tri_image

    #     # circle
    #     circle_image = np.zeros(shape=(self.h, self.w), dtype=np.uint8)
    #     cv2.circle(circle_image, (self.w//2, self.h//2), self.h//4, color=(255,255,255), thickness=-1, )
    #     self.shape_dict['circle'] = circle_image
    #     # pentagon
    #     pent_image = np.zeros(shape=(self.h, self.w), dtype=np.uint8)
    #     pent_points = np.array([[self.w/2, self.h/4.5], 
    #                        [self.w/3, self.h/3 + self.h/15], 
    #                        [(self.w/3)+(self.w/12), 2*self.h/3], 
    #                        [self.w-(self.w/3)-(self.w/12), 2*self.h/3], 
    #                        [self.w-(self.w/3), self.h/3 + self.h/15]]).astype(int)

    #     cv2.drawContours(pent_image, [pent_points], -1, (255,255,255), thickness=-1)
    #     self.shape_dict['pentagon'] = pent_image

    #     # hexagon
    #     hex_image = np.zeros(shape=(self.h, self.w), dtype=np.uint8)
    #     hex_points = np.array([[self.w/3, self.h/4], 
    #                            [self.w/5, self.h/2], 
    #                            [self.w/3, self.h-(self.h/4)], 
    #                            [self.w-(self.w/3), self.h-(self.h/4)], 
    #                            [self.w-(self.w/5), self.h/2], 
    #                            [self.w-(self.w/3), self.h/4], ]).astype(int)
    #     cv2.drawContours(hex_image, [hex_points], -1, (255,255,255), thickness=-1)
    #     self.shape_dict['hexagon'] = hex_image
