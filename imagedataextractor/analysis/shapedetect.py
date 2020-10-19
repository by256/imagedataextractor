import cv2
import numpy as np


class ShapeDetector:

    def __init__(self):
        self.h, self.w = 512, 512
        self.shape_dict = dict()
        self.create_shape_images()

    def detect(self, mask):
        shapes = list(self.shape_dict.keys())
        shape_moments = list()

        distances = []

        for shape in shapes:
            shape_image = self.shape_dict[shape] / 255
            d = cv2.matchShapes(mask, shape_image, cv2.CONTOURS_MATCH_I1, 0)
            print(shape, d)
            distances.append(d)

        min_idx = np.argmin(distances)
        return shapes[min_idx], distances[min_idx]

    def get_contours(self, x):
        contours = cv2.findContours(x.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 2:
            contours = contours[0][0]
        elif len(contours) == 3:
            contours = contours[1][0]
        return contours

    def logscale_moments(self, moments):
        return -1*np.copysign(1.0, moments)*np.log10(np.abs(moments))


    def create_shape_images(self):
        import matplotlib.pyplot as plt
        # square
        square_image = np.zeros(shape=(self.h, self.w), dtype=np.uint8)
        cv2.rectangle(square_image, (self.w//4, self.h//4), (self.w-(self.w//4), self.h-(self.h//4)), (255, 255, 255), thickness=-1)
        self.shape_dict['square'] = square_image

        # rectangle
        rect_image = np.zeros(shape=(self.h, self.w), dtype=np.uint8)
        cv2.rectangle(rect_image, (self.w//3, self.h//4), (self.w-(self.w//3), self.h-(self.h//4)), (255, 255, 255), thickness=-1)
        self.shape_dict['rectangle'] = rect_image

        # rod
        rod_image = np.zeros(shape=(self.h, self.w), dtype=np.uint8)
        cv2.ellipse(rod_image, (self.w//2, self.h//2), (self.w//3, self.h//10), 0, 0, 360, color=(255,255,255), thickness=-1, )
        self.shape_dict['rod'] = rod_image

        # triangle
        tri_image = np.zeros(shape=(self.h, self.w), dtype=np.uint8)
        points = np.array([[self.w//2, self.h//6],
                           [self.w//6, self.h-(self.h//6)], 
                           [self.w-(self.w//6), self.h-(self.h//6)]]).astype(int)
        cv2.drawContours(tri_image, [points], -1, (255,255,255), thickness=-1)
        self.shape_dict['triangle'] = tri_image

        # circle
        circle_image = np.zeros(shape=(self.h, self.w), dtype=np.uint8)
        cv2.circle(circle_image, (self.w//2, self.h//2), self.h//4, color=(255,255,255), thickness=-1, )
        self.shape_dict['circle'] = circle_image
        # pentagon
        pent_image = np.zeros(shape=(self.h, self.w), dtype=np.uint8)
        pent_points = np.array([[self.w/2, self.h/4.5], 
                           [self.w/3, self.h/3 + self.h/15], 
                           [(self.w/3)+(self.w/12), 2*self.h/3], 
                           [self.w-(self.w/3)-(self.w/12), 2*self.h/3], 
                           [self.w-(self.w/3), self.h/3 + self.h/15]]).astype(int)

        cv2.drawContours(pent_image, [pent_points], -1, (255,255,255), thickness=-1)
        self.shape_dict['pentagon'] = pent_image

        # hexagon
        hex_image = np.zeros(shape=(self.h, self.w), dtype=np.uint8)
        hex_points = np.array([[self.w/3, self.h/4], 
                               [self.w/5, self.h/2], 
                               [self.w/3, self.h-(self.h/4)], 
                               [self.w-(self.w/3), self.h-(self.h/4)], 
                               [self.w-(self.w/5), self.h/2], 
                               [self.w-(self.w/3), self.h/4], ]).astype(int)
        cv2.drawContours(hex_image, [hex_points], -1, (255,255,255), thickness=-1)
        self.shape_dict['hexagon'] = hex_image
