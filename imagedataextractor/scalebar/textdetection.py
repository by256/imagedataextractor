import cv2
import numpy as np
import pytesseract
from .utils import non_max_suppression


class TextDetector:

    """
    A lot of this code was adapted from 
    https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/.
    Credit to Adrian Rosebrock.
    """

    def __init__(self):
        # self.model_path = '../models/frozen_east_text_detection.pb'
        self.model_path = '/Users/batuhan/Documents/Projects/imagedataextractor/imagedataextractor/models/frozen_east_text_detection.pb'
        self.layer_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
        self.model = cv2.dnn.readNet(self.model_path)
        self.blob_params = {'scalefactor': 1.0, 
                            'size': (), 
                            'mean': (123.68, 116.78, 103.94),  # ImageNet
                            'swapRB': True}  # should maybe be false
        self.confidence_threshold = 0.5

    def postprocess_detections(self, scores, geometry):
        rows, cols = scores.shape[2:]
        rects = []
        confidences = []

        for y in range(0, rows):
            scores_data = scores[0, 0, y]
            x_data_0 = geometry[0, 0, y]
            x_data_1 = geometry[0, 1, y]
            x_data_2 = geometry[0, 2, y]
            x_data_3 = geometry[0, 3, y]
            angles_data = geometry[0, 4, y]
            for x in range(cols):
                # if our score does not have sufficient probability, ignore it
                if scores_data[x] < self.confidence_threshold:
                    continue
                # compute the offset factor as our resulting feature maps will be 4x smaller than the input image
                offset_x, offset_y = x*4.0, y*4.0
                # extract the rotation angle for the prediction and then compute the sin and cosine
                angle = angles_data[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
                # use the geometry volume to derive the width and height of the bounding box
                h = x_data_0[x] + x_data_2[x]
                w = x_data_1[x] + x_data_3[x]
                # compute both the starting and ending (x, y)-coordinates for the text prediction bounding box
                end_x = int(offset_x + (cos * x_data_1[x]) + (sin * x_data_2[x]))
                end_y = int(offset_y - (sin * x_data_1[x]) + (cos * x_data_2[x]))
                start_x = int(end_x - w)
                start_y = int(end_y - h)
                # add the bounding box coordinates and probability score to our respective lists
                rects.append((start_x, start_y, end_x, end_y))
                confidences.append(scores_data[x])
        
        rects = np.array(rects)
        confidences = np.array(confidences)
        # sort confidences in descending order
        sort_idx = np.argsort(confidences)[::-1]
        rects = rects[sort_idx]
        confidences = confidences[sort_idx]

        boxes = non_max_suppression(rects, probs=confidences, overlapThresh=0.3)
        return boxes


    def detect_text(self, image):
        blob_params = self.blob_params
        blob_params['size'] = (image.shape[1], image.shape[0])
        blob = cv2.dnn.blobFromImage(image, **blob_params)
        self.model.setInput(blob)
        scores, geometry = self.model.forward(self.layer_names)
        boxes = self.postprocess_detections(scores, geometry)
        return boxes