import os
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageFilter
from .utils import non_max_suppression


class TextDetector:

    """
    A lot of this code was adapted from 
    https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/.
    Credit to Adrian Rosebrock.
    """

    def __init__(self):
        self.model_path = self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models/frozen_east_text_detection.pb')
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

    def preprocess_image(self, image):
        h, w = image.shape[:2]
        new_h = int(h - (h % 32))
        new_w = int(w - (w % 32))
        image = Image.fromarray(image).resize((new_w, new_h), resample=Image.BICUBIC)
        return np.array(image)

    def detect_text(self, image):
        image = self.preprocess_image(image)
        blob_params = self.blob_params
        blob_params['size'] = (image.shape[1], image.shape[0])
        blob = cv2.dnn.blobFromImage(image, **blob_params)
        self.model.setInput(blob)
        scores, geometry = self.model.forward(self.layer_names)
        boxes = self.postprocess_detections(scores, geometry)
        return boxes

    def get_text_rois(self, image, scale=3, augment=True):
        rois = []
        roi_locs = []
        boxes = self.detect_text(image)
        for x1, y1, x2, y2 in boxes:
            box_w = int(scale * (x2 - x1))
            box_h = int(scale * (y2 - y1))
            start_x, end_x = x1 - box_w, x2 + box_w
            start_y, end_y = y1 - box_h, y2 + box_h
            if start_x < 0:
                start_x = 0
            if start_y < 0:
                start_y = 0

            roi_image = image[start_y:end_y, start_x:end_x]
            rois.append(roi_image)
            roi_locs.append((start_x, start_y, end_x, end_y))
        
        if augment:
            inverted_rois = [255-x for x in rois]
            rois = rois + inverted_rois

            gray = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            blur = lambda x: cv2.GaussianBlur(x, (5,5), 0)
            thresh = lambda x: cv2.threshold(x, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

            thresh_rois = [thresh(blur(gray(x))) for x in rois]

            rois = rois + thresh_rois  # concat augmented rois
            roi_locs = roi_locs * 4
        return rois, roi_locs