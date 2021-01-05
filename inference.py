#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: Nuvilabs - Luca Medeiros, luca.medeiros@nuvi-labs.com
"""

from utils.mmdet.apis import init_detector, inference_detector
import numpy as np
import cv2


class Nuvi_RecycleNet():
    def __init__(self,
                 config_file,
                 checkpoint_file,
                 threshold,
                 device='cuda:0',
                 tta=True):

        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.threshold = threshold  # Only boxes with the score larger than this will be detected
        self.tta = tta # Perform TTA on not detected images

        # Build the model from a config file and a checkpoint file
        self.model = init_detector(self.config_file, self.checkpoint_file, device=device)

        print('Model loaded sucessfully. Ready to perform inference.')
        print('Threshold: ', self.threshold)
        self.classes = {0: 'combustion',
                        1: 'paper',
                        2: 'steel',
                        3: 'glass',
                        4: 'plastic',
                        5: 'plasticbag',
                        6: 'styrofoam',
                        7: 'food'}

    def augmentIMG(self, img_array, augment_type):
        if augment_type == 'LR':
            print('augmented', augment_type)
            img = np.fliplr(img_array)

        elif augment_type == 'UDR':
            print('augmented', augment_type)
            img = np.flipud(img_array)

        elif augment_type == 'RT':
            print('augmented', augment_type)
            img = np.rot90(img_array)

        return img

    def predict(self, img_path):
        imageArray = cv2.imread(img_path)
        # Run inference using a model on a single picture -> img can be either path or array
        result = inference_detector(self.model, imageArray)
        res_idxs = [[i, k[0]] for i, k in enumerate(result[0]) if k.size != 0 and (k[:,4] > self.threshold).any()]
        if self.tta and not res_idxs:
            # Determine which augmentations to do when there is no detection. In order
            aug_type = ['LR', 'UDR', 'RT', 'Break']
            aug_idx = 0
            res_idxs = []
            while not res_idxs:
                augment_type = aug_type[aug_idx]
                if augment_type == 'Break':
                    result = [np.asarray([]) for i in range(8)]
                    break
                print('TTA type: ', augment_type)
                img_transformed = self.augmentIMG(imageArray, augment_type)
                result = inference_detector(self.model, img_transformed)
                res_idxs = [[i, k[0]] for i, k in enumerate(result[0]) if k.size != 0 and (k[:,4] > self.threshold).any()]

        json_result = self.make_json(res_idxs)

        return json_result

    def make_json(self, results):
        json_dict = {'Annotations': []}
        for result in results:
            label_idx = result[0]
            bbox = result[1][:4].tolist()
            score = result[1][-1]
            label_name = self.classes[label_idx]

            dict_result = {'Label': label_name, 'Bbox': bbox, 'Confidence': score}
            json_dict['Annotations'].append(dict_result)

        return json_dict
