#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 19:05:08 2020

@author: Nuvilabs-Luca
"""

from utils.mmdet.apis import init_detector, inference_detector, show_result_pyplot
import os
import numpy as np
import cv2

class Nuvi_RecycleNet():
    def __init__(self,
                 config_file='./model/model_config.py',
                 checkpoint_file='./model/model_checkpoint.pth',
                 output_dir='./output/',
                 threshold=0.5,
                 device='cuda:0',
                 tta=True):

        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.output_dir = output_dir
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
        res_idxs = [i for i, k in enumerate(result) if k.size != 0 and (k[:,4] > self.threshold).any()]
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
                res_idxs = [i for i, k in enumerate(result) if k.size != 0 and (k[:,4] > self.threshold).any()]

        json_result = self.make_json(res_idxs)

        return json_result

    def make_json(self, results):
        pass


recycler = Nuvi_RecycleNet()
json_result = recycler.predict('')
