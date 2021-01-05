#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 19:05:08 2020

@author: Nuvilabs-Luca
"""

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import glob
import os
import numpy as np
import csv
import cv2
import time

def makeBoxWider(imageArray, xMin, yMin, xMax, yMax, margin):
    imageHeight, imageWidth = imageArray.shape[:2]
    margin = int(margin)
    xMinWider = max(xMin - margin, 0)
    yMinWider = max(yMin - margin, 0)
    xMaxWider = min(xMax + margin, imageWidth)
    yMaxWider = min(yMax + margin, imageHeight)
    return xMinWider, yMinWider, xMaxWider, yMaxWider
    
def loadImage(imageFileName):
    imagePath = os.path.join(img_dir, imageFileName)
    imageArray = cv2.imread(imagePath)
    return imageArray
    
def saveImage(filename_og, labelName, imageArray):
    saveFolder = os.path.join(output_dir, labelName)
    if not os.path.isdir(saveFolder):
        os.makedirs(saveFolder)
    numFile = len(os.listdir(saveFolder))
    imageId = str(numFile)
    if len(imageId) == 1:
        imageId = '000' + imageId
    elif len(imageId) == 2:
        imageId = '00' + imageId
    elif len(imageId) == 3:
        imageId = '0' + imageId
    fileName = labelName + '_' + imageId + '.png'
    saveFilePath = os.path.join(saveFolder, fileName)
    cv2.imwrite(saveFilePath, imageArray)
    cv2.imwrite('/home/ubuntu/luca/mmdetection/output_data_bbox/cropped/'+ filename_og+'.png', imageArray)

def cropImage(imageArray, xMin, yMin, xMax, yMax):
    imageArrayCropped = imageArray[yMin:yMax, xMin:xMax]
    return imageArrayCropped

def padImage(imageArray, resultWidth, resultHeight):
    h, w = imageArray.shape[:2]
    case = checkCase(imageArray, resultWidth, resultHeight)
    if case == 0:
        imagePadded = putImageOnBlack(imageArray, resultWidth, resultHeight)
    elif case == 1:
        resizeRatio = getResizeRatio(imageArray, resultWidth, resultHeight)
        resizeShape = (round(w * resizeRatio), round(h * resizeRatio))
        imageResized = cv2.resize(imageArray, resizeShape)
        imagePadded = putImageOnBlack(imageResized, resultWidth, resultHeight)
    return imagePadded

def checkCase(imageArray, resultWidth, resultHeight):
    inputHeight, inputWidth = imageArray.shape[:2]
    if (inputHeight < resultHeight) and (inputWidth < resultWidth):
        return 0
    else:
        return 1
    
def putImageOnBlack(imageArray, resultWidth, resultHeight):
    h, w = imageArray.shape[:2]
    blackImage = np.zeros((resultHeight, resultWidth, 3))
    xMin = int((resultWidth - w)/2)
    yMin = int((resultHeight - h)/2)
    blackImage[yMin:yMin+h, xMin:xMin+w, :] = imageArray
    return blackImage

def getResizeRatio(imageArray, resultWidth, resultHeight):
    inputHeight, inputWidth = imageArray.shape[:2]
    widthRatio = resultWidth / inputWidth
    heightRatio = resultHeight / inputHeight
    return min(widthRatio, heightRatio)

def cropPolygon(filename_og, imageArray, shapes, result, padding=False, resultWidth=320, resultHeight=320, threshold=0.5):
    for idx, shape in enumerate(shapes):
        for bbox in result[shape]:
            filename_og = filename_og + '_' + str(idx+1)
            labelName = classes[shape]
            xMin, yMin, xMax, yMax, score = bbox
            if score >= threshold:
                xMin, yMin, xMax, yMax = int(xMin), int(yMin), int(xMax), int(yMax)
                xMinWider, yMinWider, xMaxWider, yMaxWider = makeBoxWider(imageArray, xMin, yMin, xMax, yMax, 25)
                imageCropped = cropImage(imageArray, xMinWider, yMinWider, xMaxWider, yMaxWider)
                if not padding:
                    saveImage(filename_og, labelName, imageCropped)
                else:
                    imagePadded = padImage(imageCropped, resultWidth, resultHeight)
                    saveImage(filename_og, labelName, imagePadded)
                
def augmentIMG(img_array, augment_type):
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

config_file = './work_dirs/dRS_rs50_FinalDataNEW_1280/newData_detRS.py'
checkpoint_file = 'work_dirs/dRS_rs50_FinalDataNEW_1280/newData_RS_e24.pth'
img_dir = '/home/ubuntu/luca/darknet/trash_Final_Dataset2/valid/'
output_dir ='./output/dRS_rs50_FinalDataNEW_1280_24_valid/'

write_csv = True
save_bbox = True
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
print('Model loaded sucessfully. Starting inference on folder: ', img_dir)
classes = {0: 'combustion', 1: 'paper', 2: 'steel', 3: 'glass', 4: 'plastic', 5: 'plasticbag', 6: 'styrofoam', 7: 'food'}


img_ext = ('png', 'jpg', 'JPG', 'jpeg')
image_list = []
for ext in img_ext:
        image_list.extend(glob.glob(img_dir + '*.' + ext))
print(len(image_list),' files to eval.')
results_output = []
c = 1
avg_score = {0:[], 1:[], 2:[], 3:[], 4:[],  5:[], 6:[], 7:[]}
for file in image_list:
    start_time = time.time()
    filename = os.path.basename(file)
    filename_og = os.path.splitext(filename)[0]
    img = loadImage(file)
    # result = inference_detector(model, img)
    # print(result)
    # *** Parameters ***
    detectoRS_Threshold = 0.5  # Only boxes with the score larger than this will be cropped
    horizontal_Rotation = False # Perform augmentation techniques if detector doesnt output any results
    correction_Measures = True # If image is in the format of 720x1280 rotate it to 1280x720. Better results with DetectoRS
    
    print('thr',detectoRS_Threshold)
    if horizontal_Rotation:
        i_H, i_W = img.shape[:2]
        if i_H > i_W:
            img = np.rot90(img)
    
    if correction_Measures:
        
        # Determine which augmentations to do when there is no detection. In order
        aug_type = ['None', 'LR', 'UDR', 'RT', 'Break']
        aug_idx = 0
        res_idxs = []
        while not res_idxs:
            augment_type = aug_type[aug_idx]

            if augment_type == 'Break':
                result = [np.asarray([]) for i in range(8)]
                break

            elif augment_type is not 'None':
                print(augment_type)
                img = augmentIMG(img, augment_type)

            # Run inference using a model on a single picture -> img can be either path or array
            result = inference_detector(model, img)
            # print(len(result))
            res_idxs = [i for i, k in enumerate(result) if k.size != 0 and (k[:,4] > detectoRS_Threshold).any()]

            aug_idx += 1
        
            model.show_result(img, result, out_file=output_dir+filename, score_thr=detectoRS_Threshold)
        # print(result)
        for r_i in res_idxs:

            avg_score[r_i].append(result[r_i][0][4]) 
    else: 
        result = inference_detector(model, img)
        model.show_result(img, result, out_file=output_dir+filename, score_thr=detectoRS_Threshold)
    # print(result)
    res = [i for i, k in enumerate(result) if k.size != 0 and (k[:,4] > detectoRS_Threshold).any()]
    # model.show_result(img, result, out_file=output_dir+augment_type+'_'+filename)
    cropPolygon(filename_og, img, res, result, padding=True, threshold =detectoRS_Threshold)
    print('Saved ({}/{}): {}   {}'.format(c, len(image_list), filename, time.time()-start_time))
    # h, w = img.shape[:2]
    # if h>w:
    #     img = np.rot90(img)
    # result = inference_detector(model, img)
    # # show the results
    # model.show_result(img, result, out_file=output_dir+filename)
    # print('Saved ({}/{}): {}'.format(c, len(image_list), filename))

    # res = [i for i, k in enumerate(result) if k.size != 0]
    if write_csv:
        for class_num in res:                              
            results_output.append({'filename': filename, 'res':class_num+1})
    # if save_bbox:
    #     img_cv = loadImage(file)
    #     cropPolygon(filename_og, img_cv, res, result, padding=True)
    #     # for i, class_num in enumerate(res):
    #     #     x, y, w, h, score = result[class_num][0]
    #     #     if score >= 0.5:
    #     #         cropped_box = img_cv[int(y):int(h) , int(x):int(w), :]
    #     #         cv2.imwrite(output_dir+'bbox_'+classes[class_num]+'_'+filename, cropped_box)
    c+=1
    
for key, val in avg_score.items():
    print(classes[key] + ' : ', np.mean(val))
        
if write_csv:
    with open(output_dir+'ans.csv', 'w') as f:
        w = csv.DictWriter(f, fieldnames=['filename', 'res'])
        w.writeheader()
        for data in results_output:
            w.writerow(data)


