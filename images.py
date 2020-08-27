# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 12:37:37 2020

@author: eric
"""

import os
import cv2
import numpy as np

def get_files_name(task_name = 'A1', label_name = 'A1', fil = 'rgb'):
    # task_name = A1 || A2 || A3
    
    path = "./LSC_2017/"
    
    image_path = path + task_name
    lable_path = path + label_name
    files_name = os.listdir(image_path)
    
    images_name, labels_name = [], []
    for file in files_name:
        if fil in file:
            images_name.append(image_path+'/'+file)
            label = file.replace(fil, 'label')
            labels_name.append(lable_path+'/'+label)
    
    return images_name, labels_name


def get_images(images_name, labels_name):
    
    images, labels = [], []
    
    for image_name, label_name in zip(images_name, labels_name):
        images.append(cv2.imread(image_name))
        label = cv2.imread(label_name)
        labels.append(label)
        #mask = transform(label)
        #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        #cv2.imwrite(label_name.replace('.png', '')+'_mask.png', mask)
        
    return images, labels


def transform(label):
    index_lib = {}
    index = 1
    im = label.copy()
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            tmp = tuple(im[i][j])
            if tmp == (0,0,0): continue
            
            if tmp not in index_lib:
                index_lib[tmp] = index
                index += 1
            im[i][j] = index_lib[tmp]
    return im
        
from PIL import Image

if __name__ == '__main__':
    images_name, labels_name = get_files_name('A4')
    images, labels = get_images(images_name, labels_name)
    
    
    #mask = Image.open('LSC_data/A1/plant005_label_mask.png')
    #print(np.array(mask).shape)
    
    #mask = Image.open('PennFudanPed/PedMasks/FudanPed00001_mask.png')
    #print(np.array(mask).shape)
    

