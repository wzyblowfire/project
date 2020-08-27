# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 10:33:31 2020

@author: eric
"""

import h5py
from PIL import Image
import cv2
import os
#打开文件
example_path = './LSC_2017/submission_example'
example = example_path + '/submission_example.h5'
example_path = './LSC_2017/CVPPP2017_testing_images'
example = example_path + '/CVPPP2017_testing_images.h5'
'''
with h5py.File(example,"r") as f:
    
    for set_key in f.keys():
        dataset = f[set_key]
        print(dataset)
        for img_name in dataset.keys():
            img = dataset[img_name]
            for label in img.keys():
                data = img[label].value
                cv2.imshow('',data)
                cv2.waitKey(0)
'''        

label_name = 'rgb_filename'        
fg_name = 'fg_filename'
with h5py.File(example,"r") as f:
    for A in f.keys():
        dataset = f[A]
        for id in dataset.keys():
            img = dataset[id]['rgb'][:]
            name = dataset[id][label_name][()]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            save_path = os.path.join(example_path,A, name)
            print(save_path)
            cv2.imwrite(save_path, img)
            
            img = dataset[id]['fg'][:]
            name = dataset[id][fg_name][()]
            save_path = os.path.join(example_path,A, name)
            print(save_path)
            cv2.imwrite(save_path, img)