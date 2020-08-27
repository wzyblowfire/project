# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 19:51:17 2020

@author: eric
"""

import h5py
import numpy as np
import os
from  PIL import Image
x = np.arange(100)

path = './LSC_2017/test/result/'


with h5py.File(path+'test_results.h5', 'w') as f:
    for i in range(1,6):
        folder = 'A' + str(i)
        A = f.create_group(folder)
        
        p = os.path.join(path, folder)
        files_name = os.listdir(p)
        
        for file in files_name:
            if 'label' in file:
                name = file.replace('_label.png','')
                
                imgdataset = A.create_group(name)
                
                img_path = os.path.join(p, file)
                img = np.array(Image.open(img_path))
                
                imgdataset.create_dataset('label', data=img)
                imgdataset.create_dataset('label_filename', data=file)
                
        
