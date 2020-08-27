# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 20:13:49 2020

@author: eric
"""

import numpy as np
import images
import cv2
from matplotlib import pyplot as plt



def get_3d_histogram(images, labels):
    hist_fore, hist_back = np.zeros((256,256,256)), np.zeros((256,256,256))
    
    for image, label in zip(images, labels):
        image_fore, image_back = label_mask.getLeafRGB(image, label) # Cut the background.
        kernel_size = (5, 5)
        sigma = 1.5
        #image_gauss_f = cv2.GaussianBlur(image_fore, kernel_size, sigma) # Gussian blur
        #image_gauss_b = cv2.GaussianBlur(image_back, kernel_size, sigma) # Gussian blur
        image_gauss_f = image_fore
        image_gauss_b = image_back
        image_fore_lab = cv2.cvtColor(image_gauss_f, cv2.COLOR_BGR2LAB) # Color conversion(RGB->L*a*b*)
        image_back_lab = cv2.cvtColor(image_gauss_b, cv2.COLOR_BGR2LAB)
        
        image_fore_sequence = np.reshape(image_fore_lab, (530*500, 3))
        image_back_sequence = np.reshape(image_back_lab, (530*500, 3))
        
        for x in image_fore_sequence:
            hist_fore[x[0]][x[1]][x[2]] += 1
        for x in image_back_sequence:
            hist_back[x[0]][x[1]][x[2]] += 1
    return hist_fore, hist_back

def threshold_fore(images, hist_fore, hist_back):
    
    fore = []
    for image in images:
        test = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        for i in range(0,len(test)):
            for col in range(0, len(test[0])):
                tmp = test[i][col]
                a = hist_back[tmp[0]][tmp[1]][tmp[2]]
                b = hist_fore[tmp[0]][tmp[1]][tmp[2]]
                if b <= a:
                    test[i][col] = np.array([0, 128, 128])
        test = cv2.cvtColor(test, cv2.COLOR_LAB2BGR)
        fore.append(test)
    return fore
    
    
if __name__ == '__main__':

    dataset = 'A1train'
    trainset = 'A1train'
    labelset = 'A1'
    image_names, label_names = images.get_files_name(dataset, labelset, 'rgb')
    images, labels = images.get_images(image_names, label_names)
    
    
    #hist_fore, hist_back = get_3d_histogram(images, labels)
    
    hist_fore = np.load('hist_fore_'+trainset+'.npy')
    hist_back = np.load('hist_back_'+trainset+'.npy')
    

    fores = threshold_fore(images, hist_fore, hist_back)
    
    #for fore, name in zip(fores, image_names):
        #cv2.imwrite(name.replace('rgb', 'filted'), fore)
    
    '''
    test = cv2.cvtColor(test, cv2.COLOR_LAB2BGR)
    imgs = np.hstack([init, test, mask])
    cv2.imshow('', imgs)
    cv2.waitKey(0)
    '''
    
    
'''  
if __name__ == '__main__':
    test = images1[0]
    
    test = cv2.cvtColor(test, cv2.COLOR_BGR2LAB)
    
    for i in range(0,len(test)):
        for col in range(0, len(test[0])):
            tmp = test[i][col]
            a = hist_back[tmp[0]][tmp[1]][tmp[2]]
            b = hist_fore[tmp[0]][tmp[1]][tmp[2]]
            
            if b < a:
                test[i][col] = np.array([0, 128, 128])
        
                
    test1 = cv2.cvtColor(test, cv2.COLOR_LAB2RGB)
    #test1 = cv2.fastNlMeansDenoisingColored(test,None,20,20,7,21)
    test = test1
    
    test = cv2.cvtColor(test, cv2.COLOR_BGR2LAB)
    for i in range(0,len(test)):
        for col in range(0, len(test[0])):
            tmp = test[i][col]
            a = hist_back[tmp[0]][tmp[1]][tmp[2]]
            b = hist_fore[tmp[0]][tmp[1]][tmp[2]]
            
            if b < a:
                test[i][col] = np.array([0, 128, 128])

    # 图集
    test = cv2.cvtColor(test, cv2.COLOR_LAB2RGB)
    imgs = np.hstack([images1[0], test1,test])
    # 展示多个
    cv2.imshow("mutil_pic", imgs)
    cv2.waitKey(0)
    #plt.figure()
    #plt.imshow(test1)
    #plt.show()
    #cv2.imshow('', test)
    #cv2.waitKey(0)
'''