# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 18:47:06 2020

@author: eric
"""

import ssl
import urllib.request
import cv2
import numpy as np


#装饰图转化为二值图
def getBinaryImage(image):
    #如果 src(x,y)>threshold,dst(x,y) = 0; 否则,dst(x,y) = max_value
    #像素>0 --> = 0
    #像素=0 --> = 255
    b,g,r = cv2.split(image)
    _, b = cv2.threshold(b,0,255,cv2.THRESH_BINARY)
    _, g = cv2.threshold(g,0,255,cv2.THRESH_BINARY)
    _, r = cv2.threshold(r,0,255,cv2.THRESH_BINARY)
    t = b | g | r
    mask_fore = cv2.merge([t,t,t])
    
    _, b = cv2.threshold(b,0,255,cv2.THRESH_BINARY_INV)
    _, g = cv2.threshold(g,0,255,cv2.THRESH_BINARY_INV)
    _, r = cv2.threshold(r,0,255,cv2.THRESH_BINARY_INV)
    t = b & g & r
    mask_back = cv2.merge([t,t,t])
    
    return mask_fore, mask_back



def getLeafRGB(image_test, image_mask):
    # Get foreground image
    mask_fore, mask_back = getBinaryImage(image_mask)
    
    image_fore = cv2.bitwise_and(mask_fore,image_test)
    image_back = cv2.bitwise_and(mask_back,image_test)
    
    return image_fore, image_back



if __name__ == '__main__':
    image_test = cv2.imread("./LSC_data/A1/plant159_rgb.png")
    
    image_mask = cv2.imread("./LSC_data/A1/plant159_label.png")
    
    fore, back = getLeafRGB(image_test, image_mask)
    cv2.imshow("Image", fore)
    cv2.waitKey(0)
    
    cv2.imshow("Image", back)
    cv2.waitKey(0)
