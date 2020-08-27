# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 18:15:34 2020

@author: eric
"""

import cv2
import numpy as np
from scipy import signal
import math

def calcGrayHist(image):
    rows,cols=image.shape
    grayHist=np.zeros([256],np.uint64)
    for r in range(rows):
        for c in range(cols):
            grayHist[image[r][c]]+=1
    return grayHist


def threshTwoPeaks(image):
    histogram=calcGrayHist(image)
    maxLoc=np.where(histogram==np.max(histogram))
    firstPeak=maxLoc[0][0]
    measureDists=np.zeros([256],np.float32)
    for k in range(256):
        measureDists[k]=pow(k-firstPeak,2)*histogram[k]
    maxLoc2=np.where(measureDists==np.max(measureDists))
    secondPeak=maxLoc2[0][0]
    thresh=0
    if firstPeak>secondPeak:
        temp=histogram[int(secondPeak):int(firstPeak)]
        minLoc=np.where(temp==np.min(temp))
        thresh=secondPeak+minLoc[0][0]+1
    else:
        temp=histogram[int(firstPeak):int(secondPeak)]
        minLoc=np.where(temp==np.min(temp))
        thresh=firstPeak+minLoc[0][0]+1
    threshImage_out=image.copy()
    threshImage_out[threshImage_out>thresh]=255
    threshImage_out[threshImage_out<=thresh]=0
    return (thresh,threshImage_out)

image=cv2.imread("./LSC_data/A1/plant001_rgb.png",cv2.IMREAD_GRAYSCALE)
cv2.imshow("image",image)
thresh,out=threshTwoPeaks(image)
print(thresh)
cv2.imshow("out",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
