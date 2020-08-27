# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 10:18:07 2020

@author: eric
"""
import os 
import numpy as np
from PIL import Image



def AbsDiffFGLabels(inlabel, gtlabel):
    '''
    inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
    gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
    score: Absolute value of difference of the number of foreground labels
    '''
    
    #check if label images have same size
    if (inlabel.shape == gtlabel.shape):
        maxInLabel = int(max(map(max,inlabel)))
        minInLabel = int(min(map(min,inlabel)))
        maxGtLabel = int(max(map(max,gtlabel)))
        minGtLabel = int(min(map(min,gtlabel)))
        
        score = (maxInLabel-minInLabel) - (maxGtLabel-minGtLabel)
        score_abs = abs(score)
        return score_abs, score
    else:
        return -1

def Dice(inlabel, gtlabel, i, j):
    # calculate Dice score for the given labels i and j.
    inMask = (inlabel==i)
    gtMask = (gtlabel==j)
    insize = np.sum(inMask)
    gtsize = np.sum(gtMask)
    overlap = np.sum(inMask & gtMask)
    return 2*overlap/float(insize+gtsize)
    

def BestDice(inlabel, gtlabel):
    '''
    inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
    gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
    score: Dice score
    '''
    if (inlabel.shape == gtlabel.shape):
        maxInLabel = int(max(map(max,inlabel)))
        minInLabel = int(min(map(min,inlabel)))
        maxGtLabel = int(max(map(max,gtlabel)))
        minGtLabel = int(min(map(min,gtlabel)))
        score = 0 # initialize output
        
        # loop all labels of inLabel.
        for i in range(minInLabel, maxInLabel+1):
            sMax = 0
            # loop all labels of gtLabel.
            for j in range(minGtLabel, maxGtLabel+1):
                s = Dice(inlabel, gtlabel, i, j)
                # keep max Dice value for label i.
                sMax = max(s, sMax)
            score += sMax # sum up best found values.
        score = score/float(maxInLabel-minInLabel+1)
        return score
    else:
        return 0

def FBDice(inlabel, gtlabel):
    if (inlabel.shape == gtlabel.shape):
        
        minInLabel = int(min(map(min,inlabel)))
        minGtLabel = int(min(map(min,gtlabel)))
        inMask = (inlabel > minInLabel)
        gtMask = (gtlabel > minGtLabel)
        inSize = np.sum(inMask)
        gtSize = np.sum(gtMask)
        overlap = np.sum(inMask & gtMask)
        return 2*overlap/float(inSize + gtSize)
    else:
        return 0


def DiffFGLabels(inLabel,gtLabel):
# input: inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
#        gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
# output: difference of the number of foreground labels

    # check if label images have same size
    if (inLabel.shape!=gtLabel.shape):
        return -1

    maxInLabel = np.int(np.max(inLabel)) # maximum label value in inLabel
    minInLabel = np.int(np.min(inLabel)) # minimum label value in inLabel
    maxGtLabel = np.int(np.max(gtLabel)) # maximum label value in gtLabel
    minGtLabel = np.int(np.min(gtLabel)) # minimum label value in gtLabel

    return  (maxInLabel-minInLabel) - (maxGtLabel-minGtLabel) 

def BestDice(inLabel,gtLabel):
# input: inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
#        gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
# output: score: Dice score
#
# We assume that the lowest label in inLabel is background, same for gtLabel
# and do not use it. This is necessary to avoid that the trivial solution, 
# i.e. finding only background, gives excellent results.
#
# For the original Dice score, labels corresponding to each other need to
# be known in advance. Here we simply take the best matching label from 
# gtLabel in each comparison. We do not make sure that a label from gtLabel
# is used only once. Better measures may exist. Please enlighten me if I do
# something stupid here...

    score = 0 # initialize output
    
    # check if label images have same size
    if (inLabel.shape!=gtLabel.shape):
        return score
    
    maxInLabel = np.max(inLabel) # maximum label value in inLabel
    minInLabel = np.min(inLabel) # minimum label value in inLabel
    maxGtLabel = np.max(gtLabel) # maximum label value in gtLabel
    minGtLabel = np.min(gtLabel) # minimum label value in gtLabel
    
    if(maxInLabel==minInLabel): # trivial solution
        return score
    
    for i in range(minInLabel+1,maxInLabel+1): # loop all labels of inLabel, but background
        sMax = 0; # maximum Dice value found for label i so far
        for j in range(minGtLabel+1,maxGtLabel+1): # loop all labels of gtLabel, but background
            s = Dice(inLabel, gtLabel, i, j) # compare labelled regions
            # keep max Dice value for label i
            if(sMax < s):
                sMax = s
        score = score + sMax; # sum up best found values
    score = score/(maxInLabel-minInLabel)
    return score

##############################################################################
def FGBGDice(inLabel,gtLabel):
# input: inLabel: label image to be evaluated. Background label is assumed to be the lowest one.
#        gtLabel: ground truth label image. Background label is assumed to be the lowest one.
# output: Dice score for foreground/background segmentation, only.

    # check if label images have same size
    if (inLabel.shape!=gtLabel.shape):
        return 0

    minInLabel = np.min(inLabel) # minimum label value in inLabel
    minGtLabel = np.min(gtLabel) # minimum label value in gtLabel

    one = np.ones(inLabel.shape)    
    inFgLabel = (inLabel != minInLabel*one)*one
    gtFgLabel = (gtLabel != minGtLabel*one)*one
    
    return Dice(inFgLabel,gtFgLabel,1,1) # Dice score for the foreground

    
##############################################################################
def Dice(inLabel, gtLabel, i, j):
# calculate Dice score for the given labels i and j
    
    # check if label images have same size
    if (inLabel.shape!=gtLabel.shape):
        return 0

    one = np.ones(inLabel.shape)
    inMask = (inLabel==i*one) # find region of label i in inLabel
    gtMask = (gtLabel==j*one) # find region of label j in gtLabel
    inSize = np.sum(inMask*one) # cardinality of set i in inLabel
    gtSize = np.sum(gtMask*one) # cardinality of set j in gtLabel
    overlap= np.sum(inMask*gtMask*one) # cardinality of overlap of the two regions
    if ((inSize + gtSize)>1e-8):
        out = 2*overlap/(inSize + gtSize) # Dice score
    else:
        out = 0

    return out    
    
##############################################################################
def AbsDiffFGLabels(inLabel,gtLabel):
# input: inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
#        gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
# output: Absolute value of difference of the number of foreground labels

    return np.abs( DiffFGLabels(inLabel,gtLabel) )    
if __name__ == '__main__':   
    path1 = "./LSC_2017/train/A1"
    path2 = "./LSC_2017/A1"
    files_name = os.listdir(path1)
    pres_name, labels_name = [], []
    
    for file in files_name:
        if 'label' in file:
            pres_name.append(path1+'/'+file)
            
            label = file.replace('label', 'label_mask')
            labels_name.append(path2+'/'+label)
    
    inlabel = np.array(Image.open(pres_name[0]))                               
    gtlabel = np.array(Image.open(labels_name[0]))
    abscores = []
    scores = []
    BD = []
    FBD = []
    for inlabel, gtlabel in zip(pres_name, labels_name):
        inlabel = np.array(Image.open(inlabel))
        gtlabel = np.array(Image.open(gtlabel))
        #score_abs, score = AbsDiffFGLabels(inlabel, gtlabel)
        score = DiffFGLabels(inlabel, gtlabel)
        score_abs = AbsDiffFGLabels(inlabel, gtlabel)
        bd = BestDice(inlabel, gtlabel)
        fbd = FBDice(inlabel, gtlabel)
        
        FBD.append(fbd)
        BD.append(bd)
        abscores.append(score_abs)
        scores.append(score)
        
    
    print('Mean SBD:\t', np.mean(BD))
    print('Std SBD:\t', np.std(BD))
    
    print('Mean FBD:\t', np.mean(FBD))
    print('Std FBD:\t', np.std(FBD))
    
    print('Mean Dic:\t',np.mean(scores))
    print('Std Dic:\t', np.std(scores))
    
    print('Mean |Dic|:\t', np.mean(abscores))
    print('Std |Dic|:\t', np.std(abscores))
    