#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test module

@author: SlimBenAmor
"""

import numpy as np
from skimage.transform import resize
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from math import floor
import OneVsAll_Classifier

def predict(filename,Logistic_Regr_List):
    """
    Predict the written digit on the input image
    
    Args:
        filename: input image filename
        Logistic_Regr_List: list of K trained logistic classifiers
    
    Return:
        predicted label for input image
    """
    # read image file
    img=mpimg.imread(filename)
    img=np.mean(img,axis=-1)  # convert to grayscale image
    
    # resize the image
    height,width = img.shape
    height_step = floor(height/20) 
    width_step = floor(width/20) 
    if(height_step>0) and (width_step>0):
        img=img[::height_step,::width_step]
    else:
        img=resize(img,(20,20),order=1)
        
    # change the color scale from 0-255 integer to [-1,1] float
    img=(img.T/128-1)
    
    # draw the digit image
    #plt.figure('handwritten digit',figsize=(9,4))
    plt.imshow(img.T,cmap="gray",vmax=1,vmin=-1)
    plt.axis('off')
    plt.show()
    
    # predict the digit
    prediction=OneVsAll_Classifier.Predict_OneVsAll(Logistic_Regr_List,img.flatten()[np.newaxis,:])
    return prediction
