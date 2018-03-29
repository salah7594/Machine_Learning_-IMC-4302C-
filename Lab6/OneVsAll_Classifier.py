#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One Vs All classifier module

@author: SlimBenAmor
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

# one vs all train function
def Train_OneVsAll(X,Y,lambda_):
    """
    Train K logitic classifiers for multiclass classification with  One vs All approach
    
    Args:
        X: input digit images
        Y: labels of the digit images
        lambda_: Regularization parameter
    
    Return:
        Logistic_Regr_List: list of K logistic classifier one per class
    """
    
    K=np.unique(Y).shape[0] # number of class
    Logistic_Regr_List=[]
    for digit in range(K):
        y=(Y==digit)                                  # create binary label vector for each classifier
        y=y.astype(int)
        
        logistic_regr=LogisticRegression(C=1/lambda_) # ** your code here**    # create logistic regression classifier object
        logistic_regr.fit(X, y)                       # ** your code here**    # train logistic regression classifier object
        
        Logistic_Regr_List.append(logistic_regr)      # append the trained logistic classifier to the output list
    return Logistic_Regr_List


# one vs all predition function
def Predict_OneVsAll(classifier_list,X):
    """
    Predict the labels of input images in 'X' array using the trained K logitic classifiers 'classifier_list' 

    Return:
        y_pred: vector of predicted label for each input image
    """
    K = len(classifier_list)                # number of class
    pred_proba=np.zeros((X.shape[0],K))     # array contains m line one for each input and K columns represent the probability to
                                            # belong to the corresponding classifier label
    
    # calculate the probabilty for an input image to have as label the number of classifier 'digit' from 0 to K
    for digit in range(K):
        pred_proba[:,digit]=classifier_list[digit].predict_proba(X)[:,1]  # ** your code here**   
        
    y_pred=np.argmax(pred_proba,axis=1)  # ** your code here**   
    return y_pred