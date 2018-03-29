#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main function that call other function and module to make training, prediction and test

@author: SlimBenAmor
"""

import numpy as np
import OneVsAll_Classifier, predict_your_digits
import Display_digits as Disp

# load data
hand_writing = np.loadtxt('hand_writing.txt') 

# extract columns
m = hand_writing.shape[0] # number of samples/images in hand_writing dataset
X=hand_writing[:,:-1]
Y=hand_writing[:,-1,np.newaxis]
n=X.shape[1]  # number of features (number of pixel per image)
nbr_class=np.unique(Y).shape[0]  # number of classes K

# visualize some digits from hand_writing dataset
nbr_select=100
rand_perm=np.random.permutation(m)
X_display=X[rand_perm[:nbr_select],:]
Disp.Display_digits(X_display,Y[rand_perm[:nbr_select],:])

# train K logistic classifiers
lambda_ = .1 # reguralization parameter
Logistic_Regr_List = OneVsAll_Classifier.Train_OneVsAll(X,Y,lambda_)

#predict class on training dataset
y_pred = OneVsAll_Classifier.Predict_OneVsAll(Logistic_Regr_List,X)# ** your code here**   

# calculate train accuracy
train_accuracy = np.sum(Y==y_pred[:,np.newaxis])/m*100 # ** your code here** 
print("train accuracy= ",train_accuracy ,"%")

# test our multiclass classifier
for c in range(4):
    predict_digit=predict_your_digits.predict("test_digit"+str(c)+".jpg",Logistic_Regr_List)
    print("the digit in the image is",c," our One Vs All classifier predict",predict_digit)
