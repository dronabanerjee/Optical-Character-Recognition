# Foundtions of Machine Learning Final Project

This document is an instruction for running the implementation of handwritten character recognition performed by GNV_OCR.

## The packages used are: -

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import pickle

import os

import argparse

import sys, getopt

from PIL import Image

from loaddata import load_pkl

from skimage.feature import hog

from skimage.color import rgb2grey

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.svm import SVC

from skimage import transform as skt

from sklearn.metrics import roc_curve, auc

from PIL import Image

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

## Model Description:

Our proposed model is a HOG feature extraction,scaling, PCA dimensionality reduction and SVM based classifier. To test our code please use svm_model_12.pkl and svm_model_all.pkl.

svm_model_12.pkl - This is for class 1 and class 2 prediction.
svm_model_all.pkl - This is for all class classification.


## The implementation has 7 files and 1 folder: -

1) train.py
2) test.py
3) svm_model_12.pkl
4) svm_model_all.pkl
5) train12.pkl
6) train_data.pkl
7) finalLabelsTrain.npy
8) Extra

## 1) train.py: -

###### The file takes in three inputs:-
 - data path 
 - path of labels 
 - Choice of classification(Either class 1 and class 2 or all classes)

###### sample command for class 1 and class2 training -  
python train.py -data 'train_data.pkl' -labels 'finalLabelsTrain.npy' -choice 'AB'

###### sample command for all classes training - 
python train.py -data 'train_data.pkl' -labels 'finalLabelsTrain.npy' -choice 'All_classes'

###### The python file train.py has the following functions: -

 - universal(argv)- This function controls the training process of the entire implementation, which has two modes. The first mode is to train the data having label's 1 and 2. The second mode is to train the data on all the given labels in the train dataset.

 - load_data_AB(data,labels)- This function takes the path of data and labels and the loads the pickle data. It then conerts the pickle data to a numpy array. Then it extracts out the corresponding data for label's 1 and 2. After that it resizes the corresponding extracted data to 50x50 size and returns the resized data and corresponding labels.

 - load_data_All_Classes(data,labels)- This function takes the directory of the data and labels. Loads the pickle data and converts the pickle data into numpy array. It then resizes all the data points into 50x50 and returns the data and labels.

 - hog_features(data1,labels)- This function takes in the data returned by the load_data function and applies Histogram of Ordinal Gradients(HOG) feature descriptor, stores the features extracted by HOG and retuns the extracted features.

 - apply_pca(feature_matrix)- This function takes the features extracted by HOG as input, scales the data using Standard Scaler, after scaling it applies Principal Component Analysis(PCA) for dimensionality reduction and return a pandas dataframe containing features after PCA.

 - model_fit(X,labels)- This function takes the panadas dataframe containing features after applying PCA as input. After that we perform a train and test data split. Where we take a certain percentage of data and labels for training and the rest of the data and labels for testing. So that we can make train the model on a particular data and after the training process is complete we test how well our model has been able to distinguish between/among different class labels by using the test data. After finding out the prediction of our model on the test data we find out the accuracy by comparing the predictions with the actual class labels. We have used different machine learning algorithms as our model for training and testing the data.

## 2) test.py: -
The file takes in three inputs:-
 - data path 
 - trained model path 
 - Choice of classification(Either class 1 and class 2 or all classes)

###### sample command for class 1 and class2 -  
python test.py -data 'train12.pkl' -model 'svm_model_12.pkl' -choice 'ab'

###### sample command for all classes - 
python test.py -data 'train_data.pkl' -model 'svm_model_all.pkl' -choice 'all'

###### The python file test.py has the following functions: -     

 - universal(argv)-This function controls the testing process of the provided model on blind test data.

 - load_data(data)- This function takes the path to the pickle data as input. It then loads the pickle data and converts the pickle data into numpy ndarray and then resizes the data points into 50x50 and returns the resized data.

 - extract_feature(data1)- This function takes in the data returned by the load_data function and applies Histogram of Ordinal Gradients(HOG) feature descriptor, stores the features extracted by HOG and retuns the extracted features.

 - apply_pca(feature_matrix)-This function takes the features extracted by HOG as input, scales the data using Standard Scaler, after scaling it applies Principal Component Analysis(PCA) for dimensionality reduction and return a pandas dataframe containing features after PCA.

 - model_predict(X1,svm_model)- This function takes the pandas dataframe returned by apply_pca function and the path to the svm model as input. It loads the svm model(If the data to be tested is having labels 1 and 2 then use 'svm_model_12.pkl' and if you want to test the data on all classes then use 'svm_model_all.pkl'). After loading the svm model it predicts the class labels and stores them in a numpy array and saves them in a file named 'predicted_labels.npy'.  

## 3) svm_model_12.pkl- 
Model of svm classifier with rbf kernel trained on dataset having class 1 and class2 
 
## 4) svm_model_all.pkl- 
Model of svm classifier with rbf kernel trained on dataset having all classes

## 5) train12.pkl- 
train data of class 1 and class 2

## 6) train_data.pkl- 
train data of all classes

## 7) finalLabelsTrain.npy- 
Labels of training data of all classes

## 8) Extra- 
It is a folder containing our CNN implementation. It is not our proposed model. SVM is our proposed model.
   
     
 
