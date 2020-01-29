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

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from skimage import transform as skt
from sklearn.metrics import roc_curve, auc
from PIL import Image
import pandas as pd


def load_data(data):
	
	data1= load_pkl(data)
	data1=np.asarray(data1)
	l=len(data1)
	for i in range(l):
		data1[i]=np.asarray(data1[i])
		
	labels=np.load('finalLabelsTrain.npy')
	for i in range(len(data1)):
		data1[i]=skt.resize(data1[i],((50, 50)),anti_aliasing=True)
		
		
	return data1



def extract_feature(data1):
	features_list = []
	count=0
	for i in range(len(data1)):
		hog_features = hog(data1[i], block_norm='L2-Hys', pixels_per_cell=(16, 16))	
		features_list.append(hog_features)
	feature_matrix = np.array(features_list)
	return feature_matrix

def apply_pca(feature_matrix):
	scale = StandardScaler()
	
	X_stand = scale.fit_transform(feature_matrix)
	
	pca = PCA(n_components=75)
	
	X_stand_pca = pca.fit_transform(X_stand)
	
	X = pd.DataFrame(X_stand_pca)
	
	return X

def model_predict(X1,svm_model,choice):
	

	

	y_pred = svm_model.predict(X1)
	print(y_pred)
	if(choice=='ab'):
		np.save('predicted_labels_12.npy', y_pred)
	elif(choice=='all'):
		np.save('predicted_labels_all.npy', y_pred)	
	return y_pred

def universal(argv):
	parser = argparse.ArgumentParser(description='Input for Machine Learning Algorithm')
	parser.add_argument('-data', '--data_file', type=str, nargs=1, help='Pickle input file')
	parser.add_argument('-model', '--model_file', type=str, nargs=1, help='Trained model for inference')
	parser.add_argument('-choice', '--choice_of_model', type=str, nargs=1, help='ab or all')
	
	#parser.add_argument('-choice', '--choice_of_classification', type=str, nargs=1, help='AB or All_classes')
	
	args = parser.parse_args()
	data=args.data_file[0]
	model=args.model_file[0]
	choice=args.choice_of_model[0]
	
	#if(choice=='AB'):
	'''data_X,label_X=load_data_AB(data,labels)
	feature_matrix=hog_features(data_X,label_X)
	X=apply_pca(feature_matrix)
	prediction,pred_prob=model_predict(X,label_X)'''
	svm_pkl = open(model, 'rb')
	model= pickle.load(svm_pkl)
	#if(choice=='All_classes'):
	data1=load_data(data)
	feature_matrix=extract_feature(data1)
	print(feature_matrix.shape)
	X=apply_pca(feature_matrix)
	prediction=model_predict(X,model,choice)












if __name__ == "__main__":
	universal(sys.argv)

