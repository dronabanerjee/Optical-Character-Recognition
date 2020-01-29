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

#from sklearn.neighbors import KNeighborsClassifiers


#This part of the code loads the data for class 1 and class 2
def load_data_AB(data,labels):
	data_X=[]
	label_X=[]
	data1= load_pkl(data)
	data1=np.asarray(data1)
	for i in range(len(data1)):
		data1[i]=np.asarray(data1[i])
	labels=np.load('finalLabelsTrain.npy')
	

	for i in range(len(data1)):
		if(labels[i]==1 or labels[i]==2):
			data_X.append(data1[i])
			label_X.append(labels[i])
	data_X=np.array(data_X)	
	label_X=np.array(label_X)

	for i in range(len(data_X)):
		data_X[i]=skt.resize(data_X[i],((50, 50)),anti_aliasing=True)
		
		
	return data_X,label_X

#This part of the code loads the data for all classes
def load_data_All_Classes(data,labels):

	data1= load_pkl(data)
	data1=np.asarray(data1)
	for i in range(len(data1)):
		data1[i]=np.asarray(data1[i])
	labels=np.load('finalLabelsTrain.npy')
	for i in range(len(data1)):
		data1[i]=skt.resize(data1[i],((50, 50)),anti_aliasing=True)
		
		
	return data1,labels

#This part of the code extracts HOG features
def hog_features(data1,labels):
	features_list = []
	count=0
	for i in range(len(labels)):
		hog_features = hog(data1[i], block_norm='L2-Hys', pixels_per_cell=(16, 16))
		#print(hog_features)
		features_list.append(hog_features)
	feature_matrix = np.array(features_list)
	return feature_matrix

#This part of the code performs scaling and then PCA
def apply_pca(feature_matrix):
	scale = StandardScaler()
	X_stand = scale.fit_transform(feature_matrix)
	pca = PCA(n_components=75)
	X_stand_pca = pca.fit_transform(X_stand)
	X = pd.DataFrame(X_stand_pca)
	return X

#This part of the code trains the data
def model_fit(X,labels,choice):

	X_train, X_test, y_train, y_test = train_test_split(X,labels,test_size=.3,random_state=1234123)
	svm = SVC(kernel='rbf', probability=True, random_state=42)
	#svm = KNeighborsClassifier(n_neighbors=5)
	#svm=LogisticRegression(random_state=0, solver='newton-cg',multi_class='multinomial')
	#svm=DecisionTreeClassifier(random_state=0)


	svm.fit(X_train, y_train)
	if(choice=='AB'):
		
		svm_pkl = open('svm_model_12.pkl', 'wb')
	elif(choice=='All_classes'):
		svm_pkl = open('svm_model_all.pkl', 'wb')
	pickle.dump(svm, svm_pkl)
	svm_pkl.close()
	y_pred = svm.predict(X_test)


	accuracy = accuracy_score(y_test, y_pred)
	print('Accuracy of the model: ', accuracy)
	probabilities = svm.predict_proba(X_test)
	print(y_pred)
	return y_pred,probabilities

#This is the function that drives the entire program
def universal(argv):
	parser = argparse.ArgumentParser(description='Input For Universl Parser: This version aimed at Trojan Insertion.')
	parser.add_argument('-data', '--data_file', type=str, nargs=1, help='Pickle input file')
	parser.add_argument('-labels', '--labels_file', type=str, nargs=1, help='labels of the data')
	parser.add_argument('-choice', '--choice_of_classification', type=str, nargs=1, help='AB or All_classes')
	
	args = parser.parse_args()
	data=args.data_file[0]
	labels=args.labels_file[0]
	choice=args.choice_of_classification[0]
	
	if(choice=='AB'):
		data_X,label_X=load_data_AB(data,labels)
		feature_matrix=hog_features(data_X,label_X)
		X=apply_pca(feature_matrix)
		prediction,pred_prob=model_fit(X,label_X,choice)

	if(choice=='All_classes'):
		data1,labels=load_data_All_Classes(data,labels)
		feature_matrix=hog_features(data1,labels)
		X=apply_pca(feature_matrix)
		prediction,pred_prob=model_fit(X,labels,choice)


if __name__ == "__main__":
	universal(sys.argv)
		
		
		
		
		


		
