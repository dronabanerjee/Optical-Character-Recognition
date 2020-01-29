import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
#from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
from skimage import transform as skt
from PIL import *
from loaddata import load_pkl
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
#from torch_conv import class_definition
#from torch_conv import Net


transformations = transforms.Compose([transforms.ToPILImage(),transforms.Grayscale(num_output_channels=1)])
def load_data():
	
	data= load_pkl('train_data(1).pkl')
	data1=np.asarray(data)
	for i in range(len(data1)):
		data1[i]=np.asarray(data1[i])

	labels=np.load('finalLabelsTrain.npy')
	for i in range(len(labels)):
		labels[i]=labels[i]-1
	
	for i in range(len(data)):
		
		data[i]=skt.resize(data[i],((50, 50)),anti_aliasing=True)
		data[i] = np.expand_dims(data[i], axis=0)
		trans = transforms.ToPILImage()
		trans1 = transforms.Grayscale(num_output_channels=1)
		data_new.append(data[i])
	data1=np.array(data_new)	

	return data1,labels	

class PrepareData(Dataset):
	def __init__(self, X, y):
		if not torch.is_tensor(X):
			self.X = torch.from_numpy(X)
			#print(image.shape)
			#image=transform(self.X)
			#print(image[0].shape)
			#for i in range(len(X)):
			#	image[i]=transformations(X[i])
			#	print(image[i].shape)
			#print('jsj')
			#print(image[i].shape)
		
		
		if not torch.is_tensor(y):
			#self.y=y.astype(int)
			self.y = torch.Tensor(y)

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		#print(self.X[idx].shape)
		return self.X[idx], self.y[idx]

class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		# 1 input image channel, 6 output channels, 3x3 square convolution
		# kernel
		self.conv1 = nn.Conv2d(1, 6, 3)
		self.conv2 = nn.Conv2d(6, 16, 3)
		# an affine operation: y = Wx + b
		self.fc1 = nn.Linear(16 * 11 * 11, 120)  # 6*6 from image dimension
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 8)

	def forward(self, x):
		# Max pooling over a (2, 2) window
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		# If the size is a square you can only specify a single number
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		x = x.view(-1, self.num_flat_features(x))
		#x=x.view(1,-1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		print(num_features)
		return num_features


# create a complete CNN
model = Net()

PATH="/home/aritrab97/GNV_OCR/data2/conv_net_model.pt"
data1,labels=load_data()
ds = PrepareData(X=data1, y=labels)
test_loader=DataLoader(ds, batch_size=len(data1), shuffle=True)	
#model = Net()
model.load_state_dict(torch.load(PATH))
#model.load_state_dict(PATH)
#model=torch.load(PATH)


model.eval()
with torch.no_grad():
	correct = 0
	total = 0
	z=0
	for images, labels in test_loader:
		images=Variable(images).float()
		#image = image.type('torch.DoubleTensor')
		labels=Variable(labels).long()
		#outputs = model(images)
		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)
		x=predicted[0]+1
		print(predicted)
		print(labels)
		total += labels.size(0)
		for i in range(len(labels)):
			if(labels[i]==predicted[i]):
				z=z+1
		print(z)
		correct += (predicted == labels).sum().item()
		print(correct)
		print(total)
	print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

