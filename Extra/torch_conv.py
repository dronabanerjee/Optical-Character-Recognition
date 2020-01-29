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



#with open('/home/aritrab97/GNV_OCR/data2/train_data.pkl', 'rb') as f:
#	data = pickle.load(f)
data= load_pkl('train_data(1).pkl')
data1=np.asarray(data)
for i in range(len(data1)):
	data1[i]=np.asarray(data1[i])

#pil_img = Image.fromarray(data[1000])
#pil_img.save('lena_square_save.png')	
#print(data1[1000].shape)
#plt.imshow(data1[1000])
#plt.show()
train_path = "/home/aritrab97/GNV_OCR/data1/train/"
test_path = "/home/aritrab97/GNV_OCR/data1/test/"
MODEL_STORE_PATH="/home/aritrab97/GNV_OCR/data2/"

#transformations = transforms.Compose([transforms.ToPILImage(),transforms.Grayscale(num_output_channels=1),transforms.ToTensor()])
transformations = transforms.Compose([transforms.ToPILImage(),transforms.Grayscale(num_output_channels=1)])
#transforms.Resize(50),transforms.ToTensor()])

'''class DatasetMNIST2(Dataset):
		
	def __init__(self, file_path, transform=transformations):
		self.data = load_pkl(file_path)
		self.data1=np.asarray(self.data)
		for i in range(len(self.data1)):
			self.data1[i]=np.asarray(self.data1[i])
		self.transform = transform
		self.labels=np.load('finalLabelsTrain.npy')
		self.image_new=[]	
		for i in range(len(self.data1)):
		
			self.data1[i]=skt.resize(self.data1[i],(50, 50,1),anti_aliasing=True)
			self.image_new.append(self.data[i])	

	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, index):
	# load image as ndarray type (Height * Width * Channels)
	# be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
	# in this example, we use ToTensor(), so we define the numpy array like (H, W, C)
		#image_new=[]
		
		for i in range(len(self.data1)):
		
			data1[i]=skt.resize(self.data1[i],(50, 50,1),anti_aliasing=True)
			self.image_new.append(data[i])
				
		image=np.array(self.image_new)
		label = self.labels
		print(image[6399].shape)
		#if self.transform is not None:
			
		image1 =torch.from_numpy(image)
		print(image1.shape)
		return image1, label'''

X_train_new=[]	
X_test_new=[]
data_X=[]
label_X=[]
data = load_pkl('train_data(1).pkl')
data1=np.asarray(data)
for i in range(len(data1)):
	data1[i]=np.asarray(data1[i])
labels=np.load('finalLabelsTrain.npy')
l = np.load('/home/aritrab97/akash/project01-desi-learners-master/ClassData.npy',allow_pickle=True)
m=np.load('/home/aritrab97/akash/project01-desi-learners-master/ClassLabels.npy',allow_pickle=True)
from sklearn.metrics import confusion_matrix
for i in range(len(data1)):
	if(labels[i]==1 or labels[i]==2):
		data_X.append(data1[i])
		label_X.append(labels[i])
data_X=np.array(data_X)	
label_X=np.array(label_X)

for i in range(len(labels)):
	labels[i]=labels[i]-1
for i in range(len(m)):
	m[i]=m[i]-1
X_train, X_test, y_train, y_test = train_test_split(data_X, label_X, test_size=0.33, random_state=42)

for i in range(len(X_train)):
		
	X_train[i]=skt.resize(X_train[i],((50, 50)),anti_aliasing=True)
	X_train[i] = np.expand_dims(X_train[i], axis=0)
	print(X_train[0].shape)
	
	
	#X_train[i]=X_train[i].astype(np.uint8)
#.reshape([50,50,1])
	
	trans = transforms.ToPILImage()
	trans1 = transforms.Grayscale(num_output_channels=1)
	#data[i]=Image.fromarray(data[i])
	#data[i].save('x.png')
	#print(data[i])
	#data[i]=trans(data[i])
	
	#print(data[i].shape)
	#data[i]=Image.resize((50,50), Image.ANTIALIAS)
	#plt.imshow(trans(data[i]))
	#plt.show()
	
#astype(np.uint8).
	X_train_new.append(X_train[i])

#transform = transform
'''labels=np.load('finalLabelsTrain.npy')


for i in range(len(labels)):
	labels[i]=labels[i]-1'''

for i in range(len(X_test)):
		
	X_test[i]=skt.resize(X_test[i],((50, 50)),anti_aliasing=True)
	X_test[i] = np.expand_dims(X_test[i], axis=0)
	print(l[0])
	
	#X_test[i]=X_test[i].astype(np.uint8)
#.reshape([50,50,1])
	
	trans = transforms.ToPILImage()
	trans1 = transforms.Grayscale(num_output_channels=1)
	#data[i]=Image.fromarray(data[i])
	#data[i].save('x.png')
	#print(data[i])
	#data[i]=trans(data[i])
	
	#print(data[i].shape)
	#data[i]=Image.resize((50,50), Image.ANTIALIAS)
	#plt.imshow(trans(data[i]))
	#plt.show()
	
#astype(np.uint8).
	X_test_new.append(X_test[i])

X_train=np.array(X_train_new)	
X_test=np.array(X_test_new)

print(type(X_train))
print(X_test.shape)
class PrepareData(Dataset):
	def __init__(self, X, y,transform=transformations):
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

print(X_test.shape)
print(y_test.shape)
ds = PrepareData(X=X_train, y=y_train,transform=transformations)
ds1=PrepareData(X=X_test, y=y_test,transform=transformations)
trans = transforms.Compose([transforms.ToPILImage(),transforms.Grayscale(num_output_channels=1)])
#for i in range(len(image)):
#	image[i]=transformations(image[i])
#plt.imshow(x)
#plt.show()
#print(image.shape)
#print(ds1.shape)
train_loader = DataLoader(ds, batch_size=50, shuffle=True)
print('sfdsfd')
print(len(train_loader))
test_loader=DataLoader(ds1, batch_size=len(y_test), shuffle=True)
print(train_loader)
dataiter = iter(train_loader)
imag, labels = dataiter.next()
#print(imag.shape)
#train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)'''
#training = DatasetMNIST2('train_data(1).pkl',transform=transformations)
#img, lab = training.__getitem__(0)
#print(img[0].shape)
#ds = DataLoader(training, batch_size=50, shuffle=True)
#testing=
#train_loader = torch.utils.data.DataLoader(
#    training,
#    batch_size=100,
#    shuffle=True)
#dataiter = iter(train_loader)
#images, labels = dataiter.next()
#print(type(images))
#print(images.shape)
#print(labels.shape)
'''testing = datasets.ImageFolder(test_path,transform=transformations)

test_loader = torch.utils.data.DataLoader(
    testing,
    batch_size=100,
    shuffle=True)'''

num_epochs=20

'''if __name__ == '__main__':

    print("Number of train samples: ", len(training))
    #print("Number of test samples: ", len(test_data))
    print("Detected Classes are: ", training.class_to_idx) # classes are detected by folder structure'''


'''class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.layer1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=5, padding=2),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2))
		self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=5, padding=2),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2))
		self.fc = nn.Linear(32*11*11, 2)
		self.out_act = nn.Sigmoid()
	def forward(self, x):
		out = self.layer1(x)	
		out = self.layer2(out)
		out = out.view(out.size(0), -1)
		out = self.fc(out)
		out=self.out_act(out)
		return out'''



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


'''class Unit(nn.Module):
	def __init__(self,in_channels,out_channels):
		super(Unit,self).__init__()
        

		self.conv = nn.Conv2d(in_channels=in_channels,kernel_size=3,out_channels=out_channels,stride=1,padding=1)
		self.bn = nn.BatchNorm2d(num_features=out_channels)
		self.relu = nn.ReLU()

	def forward(self,input):
		output = self.conv(input)
		output = self.bn(output)
		output = self.relu(output)

		return output

class SimpleNet(nn.Module):
	def __init__(self,num_classes=2):
		super(SimpleNet,self).__init__()

		#Create 14 layers of the unit with max pooling in between
		self.unit1 = Unit(in_channels=1,out_channels=32)
		self.unit2 = Unit(in_channels=32, out_channels=32)
		self.unit3 = Unit(in_channels=32, out_channels=32)

		self.pool1 = nn.MaxPool2d(kernel_size=2)

		self.unit4 = Unit(in_channels=32, out_channels=64)
		self.unit5 = Unit(in_channels=64, out_channels=64)
		self.unit6 = Unit(in_channels=64, out_channels=64)
		self.unit7 = Unit(in_channels=64, out_channels=64)

		self.pool2 = nn.MaxPool2d(kernel_size=2)

		self.unit8 = Unit(in_channels=64, out_channels=128)
		self.unit9 = Unit(in_channels=128, out_channels=128)
		self.unit10 = Unit(in_channels=128, out_channels=128)
		self.unit11 = Unit(in_channels=128, out_channels=128)

		self.pool3 = nn.MaxPool2d(kernel_size=2)

		self.unit12 = Unit(in_channels=128, out_channels=128)
		self.unit13 = Unit(in_channels=128, out_channels=128)
		self.unit14 = Unit(in_channels=128, out_channels=128)

		self.avgpool = nn.AvgPool2d(kernel_size=4)
        
		#Add all the units into the Sequential layer in exact order
		self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, self.unit4, self.unit5, self.unit6
				,self.unit7, self.pool2, self.unit8, self.unit9, self.unit10, self.unit11, self.pool3,
				self.unit12, self.unit13, self.unit14, self.avgpool)

		self.fc = nn.Linear(in_features=128,out_features=num_classes)

	def forward(self, input):
		output = self.net(input)
		output = output.view(-1,128)
		output = self.fc(output)
		return output'''




#model = CNN()
learning_rate=0.001
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
acc_list1=[]
total_step = len(train_loader)
loss_list = []
acc_list = []
val_loss_list=[]
train_loss_list=[]
train_loss=0
#print(ds)
print(labels.shape)
for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		train_loss=0
	# Run the forward pass
		images=Variable(images).float()
		#image = image.type('torch.DoubleTensor')
		labels=Variable(labels).long()
		outputs = model(images)
		print(labels)
		#print(image)
		#print(labels[49])
		#print(type(outputs))
		#print(labels)
		loss = criterion(outputs, labels)
		loss_list.append(loss.item())
		#train_loss+=loss.item()

        # Backprop and perform Adam optimisation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

        # Track the accuracy
		total = labels.size(0)
		_, predicted = torch.max(outputs.data, 1)
		#print(predicted)
		#print('s')		
		#print(labels)
		#break
		#correct = (predicted == labels).sum().item()
		#acc_list.append(correct / total)
		
		print(i)
		if (i + 1) % 86 == 0:
			correct = (predicted == labels).sum().item()
			acc_list.append(correct / total)
			train_loss+=loss
			test_loss = 0
			print('x')
			print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),(correct / total) * 100))
			'''model.eval()
			with torch.no_grad():
				correct1 = 0
				total = 0
				accuracy=0
				for images1, labels1 in test_loader:
					print(images1.shape)
					print(labels1.shape)
					images1=Variable(images1).float()
		#image = image.type('torch.DoubleTensor')
					labels1=Variable(labels1).long()
					print(labels1.shape)
					logps = model.forward(images1)
					print(logps.shape)
					batch_loss = criterion(logps, labels1)
					test_loss += batch_loss.item()
					total = labels1.size(0)
					_, predicted1 = torch.max(logps.data, 1)
					#print(predicted)
					#print('s')		
					#print(labels)
					#break
				correct1 = (predicted1 == labels1).sum().item()

				acc_list1.append(correct1 / total)
				#train_loss_list.append(train_loss/len(train_loader))
				train_loss_list.append(train_loss)
				val_loss_list.append(test_loss/len(test_loader))'''
				#outputs = model(images)
		
'''outputs = model(images)
_, predicted = torch.max(outputs.data, 1)
x=predicted[0]+1
#print(x)
print(predicted)
#print(labels)
total += labels.size(0)
correct += (predicted == labels).sum().item()'''


#	print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))
'''plt.plot(train_loss_list, label='Training loss')
plt.plot(val_loss_list, label='Validation loss')
plt.legend(frameon=False)
plt.yscale('symlog')
plt.show()'''


model.eval()
with torch.no_grad():
	correct = 0
	total = 0
	for images, labels in test_loader:
		images=Variable(images).float()
		#image = image.type('torch.DoubleTensor')
		labels=Variable(labels).long()
		#outputs = model(images)
		outputs = model(images)
		_, predicted = torch.max(outputs.data, 1)
		x=predicted[0]+1
		#print(x)
		print(predicted)
		#print(labels)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()
	cm = confusion_matrix(labels, predicted)
	print(cm)

	print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

# Save the model and plot
torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.pt')
#torch.save(model,MODEL_STORE_PATH + 'conv_net_model.pt')
#plot = plt.imshow(images[0], cmap="Greys")
