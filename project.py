import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from skorch import NeuralNetClassifier
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from skorch.helper import SliceDataset
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import cross_validate

import os
import random

from MasksDataSet import MasksDataSet
# from SliceDataset import SliceDataset

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(16384, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 4)
        )

    def forward(self, x):
        # conv layers
        x = self.conv_layer(x)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = self.fc_layer(x)
        return x


model = CNN()

num_classes = 4
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
batch_size = 64

transform = transforms.Compose(
		[transforms.ToTensor(),
		 transforms.Resize((64, 64)),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
	)


root = 'dataset2'
# tuple of (path, label)
dirs = [(root + '/' + 'nomask',0), (root + '/' + 'surgicalmask',1), (root + '/' + 'clothmask',2), (root + '/' + 'n95',3)]

# this array will arrays of all of the images
# each index corresponds to an image type
# 0 = no mask, 1 = surgical mask, 2 = clothmask, 3 = n95
items = [[],[],[],[]]
# dirssurgicalmask = [(root + '/' + 'surgicalmask',1)]
# dirsclothmask = [(root + '/' + 'clothmask',2)]
# dirsn95 = [(root + '/' + 'n95',3)]

for dir in dirs:
    name = dir[0]
    label = dir[1]
    print("Name of folder", dir)
    files = os.listdir(name)

    images = []
    for f in files:
        #print(f)
        # append tuple of (path, label)
        images.append((dir[0] + '/' + f, label))

    # we shuffle the array everytime so that even thorugh were reading the same images in the same order every time
    # we get a new order of images to split the array on
    random.shuffle(images)

    # afterwards we can append this image a
    items[label] = images

print("No mask",items[0][34])
print("Surgical mask",items[1][209])
print("Cloth",items[2][55])
print("N95",items[3][106])



train_NoMask = items[0][0:299]
test_NoMask = items[0][299:399]

train_SurgicalMask = items[1][0:299]
test_SurgicalMask = items[1][299:399] 

train_ClothMask = items[2][0:299]
test_ClothMask = items[2][299:399]

train_N95 = items[3][0:299]
test_N95 = items[3][299:399]

training = train_NoMask + train_SurgicalMask + train_ClothMask + train_N95
testing = test_NoMask + test_SurgicalMask + test_ClothMask + test_N95
#print("Training", training)
print("testing", testing)


datasetTraining = MasksDataSet(training, transform = transform)
datasetTest = MasksDataSet(testing, transform = transform)



def collate_fn(batch):
   batch = list(filter(lambda x: x is not None, batch))
   return torch.utils.data.dataloader.default_collate(batch) 

# dataLoaderTrain = DataLoader(dataset=datasetTraining, batch_size =batch_size, shuffle=True, collate_fn=collate_fn)
# dataLoaderTest = DataLoader(dataset=datasetTest, batch_size =batch_size, shuffle=True, collate_fn=collate_fn)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("Begin training", DEVICE)

y_train = np.array([y for x, y in iter(datasetTraining)])


classes = ('no mask', 'surgical mask', 'cloth mask',  'n95')

torch.manual_seed(0)
net = NeuralNetClassifier(
		CNN,
		max_epochs=1,
		iterator_train__num_workers=0,
		iterator_valid__num_workers=0,
		lr=1e-3,
		batch_size= batch_size,#8,
		optimizer=optim.Adam,
		criterion=nn.CrossEntropyLoss,
		device=DEVICE
	)

#print(datasetTraining[0])

net.fit(datasetTraining, y=y_train)

print("Done fitting data")


y_pred = net.predict(datasetTest)
y_test = np.array([y for x, y in iter(datasetTest)])
plot_confusion_matrix(net, datasetTest, y_test.reshape(-1, 1))

# print("Accuracy:    ", accuracy_score(y_test, y_pred)*100)
# print("Recall:      ", recall_score(y_true=y_test, y_pred=y_pred, average='weighted')*100)
# print("Precision:   ", precision_score(y_true=y_test, y_pred=y_pred, average='weighted')*100)
# print("F1_Score:    ", f1_score(y_true=y_test, y_pred=y_pred, average='weighted')*100)

torch.save(model.state_dict(), "Model_Phase2.pt")
# print("program done")
# plt.show()

train_sliceable = SliceDataset(datasetTraining)
accuracy = cross_val_score(net, train_sliceable, y_train, cv=10, scoring="accuracy")
print("Accuracy score across 10 folds: ", accuracy)
print("Accuracy mean: ",np.mean(accuracy))


recall = cross_val_score(net, train_sliceable, y_train, cv=10, scoring="recall_macro")
print("Recall score across 10 folds: ", recall)
print('Recall: ',  np.mean(recall))

precision = cross_val_score(net, train_sliceable, y_train, cv=10, scoring="precision_macro")
print("Precision score across 10 folds: ", precision)
print('precision: ', np.mean(precision))

f1 = cross_val_score(net, train_sliceable, y_train, cv=10, scoring="f1_macro")
print("F1 score across 10 folds: ", f1)
print('f1: ', np.mean(f1))














































# dirssurgicalmask = MasksDataSet(dirssurgicalmask, transform = transform)
# dirsclothmask = MasksDataSet(dirsclothmask, transform = transform)
# dirsn95 = MasksDataSet(dirsn95, transform = transform)


# datasetTraining = train_NoMask +train_SurgicalMask + train_ClothMask + train_N95
# datasetTest = test_NoMask + test_SurgicalMask + test_ClothMask + test_N95

# root = 'dataset'
# # tuple of (path, label)
# dirsTraining = [
# 			(root + '/' + 'Train_NoMask',0), 
# 			(root + '/' + 'Train_SurgicalMask', 1),
# 			(root + '/' + 'Train_ClothMask', 2),
# 			(root + '/' + 'Train_N95Mask', 3)
# 		]
# datasetTraining = MasksDataSet(dirsTraining, transform = transform)

# # tuple of (path, label)
# dirsTesting = [
# 			(root + '/' + 'Test_NoMask',0), 
# 			(root + '/' + 'Test_SurgicalMask', 1),
# 			(root + '/' + 'Test_ClothMask', 2),
# 			(root + '/' + 'Test_N95Mask', 3)
# 		]
# datasetTest = MasksDataSet( dirsTesting, transform = transform)
#train_data, val_data = random_split(dataset, [1200, 400])