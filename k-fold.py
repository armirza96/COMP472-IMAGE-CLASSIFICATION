

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
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold

import os
import random

from MasksDataSet import MasksDataSet

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












# digits = datasets.load_digits() # features matrix
# n_samples = len(items) * len(items[0])
ini_array1 = np.array(datasetTest, dtype="object")

# X = ini_array1.flatten()
X = datasetTest
Y = datasetTest
kf = KFold(n_splits=10)
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, alpha=1e-4, solver='sgd', verbose=10, random_state=1, learning_rate_init=0.001)
for train_index, test_index in kf.split(X, Y):
    x_train_fold = X[train_index]
    y_train_fold = Y[train_index]
    x_test_fold = X[test_index]
    y_test_fold = Y[test_index]
    mlp.fit(x_train_fold, y_train_fold)
    print(mlp.score(x_test_fold, y_test_fold))