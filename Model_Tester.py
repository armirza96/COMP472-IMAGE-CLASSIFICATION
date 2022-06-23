
# 1) Change name to model that you wish to evaluate, and ensure it is in same directory
model_name = "model_phase2.pt" #"model.pt"

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

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = NeuralNetClassifier(
        CNN,
        max_epochs=1,
        iterator_train__num_workers=0,
        iterator_valid__num_workers=0,
        lr=1e-3,
        batch_size= 64,#8,
        optimizer=optim.Adam,
        criterion=nn.CrossEntropyLoss,
        device=DEVICE
)

model = CNN()
model.load_state_dict(torch.load(model_name))
model.eval()

transform = transforms.Compose(
		[transforms.ToTensor(),
		 transforms.Resize((64, 64)),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
	)


# 2) Change Root by uncommenting the line you wish to test
#root = 'Dataset_General'
root = 'Dataset_Test_Bias_Gender'
#root = 'Dataset_Test_Bias_Race'


dirs = [(root + '/' + 'nomask',0), (root + '/' + 'surgicalmask',1), (root + '/' + 'clothmask',2), (root + '/' + 'n95',3)]
items = [[],[],[],[]]

for dir in dirs:
    name = dir[0]
    label = dir[1]
    print("Name of folder", dir)
    files = os.listdir(name)

    images = []
    for f in files:
        images.append((dir[0] + '/' + f, label))

    items[label] = images


# 3) Once again, uncomment the lines you wish to test

#test_NoMask_General = items[0][299:399]
test_NoMask_Bias_Gender = items[0]
#test_NoMask_Bias_Race = items[0][0:10]

#test_SurgicalMask_General = items[1][299:399] 
test_SurgicalMask_Bias_Gender = items[1]
#test_SurgicalMask_Bias_Race = items[1][0:10]

#test_ClothMask_General = items[2][299:399]
test_ClothMask_Bias_Gender = items[2]
#test_ClothMask_Bias_Race = items[2][0:10]

#test_N95_General = items[3][299:399]
test_N95_Bias_Gender = items[3]
#test_N95_Bias_Race = items[3][0:10]


# 4) Uncomment the lines you wish to test

#testing_General = test_NoMask_General + test_SurgicalMask_General + test_ClothMask_General + test_N95_General
testing_Bias_Gender = test_NoMask_Bias_Gender + test_SurgicalMask_Bias_Gender + test_ClothMask_Bias_Gender + test_N95_Bias_Gender 
#testing_Bias_Race = test_NoMask_Bias_Race + test_SurgicalMask_Bias_Race + test_ClothMask_Bias_Race + test_N95_Bias_Race 

#print("testing", testing_General[50], testing_General[150], testing_General[250],testing_General[350])


# 5) In the below line, replace 'testing_General' with the dataset you wish to test, if you changed it
datasetTest = MasksDataSet(testing_Bias_Gender, transform = transform)

# y_pred = net.predict(datasetTest)
# y_test = np.array([y for x, y in iter(datasetTest)])
# plot_confusion_matrix(net, datasetTest, y_test.reshape(-1, 1))

bar_graph_values = [[],[],[],[]]

labels = ['No Mask', 'Surgical', 'Cloth', 'N95']
true_positives = [0, 0, 0, 0]
false_positives = [0, 0, 0, 0]
  
plt.title(f'True positives vs False positives')

for img in datasetTest:
    portion = img # tuple of (image, label)
    image = portion[0] # image that we are working with from the dataset test index
    true_target = portion[1] # label of the image

    # Reshape image
    #image = image.reshape(64, 64, 1)

    #x = transform(image)  # Preprocess image
    x = image.unsqueeze(0)  # Add batch dimension

    # Generate prediction
    prediction = model(x)

    # Predicted class value using argmax
    predicted_class = torch.argmax(prediction)

    # Show result
    #plt.imshow(image)
    #plt.title(f'Prediction: {predicted_class} - Actual target: {true_target}')
    #plt.show()
    #print(f'Prediction: {predicted_class} - Actual target: {true_target}')

    if predicted_class == true_target:
        true_positives[true_target] = true_positives[true_target] + 1
    else:
        false_positives[true_target] = false_positives[true_target] + 1
    # print("Accuracy:    ", accuracy_score(y_test, y_pred)*100)
    # print("Recall:      ", recall_score(y_true=y_test, y_pred=y_pred, average='weighted')*100)
    # print("Precision:   ", precision_score(y_true=y_test, y_pred=y_pred, average='weighted')*100)
    # print("F1_Score:    ", f1_score(y_true=y_test, y_pred=y_pred, average='weighted')*100)

    # print("program done")
    # plt.show()
print(true_positives, false_positives)
# plot bars in stack manner
plt.bar(labels, true_positives, color='g')
plt.bar(labels, false_positives, bottom=true_positives, color='r')
plt.show()
