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
from torch.utils.data import random_split
from dataSet import MasksDataSet

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
            nn.Linear(8 * 8 * 64, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
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
num_epochs = 10
num_classes = 4
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
batch_siuze = 32

transform = transforms.Compose(
		[transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
	)

datasetTraining = MasksDataSet( root = 'dataset', transform = transforms.ToTensor(), 0, 299)
datasetTest = MasksDataSet(root = 'dataset', transform = transforms.ToTensor(), 300, 399)
#train_data, val_data = random_split(dataset, [1200, 400])

dataLoaderTrain = DataLoader(dataset=datasetTraining, batch_size =batch_size, shuffle=True)
dataLoaderTest = DataLoader(dataset=datasetTest, batch_size =batch_size, shuffle=True)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("Begin training", DEVICE)

y_train = np.array([y for x, y in iter(train_data)])

classes = ('no mask', 'cloth mask', 'regular mask', 'n95')

torch.manual_seed(0)
# net = NeuralNetClassifier(
# 		CNN,
# 		max_epochs=1,
# 		iterator_train__num_workers=0,
# 		iterator_valid__num_workers=0,
# 		lr=1e-3,
# 		batch_size=64,
# 		optimizer=optim.Adam,
# 		criterion=nn.CrossEntropyLoss,
# 		device=DEVICE
# 	)



# net.fit(train_data, y=y_train)

# print("Done fitting data")


# y_pred = net.predict(testset)
# y_test = np.array([y for x, y in iter(testset)])
# accuracy_score(y_test, y_pred)
# plot_confusion_matrix(net, testset, y_test.reshape(-1, 1))
# plt.show()

# print("115")

# net.fit(train_data, y=y_train)
# train_sliceable = SliceDataset(train_data)
# scores = cross_val_score(net, train_sliceable, y_train, cv=5,
# scoring="accuracy")

# print("program done")
for epoch in range(num epochs):
   losses = []

   for batch_idx, (data, targets) in enumerate(dataLoaderTrain):
     data • data.to(device=device)
     targets = targets.to(device=device)

     scores = model(data)
     loss = criterion(scores, targets) 

     losses.append(loss.item())

     optimizer.zero_grad()
     loss.backward()

     optimizer.step()

