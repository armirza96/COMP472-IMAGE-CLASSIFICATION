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
# num_epochs = 10
num_classes = 4
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
batch_size = 8

transform = transforms.Compose(
		[transforms.ToTensor(),
		 transforms.Resize((64, 64)),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
	)
	
root = 'dataset'
# tuple of (path, label)
dirsTraining = [
			(root + '/' + 'Train_NoMask',0), 
			(root + '/' + 'Train_SurgicalMask', 1),
			(root + '/' + 'Train_ClothMask', 2),
			(root + '/' + 'Train_N95Mask', 3)
		]
datasetTraining = MasksDataSet(dirsTraining, transform = transform)

# tuple of (path, label)
dirsTesting = [
			(root + '/' + 'Test_NoMask',0), 
			(root + '/' + 'Test_SurgicalMask', 1),
			(root + '/' + 'Test_ClothMask', 2),
			(root + '/' + 'Test_N95Mask', 3)
		]
datasetTest = MasksDataSet( dirsTesting, transform = transform)
#train_data, val_data = random_split(dataset, [1200, 400])

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

net.fit(datasetTraining, y=y_train)

print("Done fitting data")


y_pred = net.predict(datasetTest)
y_test = np.array([y for x, y in iter(datasetTest)])
plot_confusion_matrix(net, datasetTest, y_test.reshape(-1, 1))

print("Accuracy:    ", accuracy_score(y_test, y_pred)*100)
print("Recall:      ", recall_score(y_true=y_test, y_pred=y_pred, average='weighted')*100)
print("Precision:   ", precision_score(y_true=y_test, y_pred=y_pred, average='weighted')*100)
print("F1_Score:    ", f1_score(y_true=y_test, y_pred=y_pred, average='weighted')*100)

torch.save(model.state_dict(), "C:/Users/lemai/comp472")
print("program done")
plt.show()