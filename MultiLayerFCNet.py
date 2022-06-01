from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets

num_epochs = 4
num_classes = 10
learning_rate = 0.001
transform = transforms.Compose(
[transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=1000,shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class MultiLayerFCNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(MultiLayerFCNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, D_out)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return F.log_softmax(x)

model = MultiLayerFCNet(D_in, H, D_out)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(num_epochs):
    avg_loss_epoch = 0
    batch_loss = 0
    total_batches = 0

    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 32 * 32 * 3)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_batches += 1
        batch_loss += loss.item()
        avg_loss_epoch = batch_loss / total_batches
print('Epoch [{}/{}], Averge Loss:for epoch[{}, {:.4f}]'
.format(epoch + 1, num_epochs, epoch + 1, avg_loss_epoch))

correct = 0
total = 0
for images, labels in test_loader:
    images = images.reshape(-1, 3 * 32 * 32)
    outputs_test = model(images)
    _, predicted = torch.max(outputs_test.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))