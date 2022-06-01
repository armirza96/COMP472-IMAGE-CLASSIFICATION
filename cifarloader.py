import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td

def cifar_loader(batch_size, shuffle_test=False):
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.225, 0.225, 0.225])
	train = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(32, 4), transforms.ToTensor(),normalize]))
	test = datasets.CIFAR10('./data', train=False, transform=transforms.Compose([transforms.ToTensor(), normalize]))
	train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=shuffle_test, pin_memory=True)
	return train_loader, test_loader

batch_size = 64
test_batch_size = 64
input_size = 3072
N = batch_size
D_in = input_size
H = 50
D_out = 10
num_epochs = 10
train_loader, _ = cifar_loader(batch_size)
_, test_loader = cifar_loader(test_batch_size)