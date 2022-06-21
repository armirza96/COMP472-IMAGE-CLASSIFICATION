import os
import torch
from torch.utils.data import Dataset
from skimage import io
import random

class MasksDataSet(Dataset):
   def __init__(self, list, transform=None): 
    self.transform = transform
    self.items = list
    # for dir in dirs:
    #     name = dir[0]
    #     label = dir[1]
    #     print("Name of folder", dir)
    #     files = os.listdir(name)

    #     for f in files:
    #         #print(f)
    #         # append tuple of (path, label)
    #         self.items.append((dir[0] + '/' + f, label))

    #random.shuffle(self.items)
    print("Items length", len(self.items))

   def __len__(self):
        return len(self.items)

   def __getitem__ (self, index):
        label = self.items[index][1]
        path = self.items[index][0] #os.path.join(dir, self.items[index][0])
        image = io.imread(path)
        label = torch.tensor(label)

        #print("PATH:LABEL", path, label)

        if self.transform:
            image = self.transform(image)

        return image, label
