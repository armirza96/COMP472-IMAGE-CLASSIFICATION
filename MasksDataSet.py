import os
import torch from torch.utils.data 
import Dataset from skimage 
import io
import random

class MasksDataSet(Dataset):
   def init (self, root, transform=None, begin, end): 
    self.root = root
    self.transform = transform
    self.files = []
    folders = os.walk(self.root) #listdir
    for subDir in folders:
        name = subDir.split(os.path.sep)[-1]
        label = 0

        if dir == 'Train_NoMask'
            label = 0
        elif dir == 'Train_SurgicalMask'
            label = 1
        elif dir == 'Train_ClothMask'
            label = 2
         elif dir = 'Train_N95Mask'
            label = 3

        arr = [begin:end]os.listdir(name) # access array from begin to end to divide data
        for item in arr:
            files.append((item, label))

        #files = files +  # i think this will over write we need to append the array every time 
    random.shuffle(files)

   def len (self):
        return len(noMask) + len(surgicalMask) + len(n95Mask) + len(clothMask)

   def __getitem (self, index):
        dir = ''
        label = files[index][1]
        if index < 299
            dir = 'Train_NoMask'
        elif index > 299 and index < 599
            dir = 'Train_SurgicalMask'
        elif index > 599 and index < 899
            dir = 'Train_ClothMask'
         elif index > 899 and index < 1199
            dir = 'Train_N95Mask'

       path = os.path.join(self.root + '/' + dir, files[index][0])
       image = io.imread(path)
       label = torch.tensor(label)

       if self.transform:
           image = self.transform(image)

    return image, label
