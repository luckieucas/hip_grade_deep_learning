import os 
import numpy as np
import time 
import tempfile 
from PIL import Image 

import torch 
import torch.nn.functional as F 
from torch.utils.data import Dataset 
from torch.utils.data.sampler import Sampler 
import itertools 



class DatasetGenerator(Dataset):
    def __init__(self,pathImageDirectory, pathDatasetFile,  transform, class_num=7):

        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform 
        self.class_num = class_num

        #--- Open file, get iamge paths and labels 

        #--- get into the loop
        with open(pathDatasetFile,'r') as f:
            for line in f:
                lineItems = line.split()
                imagePath = os.path.join(pathImageDirectory, lineItems[0])
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]
                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)
        self.listImageLabels = torch.FloatTensor(self.listImageLabels)

    def __getitem__(self, index):
        imagePath = self.listImagePaths[index]
        image = Image.open(imagePath).convert('RGB')
        imageLabel = self.listImageLabels[index]

        if self.transform != None: image = self.transform(image)

        return image,imageLabel

    def __len__(self):
        return len(self.listImagePaths)

