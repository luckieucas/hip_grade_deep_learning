import os
import numpy as np 

import torch 
import torch.nn as nn
import torch.backends.cudnn as cudnn 

import torchvision
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader 
import torch.nn.functional as F 


class DenseNet121(nn.Module):
    def __init__(self, classCount=7, isTrained=True):
        super(DenseNet121, self).__init()
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)
        kernelCount = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
    
    def forward(self, x):
        x = self.densenet121(x)
        return x 
    
class DenseNet169(nn.Module):
    def __init__(self, classCount=7, isTrained=True):
        super(DenseNet169, self).__init__()
        self.densenet169 = torchvision.models.densenet169(pretrained=isTrained)
        kernelCount = self.densenet169.classifier.in_features
        self.densenet169.classifier = nn.Sequential(nn.Linear(kernelCount, classCount),nn.Sigmoid())
        
    def forward(self, x):
        x = self.densenet169(x)
        return x