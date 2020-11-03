import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models,transforms
import torch.nn.functional as F
import torch.nn as nn
import pretrainedmodels
from glob import glob
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader 

from DatasetGenerator import DatasetGenerator
#setting info
label_3cls = ["Normal","ONFH I","ONFH II"]

#load trained model 
modelCheckpoint = torch.load("../models/xception_fold1/epoch_2_.pth.tar") # 3cls best model
model = pretrainedmodels.__dict__['xception'](num_classes=1000,
                                            pretrained='imagenet')
num_fc = model.last_linear.in_features
model.last_linear = nn.Linear(num_fc, 3)
model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(modelCheckpoint['state_dict'])

#prepare dataset
test_data_path = '../data/onfh_3cls/training_data/'
test_file = '../data/onfh_3cls/testing_fold1.txt'
mean = 0.605
std = 0.156
mean = [mean, mean, mean]
std = [std, std, std]
normalize = transforms.Normalize(mean,std)
test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.TenCrop(224),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
            ])

datasetTest = DatasetGenerator(pathImageDirectory=test_data_path, pathDatasetFile=test_file,
                                transform=test_transform)

dataloaderTest = DataLoader(dataset=datasetTest,batch_size=1,num_workers=8, pin_memory=True)

#test_data
model.eval()
cudnn.benchmark = True

for step, (input, target) in enumerate(dataloaderTest):
    target = target.cuda()
    bs, n_crops, c, h, w = input.size()
    with torch.no_grad():
        out = model(input.view(-1, c, h, w).cuda())
    outMean = out.view(bs, n_crops, -1).mean(1)
    pred = torch.max(outMean, 1)[1].item()
    gt = torch.max(target,1)[1].item()
    print("pred:",label_3cls[pred],'gt:',label_3cls[gt])