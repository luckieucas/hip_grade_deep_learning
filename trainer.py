import os 
import sys 
import time 

import numpy as np 
import torch 
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms 
import torch.optim as optim 
from torch.optim.lr_scheduler import ReduceLROnPlateau 
import torch.nn.functional as F 
from torch.utils.data import DataLoader 
import torch.backends.cudnn as cudnn 
from sklearn.metrics.ranking import roc_auc_score 

from DatasetGenerator import DatasetGenerator
from utils import AverageMeter
import losses

class Trainer():
    #train the classification model 
    def train(model, args, wandb, logging, loss_fn):
        """
        train model
        """
        #TODO create  model architecture

        # DataLoader
        dataLoaderTrain,dataLoaderTest = Trainer.get_dataloader(args)

        #OPTIMIZER & SCHEDULER
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.999), eps=1e-8, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode='min')

        #TODO load checkpoint

        #-------TRAIN MODEL
        for epoch in range(args.start_epoch, args.epochs):
            Trainer.epochTrain(args, wandb, model, epoch, dataLoaderTrain, optimizer, scheduler, loss_fn, logging)
            Trainer.epochTest(args, wandb, logging, model, epoch, dataLoaderTest)
            # save checkpoint model
            torch.save({'epoch':epoch+1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                        '../models/'+args.model_id+'/epoch_' + str(epoch) + '_.pth.tar')

    def epochTrain(args, wandb, model, epoch, dataloader, optimizer, scheduler, criterion, logging):
        model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss = AverageMeter()
        cls_loss_avg = AverageMeter()
        train_acc = AverageMeter()
        end = time.time()
        
        num_class = args.num_class
        correct = list(0. for i in range(num_class))
        total = list(0. for i in range(num_class))
        for step, (input, target) in  enumerate(dataloader):
            data_time.update(time.time() - end) # measure data loading time 
            input = input.type(torch.FloatTensor).cuda(args.gpu_id)
            target = target.type(torch.LongTensor).cuda(args.gpu_id)
            target_label = torch.max(target, 1)[1]
            out = model(input)
            cls_loss = criterion(out, target_label)
            if args.bnm_loss:
                bnm_loss = args.bnm_loss_weight * losses.bnm_loss(out)
            else:
                bnm_loss = 0.0
            train_loss = cls_loss + bnm_loss
            loss.update(train_loss)
            cls_loss_avg.update(cls_loss)

            # get the train acc
            pred = torch.max(out, 1)[1]
            res = pred == target_label 
            train_correct = (pred == target_label).sum()
            for label_idx in range(len(target_label)):
                label_single = target_label[label_idx]
                correct[label_single] += res[label_idx].item()
                total[label_single] += 1

            acc_str = ''
            if step%50 == 0:
                for acc_idx in range(num_class):
                    try:
                        acc = correct[acc_idx] / total[acc_idx]
                    except:
                        acc = 0.0
                    finally:
                        acc_str += ' classId%d acc:%.4f ' % (acc_idx + 1, acc)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if step % 50 == 0:
                logging.info("\ntrain class acc:{}".format(acc_str))

    def epochTest(args, wandb, logging, model, epoch, dataloader):
        model.eval()
        cudnn.benchmark = True 

        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()
        
        num_class = args.num_class
        correct = list(0. for i in range(num_class))
        total = list(0. for i in range(num_class))
        for step, (input, target) in enumerate(dataloader):
            target = target.cuda()
            outGT = torch.cat((outGT, target), 0)
            bs, n_crops, c, h, w = input.size()
            with torch.no_grad():
                out = model(input.view(-1, c, h, w).cuda())
            outMean = out.view(bs, n_crops, -1).mean(1)
            outPRED = torch.cat((outPRED, outMean.data), 0)
        # calculate acc
        pred = torch.max(outPRED, 1)[1]
        target_label = torch.max(outGT, 1)[1]
        res = pred == target_label 
        train_correct = (pred == target_label).sum()
        acc_str = ""
        avg_acc = 0.0
        for label_idx in range(len(target_label)):
            label_single = target_label[label_idx]
            correct[label_single] += res[label_idx].item()
            total[label_single] += 1
        for acc_idx in range(num_class):
            try:
                acc = correct[acc_idx] / total[acc_idx]
            except:
                acc = 0.0
            finally:
                wandb.log({'classID'+str(acc_idx+1)+' acc':acc})
                avg_acc += acc
                acc_str += ' classId%d acc:%.4f ' % (acc_idx + 1, acc)
        avg_acc = avg_acc/float(num_class)
        wandb.log({'Test_avg_acc':avg_acc})
        logging.info("\ntest class acc:{}".format(acc_str))



    def get_dataloader(args):
        data_path = args.data 
        mean_std = args.mean_std.split(",")
        mean,std = float(mean_std[0]),float(mean_std[1])
        mean = [mean, mean, mean]
        std = [std, std, std]
        normalize = transforms.Normalize(mean,std)
        resize = (args.resize, args.resize)
        train_transform = transforms.Compose(
            [
                #transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize(resize),
                transforms.RandomCrop(args.crop_size),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=10, scale=(0.8, 1.2)),
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
            ])
        #TODO add the tencrop in the test dataset
        test_transform = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.TenCrop(args.crop_size),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
            ])
        datasetTrain = DatasetGenerator(pathImageDirectory=args.root_path, pathDatasetFile=args.train_file,
                                        transform=train_transform)
        datasetTest = DatasetGenerator(pathImageDirectory=args.root_path, pathDatasetFile=args.test_file,
                                       transform=test_transform)
        dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=args.train_bs, num_workers=8, pin_memory=True, shuffle=True)
        dataloaderTest = DataLoader(dataset=datasetTest,batch_size=args.test_bs,num_workers=8, pin_memory=True)
        return dataLoaderTrain,dataloaderTest


