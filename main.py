import sys
import argparse 
import torch 
from torchvision import transforms, utils 
import random 
from torch import nn 
from dataloader import data_loader 
from trainer import Trainer
import utils
import wandb
import torchvision
import pretrainedmodels
import logging

from models import DenseNet121,DenseNet169
from config import CLASS_NUM_DICTS

torch.manual_seed(1)
torch.random.manual_seed(1)
random.seed(232)

parser = argparse.ArgumentParser(description='Hip classification Training')
parser.add_argument('--model_id', type=str, default='resnet18_test', help=' model id')
parser.add_argument('--task', type=str, default='onfh_3cls', help=' model id')
parser.add_argument('--data', default='../data/512_2-1_train/', type=str, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--root_path', type=str, default='../data/onfh_3cls/training_data/',
                    help='dataset root dir')
parser.add_argument('--train_file', type=str, default='../data/onfh_3cls/training_fold1.txt')
parser.add_argument('--test_file', type=str, default='../data/onfh_3cls/testing_fold1.txt')
parser.add_argument('--lr_decay', type=float, default=0.5, help='lr decay rate')
parser.add_argument('--lr_decay_step', type=int, default=30, help='lr decay step')
parser.add_argument('--resize', type=int, default=256, help='image resize size')
parser.add_argument('--crop_size', type=int, default=224, help='image resize size')
parser.add_argument('--criterion', type=str, default='cel',
                    help='criterion,cel mean cross entropy loss,fcl mean focal loss')
parser.add_argument('--batch_size', type=int, default=60, help='batch_size')
parser.add_argument('--train_bs', type=int, default=64, help='train batch size')
parser.add_argument('--test_bs', type=int, default=64, help='test batch size')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
parser.add_argument('--epochs', type=int, default=200, help='start epoch')
parser.add_argument('--start_epoch', type=int, default=0, help='num of epoch')
parser.add_argument('--num_class', type=int, default=3, help='classification num class')
parser.add_argument('--class_weight', type=str, default='1,6,1.5', help='class weight')
parser.add_argument('--is_pretrained', type=bool, default=True,
                    help='is use the pretrained model')
parser.add_argument('--img_aug_batch', type=bool, default=False,
                    help='use img aug during training batch')
parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu id')
parser.add_argument('--mean_std', type=str, default='0.605,0.156')
parser.add_argument('--process_label', type=int, default=0)
parser.add_argument('--bnm_loss', type=int, default=1, help='whether use bnm loss to train model')
parser.add_argument('--bnm_loss_weight', type=float, default=1.0, help='weight of bnm_loss')


def main():
    args = parser.parse_args()
    #train_loader, test_loader, test_size = get_data(args)
    model = get_model(args)
    print("num class:", args.num_class)
    gpu_ids = [int(i) for i in args.gpu_ids.split(",")]
    gpu_id = gpu_ids[0]
    args.gpu_id = gpu_id
    print("gpu  ids:", gpu_ids)
    model_id = args.model_id 
    epoch = args.epochs
    args.class_num_dict = CLASS_NUM_DICTS[args.task]
    #wandb config 
    wandb.init(project="hip-classification", name="hip-classification_"+model_id)
    wandb.config.update(args)

    #TODO need to use the num of class to get the class weight
    class_weight = [float(i) for i in args.class_weight.split(",")]
    class_weight = torch.FloatTensor(class_weight).cuda(gpu_id)
    class_num = args.class_num_dict
    class_weight = torch.Tensor([sum(class_num)/i for i in class_num]).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model = model.cuda(gpu_id)
    #args.val_size = test_size
    model = nn.DataParallel(model, device_ids=gpu_ids)
    #train(train_loader,test_loader, model_id, model, criterion, optimizer, epoch, wandb, args)
    log_name = '../train_log/' + model_id + '.txt'
    
    #config logging
    #logging.basicConfig(filename=log_name,level=logging.INFO,format='%(message)s')
    logging.basicConfig(filename=log_name, level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    Trainer.train(model, args, wandb, logging, criterion)

def get_data(args):
    data_path = args.data 
    mean_std = args.mean_std.split(",")
    mean,std = float(mean_std[0]),float(mean_std[1])
    mean = [mean, mean, mean]
    std = [std, std, std]
    resize = (args.resize, args.resize)
    train_transform = transforms.Compose(
        [
            #transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(480),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=10, scale=(0.8, 1.2)),
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    #TODO add the tencrop in the test dataset
    test_transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    args.train_transform = train_transform
    args.test_transform = test_transform
    #TODO data_path 修改
    train_loader, test_loader, test_size = data_loader(data_path+'/train/',data_path+'/test/',train_transform,
                                                       test_transform,args)
    return train_loader, test_loader, test_size

def get_model(args):
    model = torchvision.models.resnet18(pretrained=args.is_pretrained)
    #TODO use backbone replace the model_id
    if 'resnet18' in args.model_id:
        num_fc = model.fc.in_features
        model.fc = torch.nn.Linear(num_fc, args.num_class)
    if 'xception' in args.model_id:
        model = pretrainedmodels.__dict__['xception'](num_classes=1000,
                                                      pretrained='imagenet')
        num_fc = model.last_linear.in_features
        model.last_linear = nn.Linear(num_fc, args.num_class)
    if 'efficientnetb5' in args.model_id:
        model = EfficientNet.from_pretrained('efficientnet-b5')
        num_fc = model._fc.in_features
        model._fc = torch.nn.Linear(num_fc, args.num_class)
    if 'densenet121' in args.model_id:
        print("-------------------use  densenet121-------------------")
        model = torchvision.models.densenet121(pretrained=is_pretrained)
        num_fc = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_fc, args.num_class)
    return model
    

if __name__ == '__main__':
    main()