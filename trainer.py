import sys
import torch
from torch import nn
from torch.autograd import Variable
from utils import process_label
#from img_aug import img_aug
import losses
import logging
import time


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_decay_rate = args.lr_decay
    lr_decay_step = args.lr_decay_step
    lr = args.lr * (lr_decay_rate ** (epoch // lr_decay_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
def train(train_loader, val_loader, model_id, model, criterion, optimizer, epochs, wandb, args):
    model.train()
    best_acc = 0.0
    num_class = args.num_class
    batch_size = args.batch_size
    val_size = args.val_size
    log_name = '../train_log/' + model_id + '.txt'
    
    #config logging
    #logging.basicConfig(filename=log_name,level=logging.INFO,format='%(message)s')
    logging.basicConfig(filename=log_name, level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


    iteration = 0
    loss_list = []
    train_accuracy_list = []
    test_accuracy_list = []
    for epoch in range(epochs):
        train_loss = 0.
        train_acc = 0.
        correct = list(0. for i in range(num_class))
        total = list(0. for i in range(num_class))
        if args.lr_decay < 1.0:
            adjust_learning_rate(optimizer, epoch, args)
        for step, data in enumerate(train_loader):
            iteration += 1
            batch_x, batch_y = data
            if args.process_label > 0:
                batch_y_list = process_label(batch_y)
                batch_y = batch_y_list[args.process_label-1]
            # if args.img_aug_batch:
            #     batch_x, batch_y = img_aug(batch_x, batch_y)
            batch_x = batch_x.type(torch.FloatTensor).cuda(args.gpu_id)
            batch_y = batch_y.type(torch.LongTensor).cuda(args.gpu_id)
            out = model(batch_x)
            cls_loss = criterion(out, batch_y)
            if args.bnm_loss:
                bnm_loss = args.bnm_loss_weight * losses.bnm_loss(out)
            else:
                bnm_loss = 0.0
            loss = cls_loss + bnm_loss
            #         print(loss)
            train_loss += loss
            # pred is the expect class
            # batch_y is the true label
            pred = torch.max(out, 1)[1]
            #         print("train label:",batch_y)
            #         print("train pred:",pred)
            res = pred == batch_y
            train_correct = (pred == batch_y).sum()
            for label_idx in range(len(batch_y)):
                label_single = batch_y[label_idx]
                correct[label_single] += res[label_idx].item()
                total[label_single] += 1

            acc_str = ''
            train_acc_2 =0.0
            if iteration%50 == 0:
                for acc_idx in range(num_class):
                    try:
                        acc = correct[acc_idx] / total[acc_idx]
                        #plotter.plot('acc'+str(acc_idx), 'train', 'Class'+str(acc_idx)+'Accuracy', iteration, acc)
                        train_acc_2 +=acc
                    except:
                        acc = 0
                    finally:
                        acc_str += ' classID%d acc:%.4f ' % (acc_idx + 1, acc)
                #         print("pred:",pred)
                #         print("batch_y:",batch_y)
                #         print("train_correct:",train_correct)
                train_acc_2 =train_acc_2/float(num_class)
                train_acc += train_correct.cpu().numpy()

            #         print("train_acc:",train_acc)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iteration % 50 == 0:
                logging.info("\ntrain class acc:{}".format(acc_str))
                torch.save(model, "../models/" + model_id + '_iter'+str(iteration)+'.pth')
                # get the acc of test dataset
                eval_loss = 0
                eval_acc = 0
                correct = list(0. for i in range(num_class))
                total = list(0. for i in range(num_class))
                model.eval()
                for step_test, data in enumerate(val_loader):
                    batch_x, batch_y = data
                    if args.process_label > 0:
                        batch_y_list = process_label(batch_y)
                        batch_y = batch_y_list[args.process_label-1]
                    batch_x = batch_x.type(torch.FloatTensor).cuda(args.gpu_id)
                    batch_y = batch_y.type(torch.LongTensor).cuda(args.gpu_id)
                    batch_x, batch_y = Variable(batch_x), Variable(batch_y)
                    out = model(batch_x)

                    # pred is the expect class
                    # batch_y is the true label
                    pred = torch.max(out, 1)[1]
                    # print("out:",out)
                    #                 print("label:",batch_y)
                    #                 print("pred:",pred)
                    test_correct = (pred == batch_y).sum()
                    eval_acc += test_correct.cpu().numpy()
                    res = pred == batch_y
                    # get the acc by class
                    for label_idx in range(len(batch_y)):
                        label_single = batch_y[label_idx]
                        correct[label_single] += res[label_idx].item()
                        total[label_single] += 1
                model.train()
                test_accuracy_list.append(eval_acc / val_size)
                if eval_acc / val_size > best_acc:
                    best_acc = eval_acc / val_size
                    torch.save(model, "../models/" + model_id + '_best.pth')
                acc_str = ''
                eval_acc_2 = 0.0
                for acc_idx in range(num_class):
                    try:
                        acc = correct[acc_idx] / total[acc_idx]
                        #plotter.plot('acc'+str(acc_idx), 'val', 'Class'+str(acc_idx)+'Accuracy', iteration, acc)
                        eval_acc_2 += acc
                    except:
                        acc = 0
                    finally:
                        #test_acc_class_list[acc_idx].append(acc)
                        wandb.log({'classID'+str(acc_idx+1)+' acc':acc})
                        acc_str += ' classID%d acc:%.4f ' % (acc_idx + 1, acc)
                eval_acc_2 = eval_acc_2/float(num_class)
                train_loss_print = train_loss.cpu().detach().numpy() / ((step + 1) * batch_size)
                #plotter.plot('Train_loss', 'train', 'Train_loss', iteration, train_loss_print)
                #plotter.plot('avg_acc', 'train', 'acc', iteration, train_acc_2)
                #plotter.plot('avg_acc', 'val', 'acc', iteration, eval_acc_2)
                logging.info("\niter: {},Epoch: {},Step: {},Train loss: {:.6f},Train acc: {:.6f},eval acc:{:.6f}".format(
                   iteration,epoch,step, train_loss_print,train_acc_2,eval_acc_2))
                wandb.log({"iter":iteration,'epoch':epoch,'Train loss': train_loss_print,'Train acc':train_acc_2,'eval acc':eval_acc_2})
                logging.info("\n{}".format(acc_str))
                loss_list.append(train_loss / ((step_test + 1) * batch_size))
                train_accuracy_list.append(train_acc / ((step + 1) * batch_size))
