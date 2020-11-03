import matplotlib.pyplot as plt
from sklearn.metrics  import roc_curve, auc 
import matplotlib.pyplot as plt 
import torch 
from torch.autograd import Variable 
import numpy as np 
np.random.seed(1234)
rng = np.random.RandomState(1234)

def process_label(batch_y):
    label_dict_1={0:0,1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:1,10:1}
    label_dict_2={0:0,1:1,2:1,3:1,4:2,5:2,6:2,7:3,8:3,9:3,10:3}
    np_y = batch_y.numpy()
    batch_size = len(batch_y)
    batch_y_1 = torch.zeros(batch_size).type(torch.LongTensor)
    batch_y_2 = torch.zeros(batch_size).type(torch.LongTensor)
    for i,x in enumerate(np_y):
        batch_y_1[i] = label_dict_1[x]
        batch_y_2[i] = label_dict_2[x]
    return batch_y_1,batch_y_2

def get_ci_auc( y_true, y_pred): 
    
    from scipy.stats import sem
    from sklearn.metrics import roc_auc_score 
   
    n_bootstraps = 1500   
    bootstrapped_scores = []   
   
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.random_integers(0, len(y_pred) - 1, len(y_pred))
#         print("indices:",indices)
#         print(len(indices))
       
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(y_true[indices], y_pred[indices])
#         print("score:",score)
        bootstrapped_scores.append(score)   
 
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

   # 90% c.i.
#     confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
#     confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
 
   # 95% c.i.
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
   
    return confidence_lower,confidence_upper

def plot_auc_curve(test_loader, test_size, n_classes, net, name, gpu_id,process_label_ID,label_dict):
    y_test = []
    net.eval()
    print('---------get y_test--------')
    for step ,data in enumerate(test_loader):
        batch_x,batch_y = data
        if process_label_ID > 0:
            batch_y_list = process_label(batch_y)
            batch_y = batch_y_list[process_label_ID-1]
        #_,batch_y = process_label(batch_y)
        y_test.append(batch_y.numpy()[0])
    #load model
    y_pred_prob = np.zeros([test_size,n_classes])
    # test dataset
    print("-----use model predict------")

    for step ,data in enumerate(test_loader):
        #batch_x = torch.tensor(cv2.imread('heatmap7_no_onfh.jpg'))
        batch_x,batch_y=data
        if process_label_ID > 0:
            batch_y_list = process_label(batch_y)
            batch_y = batch_y_list[process_label_ID-1]
        batch_x = batch_x.type(torch.FloatTensor).cuda(gpu_id)
        batch_y = batch_y.type(torch.LongTensor).cuda(gpu_id)
        batch_x,batch_y=Variable(batch_x),Variable(batch_y)
        logit =net(batch_x)
        h_x= F.softmax(logit, dim=1).data.squeeze()
        y_pred_prob[step] = h_x.cpu().numpy()
        probs, idx = h_x.sort(0, True)
        probs = probs.cpu().numpy()
        idx = idx.cpu().numpy()
    print("-------began plot auc curve------")
    y_test = np.array(y_test)
    y_predict_proba = y_pred_prob
    plt.style.use('ggplot')
    # Compute ROC curve and ROC AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    lower = {0:0.916,1:0.842,2:0.962}
    upper = {0:0.944,1:0.878,2:0.978}
    all_y_test_i = np.array([])
    all_y_predict_proba = np.array([])
    for i in range(n_classes):
        y_test_i = y_test == i
        y_test_i = y_test_i.astype(int)
        #y_test_i = y_test[lambda x: 1 if x == i else 0]
        all_y_test_i = np.concatenate([all_y_test_i, y_test_i])
        all_y_predict_proba = np.concatenate([all_y_predict_proba, y_predict_proba[:, i]])
        fpr[i], tpr[i], _ = roc_curve(y_test_i, y_predict_proba[:, i])
        lower_bound,upper_bound = get_ci_auc(y_test_i, y_predict_proba[:, i])
        lower[i] = lower_bound
        upper[i] = upper_bound
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    # fpr["average"], tpr["average"], _ = roc_curve(all_y_test_i, all_y_predict_proba)
    # roc_auc["average"] = auc(fpr["average"], tpr["average"])


    # Plot average ROC Curve
    plt.figure(figsize=(10,10))
    # plt.plot(fpr["average"], tpr["average"],
    #          label='Average ROC curve (area = {0:0.3f})'
    #                ''.format(roc_auc["average"]),
    #          color='deeppink', linestyle=':', linewidth=2)

    #Plot each individual ROC curve
    print("roc auc:",roc_auc)
    for i in range(0,n_classes):
        plt.plot(fpr[i], tpr[i], lw=2,
                label='ROC curve of {0} (area = {1:0.3f})\n(95% CI:{2:0.3f}-{3:0.3f})'
                ''.format(label_dict[i], roc_auc[i],lower[i],upper[i]))

    plt.plot([0, 5], [0, 5], 'k--', lw=2)
    ##
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    font2 = {'family':'Times New Roman',
    'weight':'normal',
    'size':30,}
    plt.xlabel('1-Specificity',font2)
    plt.ylabel('Sensitivity',font2)
    plt.title(' Hip x roc curve')
    plt.legend(loc="lower right")
    plt.savefig(name + '.png')
    plt.show()

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val 
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
