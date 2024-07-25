import copy
import os
import random
import re
import warnings
import numpy as np
import torch
import torch.nn as nn
# import dgl
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch.nn.functional as F

from scipy import stats

from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy,BinaryAUROC,BinaryAveragePrecision,BinaryMatthewsCorrCoef,BinaryF1Score,BinaryPrecision,BinaryRecall
from torchmetrics import MeanAbsoluteError,MeanAbsolutePercentageError,MeanSquaredError,R2Score
from munch import Munch

class ReduceLROnPlateau(object):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        factor: factor by which the learning rate will
            be reduced. new_lr = lr * factor
        patience: number of epochs with no improvement
            after which learning rate will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {min, max}. In `min` mode,
            lr will be reduced when the quantity
            monitored has stopped decreasing; in `max`
            mode it will be reduced when the quantity
            monitored has stopped increasing.
        epsilon: threshold for measuring the new optimum,
            to only focus on significant changes.
        cooldown: number of epochs to wait before resuming
            normal operation after lr has been reduced.
        min_lr: lower bound on the learning rate.


    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = ReduceLROnPlateau(optimizer, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_acc, val_loss = validate(...)
        >>>     scheduler.step(val_loss, epoch)
    """

    def __init__(self, optimizer, mode='min', factor=0.9, patience=10,
                 verbose=0, epsilon=1e-4, cooldown=0, min_lr=0):
        super(ReduceLROnPlateau, self).__init__()

        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau '
                             'does not support a factor >= 1.0.')
        self.factor = factor
        self.min_lr = min_lr
        self.epsilon = epsilon
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.monitor_op = None
        self.wait = 0
        self.best = 0
        self.mode = mode
        assert isinstance(optimizer, Optimizer)
        self.optimizer = optimizer
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['min', 'max']:
            raise RuntimeError('Learning Rate Plateau Reducing mode %s is unknown!')
        if self.mode == 'min':
            self.monitor_op = lambda a, b: np.less(a, b - self.epsilon)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.epsilon)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0
        self.lr_epsilon = self.min_lr * 1e-4

    def reset(self):
        self._reset()

    def step(self, metrics, epoch):
        current = metrics
        if current is None:
            warnings.warn('Learning Rate Plateau Reducing requires metrics available!', RuntimeWarning)
        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                if self.wait >= self.patience:
                    for param_group in self.optimizer.param_groups:
                        old_lr = float(param_group['lr'])
                        if old_lr > self.min_lr + self.lr_epsilon:
                            new_lr = old_lr * self.factor
                            new_lr = max(new_lr, self.min_lr)
                            param_group['lr'] = new_lr
                            if self.verbose > 0:
                                print('\nEpoch %05d: reducing learning rate to %s.' % (epoch, new_lr))
                            self.cooldown_counter = self.cooldown
                            self.wait = 0
                self.wait += 1

    def in_cooldown(self):
        return self.cooldown_counter > 0


def norm_func(feature):
    feature_mean = torch.mean(feature, dim=-1, keepdim=True)
    feature_std = torch.std(feature, dim=-1, keepdim=True)
    feature_norm = (feature - feature_mean) / feature_std
    feature_norm = torch.sigmoid(feature_norm)  # bsz*271
    return feature_norm

class TopKModel(object):
    def __init__(self,k):
        self.k=k
        self.acc_list=[]
    def update(self,acc):

        if len(self.acc_list)<self.k:
            self.acc_list.append(acc)
            arr = np.array(self.acc_list)
            sort_index = np.argsort(arr)
            index=len(self.acc_list)-1
            return np.where(sort_index == index)[0].item()
        else:
            arr = np.array(self.acc_list)
            sort_index = np.argsort(arr)
            min_acc=arr[sort_index[0]]
            replace_index=sort_index[0]
            if  min_acc < acc:
                arr[replace_index] = acc
                self.acc_list[replace_index]=acc
                #重新排序
                sort_index = np.argsort(arr)
                return np.where(sort_index == replace_index)[0].item()
            else:
                return -1

class EvaMetric(object):
    def __init__(self, task,device):
        self.task=task
        self.device=device
        if task=='cls':
            #roc_auc, auprc, mcc, acc, f1_s,precision, recall
            self.all_metrics = MetricCollection({
                'roc_auc': BinaryAUROC().to(device),
                'auprc': BinaryAveragePrecision().to(device),
                'mcc': BinaryMatthewsCorrCoef().to(device),
                'acc': BinaryAccuracy().to(device),
                'f1_s': BinaryF1Score().to(device),
                'precision': BinaryPrecision().to(device),
                'recall': BinaryRecall().to(device),
            })
        else:
            #mse, mae, mape, r2
            self.all_metrics = MetricCollection({
                'mse': MeanSquaredError().to(device),
                'mae': MeanAbsoluteError().to(device),
                'mape': MeanAbsolutePercentageError().to(device),
                'r2': R2Score().to(device),
            })

    def update(self,logits,y):
        self.all_metrics.update(logits,y)
    # def compute(self,):
    #     result=self.metrics.compute()

    def reset(self):
        self.all_metrics.reset()

    def get_metric(self):
        result = self.all_metrics.compute()
        for key,value in result.items():
            result[key]=value.item()

        # print(result)
        # print(tuple(result.values()))
        return Munch(result)

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def cross_entropy_logits(linear_output, label, weights=None):
    """
    :param linear_output: batch*num_class
    :param label:batch
    :param weights:#取1的概率值
    :return:
    """
    class_output = F.log_softmax(linear_output, dim=1)  # 对行进行归一化,然后取softmax
    n = F.softmax(linear_output, dim=1)[:, 1]  # 取1的概率值
    max_class = class_output.max(1)  # (每一行的最大值,最大值对应的index)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss

def binary_cross_entropy(pred_output, labels):
    """
    :param pred_output: batch*1,linear_output
    :param labels: batch
    :return: n:对应标签的预测值0-1之间
    """
    loss_fct = torch.nn.BCELoss()
    if len(pred_output.shape)>1:
        # m = nn.Sigmoid()
        n = torch.squeeze(pred_output, -1)  # m(pred_output):batch*1 压缩维度之后为batch*1
    else:
        n=pred_output
    loss = loss_fct(n, labels.float())  # n必须是batch,labels:batch
    return n, loss

def return_loss(pred_output, labels):
    if len(pred_output.shape) > 1:
        n, loss = cross_entropy_logits(pred_output, labels)
        return Munch(n=n, loss=loss)
    else:
        n, loss = binary_cross_entropy(pred_output, labels)

        return Munch(n=n, loss=loss)

amino_map_idx = {
    "A": 1,
    "R": 2,
    "N": 3,
    "D": 4,
    "C": 5,
    "Q": 6,
    "E": 7,
    "G": 8,
    "H": 9,
    "I": 10,
    "L": 11,
    "K": 12,
    "M": 13,
    "F": 14,
    "P": 15,
    "S": 16,
    "T": 17,
    "W": 18,
    "Y": 19,
    "V": 20,
    "*": 21,
    "X": 21,
}

def amino_seq_to_one_hot(protein_seq_list,max_len):
    max_sql_len = max_len
    ft_mat = []
    for protein_seq in protein_seq_list:
        protein_seq = protein_seq.replace(' ', '')
        protein_seq = re.sub(r"[UZOBJ?*X]", "", protein_seq)
        amino_ft=integer_label_protein(protein_seq, max_sql_len)

        ft_mat.append(amino_ft)
    ft_mat=np.stack(ft_mat)
    # ft_mat = np.array(ft_mat).astype(np.float32)
    return ft_mat


def get_pos_in_raw_arr(sub_arr, raw_arr):
    '''
    :param sub_arr:  np array shape = [m]
    :param raw_arr:  np array shape = [n] 无重复数字
    :return: np array  shape = [m]
    '''
    raw_pos_dict = {}
    for i in range(raw_arr.shape[0]):
        raw_pos_dict[raw_arr[i]] = i
    trans_pos_arr = []
    for num in sub_arr:
        trans_pos_arr.append(raw_pos_dict[num])

    return np.array(trans_pos_arr, dtype=np.long)


def get_map_index_for_sub_arr(sub_arr, raw_arr):
    map_arr = np.zeros(raw_arr.shape)
    map_arr.fill(-1)
    for idx in range(sub_arr.shape[0]):
        map_arr[sub_arr[idx]] = idx
    return map_arr

def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# def graph_collate_func(x):
#     d, p, y = zip(*x)
#     d = dgl.batch(d)
#     return d, torch.tensor(np.array(p)), torch.tensor(y)


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)


def integer_label_protein(sequence, max_length=1200):
    """
    Integer encoding for protein string sequence.
    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
    """
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = amino_map_idx[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
            )
    return encoding


def init_lecun(m):
    nn.init.normal_(m.weight, mean=0.0, std=torch.sqrt(torch.tensor([1.0]) / m.in_features).numpy()[0])
    nn.init.zeros_(m.bias)

def init_kaiming(m, nonlinearity):
    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity=nonlinearity)
    nn.init.zeros_(m.bias)


@torch.no_grad()
def init_weights(m, activation_function='linear'):
    if activation_function == 'relu':
        if type(m) == nn.Linear:
            init_kaiming(m, nonlinearity='relu')
    elif activation_function == "selu":
        if type(m) == nn.Linear:
            init_lecun(m)
    elif activation_function == 'linear':
        if type(m) == nn.Linear:
            init_lecun(m)


def t_sne_draw(save_path:str,epoch:int,data: np.array, label: list,name='ls_snr',end=False,stage='pretrain'):
    stand_data = StandardScaler().fit_transform(data)
    tsne = TSNE(n_components=2)
    data_tsne = tsne.fit_transform(stand_data)

    df = pd.DataFrame(data_tsne, columns=['Dim1', 'Dim2'])
    df.insert(2, "class", label)
    df['class'] = df['class'].astype('category')
    plt.figure(dpi=100, figsize=(10, 10))
    plt.title(name)
    ax=sns.scatterplot(data=df, hue='class', x='Dim1', y='Dim2', alpha=0.4)
    l = ax.legend()
    l.get_texts()[0].set_text('NoBinding') # You can also change the legend title
    l.get_texts()[1].set_text('Binding')
    if end:
        plt.savefig(save_path + f'/{name}.jpg')
    else:
        plt.savefig(save_path+f'/{stage}_{name}_{epoch}.jpg')

if __name__ == '__main__':
    # seq='MVKVGVNGFGRIGRLVTRAAFNSGKVD'
    # physicochemical_dict=load_stand_physicochemical_ft()
    # l=[]
    # max_length=271
    # length=len(seq)
    # for i in seq:
    #     l.append(torch.tensor(physicochemical_dict[i]))
    # phy=torch.stack(l)
    # padding=torch.zeros((max_length-length,14))
    # phy=torch.cat((phy,padding))
    # print(phy.shape)

    topk=TopKModel(3)
    result=torch.tensor([0.001,0.6,0.1,0.2,0.3,0.3,0.2,0.01,0.05,0.8])
    for i in result:
        top=topk.update(i)
        print(topk.acc_list)
        print(top)

    print(topk.acc_list)


def draw_prauc(y_pred_list, y_label_list, method_name_list, figure_file):
    fig = plt.figure(figsize=(10,6))
    for y_pred,y_label, method_name in zip(y_pred_list, y_label_list, method_name_list):
        lr_precision, lr_recall, _ = precision_recall_curve(y_label, y_pred)
    #   plt.plot([0,1], [no_skill, no_skill], linestyle='--')
        plt.plot(lr_recall, lr_precision, lw = 2, label= method_name + ' (area = %0.3f)' % average_precision_score(y_label, y_pred))
        fontsize = 14

    plt.xlabel('Recall', fontsize = 20)
    plt.ylabel('Precision', fontsize = 20)
    # plt.title('Precision Recall Curve')
    plt.legend()
    plt.savefig(figure_file,dpi=300, bbox_inches='tight')


def draw_calibration_curve(y_label,y_pred,local_y_pred,global_y_pred,save_path,epoch):
    trueproba1, predproba1 = calibration_curve(y_label, y_pred,
                                               n_bins=10  # 输入希望分箱的个数
                                               )
    trueproba2, predproba2 = calibration_curve(y_label, local_y_pred,
                                               n_bins=10  # 输入希望分箱的个数
                                               )
    trueproba3, predproba3 = calibration_curve(y_label, global_y_pred,
                                               n_bins=10  # 输入希望分箱的个数
                                               )
    fig = plt.figure(figsize=(10, 6))
    ax1 = plt.subplot()
    fusion_Brier_Score = brier_score_loss(y_label, y_pred, sample_weight=None, pos_label=None)
    # fusion_log_loss = log_loss(y_label, y_pred)
    local_Brier_Score = brier_score_loss(y_label, local_y_pred, sample_weight=None, pos_label=None)
    # local_log_loss = log_loss(y_label, local_y_pred)
    global_Brier_Score = brier_score_loss(y_label, global_y_pred, sample_weight=None, pos_label=None)
    # global_log_loss = log_loss(global_y_label, global_y_pred)
    # fig, (ax1,ax2) = plt.subplots(2, 1,figsize=(10,12),sharey=True)
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax1.plot(predproba1, trueproba1, "s-", label="Fusion Feature (Brier score:%1.3f) " % (fusion_Brier_Score))
    ax1.plot(predproba2, trueproba2, "s-", label="Local Feature (Brier score:%1.3f)" % (local_Brier_Score))
    ax1.plot(predproba3, trueproba3, "s-", label="Global Feature (Brier score:%1.3f)" % (global_Brier_Score))
    ax1.set_ylabel("Fraction of Positives", fontsize=20)
    ax1.set_xlabel("Mean predcited probability", fontsize=20)
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(fontsize=15)

    plt.savefig(save_path + f'/confidence_{epoch}.svg', dpi=300, bbox_inches='tight')
    plt.show()
    y_pred_list = [y_pred,local_y_pred,global_y_pred]
    y_label_list = [y_label, y_label, y_label]
    method_name_list = ['Fusion Feature', 'Local Feature', 'Global Feature']
    figure_file = save_path + f'/auprc_{epoch}.svg'
    draw_prauc(y_pred_list, y_label_list, method_name_list, figure_file)