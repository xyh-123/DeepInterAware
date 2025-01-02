import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy,BinaryAUROC,BinaryAveragePrecision,BinaryMatthewsCorrCoef,BinaryF1Score,BinaryPrecision,BinaryRecall
from torchmetrics import MeanAbsoluteError,MeanAbsolutePercentageError,MeanSquaredError,R2Score
from munch import Munch
import numpy as np
import torch.nn as nn
import re
import logging
import os
import random
from torch.optim.optimizer import Optimizer
import warnings
import torch.nn.functional as F
import pandas as pd
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

def init_kaiming(m, nonlinearity):
    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity=nonlinearity)
    nn.init.zeros_(m.bias)

def init_lecun(m):
    nn.init.normal_(m.weight, mean=0.0, std=torch.sqrt(torch.tensor([1.0]) / m.in_features).numpy()[0])
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

def get_map_index_for_sub_arr(sub_arr, raw_arr):
    map_arr = np.zeros(raw_arr.shape)
    map_arr.fill(-1)
    for idx in range(sub_arr.shape[0]):
        map_arr[sub_arr[idx]] = idx
    return map_arr

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

def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)

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

def return_loss(pred_output, labels):
    if len(pred_output.shape) > 1:
        n, loss = cross_entropy_logits(pred_output, labels)
        return Munch(n=n, loss=loss)
    else:
        n, loss = binary_cross_entropy(pred_output, labels)

        return Munch(n=n, loss=loss)

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

def cauclate_avg_var(path):
    data = pd.read_csv(path)
    var = data.std().to_list()
    average = data.mean().to_list()
    float2str = lambda x, y: ('&%0.3f' % x) + '±' + ('%0.3f' % y)
    result = list(
        map(float2str, [i for i in average], [i for i in var]))
    return result

# path = f'/home/u/data/xyh/project/deepinteraware/result/HIV/DeepInterAware/ag_unseen.csv'
# result = cauclate_avg_var(path)
# print(result)