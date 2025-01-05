import pandas as pd
from prettytable import PrettyTable
from configs import get_cfg_defaults
from dataloader import SiteDataset
from utools.comm_utils import set_seed, ReduceLROnPlateau, EvaMetric
from networks.models import DeepInterAwareSite
import torch
from tqdm import tqdm
from munch import Munch
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import KFold

import os
from torch.utils.data import DataLoader
import argparse
project_dir = os.path.dirname(os.path.abspath(__file__))

class SiteTrainer(object):
    def __init__(self, trainer_parameter):
        self.seed = trainer_parameter.seed
        self.lr = trainer_parameter.lr

        self.model = None
        self.optim = None
        self.scheduler = None

        self.data_path = trainer_parameter.data_path
        self.train_dataloader = None
        self.test_dataloader = None
        self.result_path = os.path.join(trainer_parameter.result_path, 'DeepinterAware')
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        self.end_epoch = trainer_parameter.end_epoch
        self.epochs = trainer_parameter.epochs
        self.device = trainer_parameter.device
        self.alpha = trainer_parameter.alpha
        self.ab_metrics = EvaMetric(task='cls', device=self.device)
        self.ag_metrics = EvaMetric(task='cls', device=self.device)
        self.batch_size = trainer_parameter.batch_size
        self.stage = 1
        # self.fold = 1
        self.test_table = None
        self.train_table = None
        # fold_header = ['Fold']+header
        # self.fold_table = PrettyTable(fold_header)



    def train_epoch(self,epoch):
        float2str = lambda x: '%0.4f' % x
        num_batches = len(self.train_dataloader)
        pbar = tqdm(enumerate(self.train_dataloader), total=num_batches)

        self.model.train()
        loss_epoch = 0
        for i, (batch, key_id) in pbar:
            batch = Munch({k: v.to(self.device)
                               for k, v in batch.items()})

            if epoch >= self.end_epoch:
                self.stage = 2
            inputs = Munch(
                batch=batch,
                device=self.device,
                stage=self.stage
            )
            self.optim.zero_grad()
            output = self.model(inputs)

            iil_loss = output.iil_loss
            sil_loss = output.sil_loss

            if self.stage == 1:
                loss = iil_loss + sil_loss
                loss.backward()
                loss_info = f'iil_loss {float2str(iil_loss.item())} sil_loss {float2str(sil_loss.item())} '
                loss = iil_loss + sil_loss
            else:
                loss = (iil_loss + sil_loss) * self.alpha + output.jcl_loss * (1 - self.alpha)
                # loss = output.jcl_loss
                loss.backward()
                loss_info = f'loss {float2str(loss.item())} iil_loss {float2str(iil_loss.item())} sil_loss {float2str(sil_loss.item())} '

            self.optim.step()
            if self.stage != 1:
                ab_n = F.softmax(output.ab_score, dim=1)[:, 1]
                ag_n = F.softmax(output.ag_score, dim=1)[:, 1]
                # y_ag_label = y_ag_label + output.ag_label.to("cpu").tolist()
                # y_ab_label = y_ab_label + output.ab_label.to("cpu").tolist()

                self.ab_metrics.update(ab_n, output.ab_label.long())
                self.ag_metrics.update(ag_n, output.ag_label.long())

            loss_epoch += loss.item()
            lr = self.optim.state_dict()['param_groups'][0]['lr']
            pbar.set_description(f"{loss_info} lr {lr} ")
            # else:
            #     pbar.set_description(f"train loss {loss.item()} lr {lr} class_weight {self.class_weight}")
            # self.scheduler.step(acc,self.current_epoch)

        if self.stage != 1:
            ag_res = self.ag_metrics.get_metric()
            ab_res = self.ab_metrics.get_metric()
        else:
            ag_res,ab_res = None,None
        loss_epoch = loss_epoch / num_batches

        self.ab_metrics.reset()
        self.ag_metrics.reset()
        epoch_output = Munch(
            loss=loss_epoch,
            ag_res=ag_res,
            ab_res=ab_res
        )
        return epoch_output

    def train(self):
        float2str = lambda x: '%0.4f' % x
        kf = KFold(n_splits=5, shuffle=True)
        # for fold in range(10):
        map_index = pd.read_csv(f'{self.data_path}/map_index.csv')
        index = range(map_index.shape[0])
        fold = 1
        for train_index, test_index in kf.split(index):
            # for train_index, test_index in kf.split(index):
            header = ["Epoch", "Ag ACC", "Ag AUPRC", "Ag ROC_AUC", "Ag F1_Score", 'Ag MCC', 'Ag Precision', 'Ag Recall',
                      "Ab ACC", "Ab AUPRC", "Ab ROC_AUC", "Ab F1_Score", 'Ab MCC', 'Ab Precision', 'Ab Recall']
            self.test_table = PrettyTable(header)
            self.train_table = PrettyTable(header)

            if os.path.exists(f'{self.data_path}/train_index_fold_{fold}.npy'):
                print("load save dataset")
                train_index = np.load(f'{self.data_path}/train_index_fold_{fold}.npy')
                test_index = np.load(f'{self.data_path}/val_index_fold_{fold}.npy')
            else:
                np.save(f'{self.data_path}/train_index_fold_{fold}', train_index)
                np.save(f'{self.data_path}/val_index_fold_{fold}', test_index)

            cfg = get_cfg_defaults()
            cfg.merge_from_file(f'{os.getcwd()}/configs/SAbDab.yml')
            self.model = DeepInterAwareSite(cfg)
            self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            self.scheduler = ReduceLROnPlateau(self.optim, mode='max', patience=3)
            best_auprc = 0
            self.model.to(self.device)
            print(f"train on the {fold} fold")
            train_dataset = SiteDataset(self.data_path,train_index)
            test_dataset = SiteDataset(self.data_path,test_index)
            params = {'batch_size': self.batch_size, 'shuffle': True, 'drop_last': False}
            self.train_dataloader = DataLoader(train_dataset, **params)
            self.test_dataloader = DataLoader(test_dataset, **params)
            self.stage = 1
            for epoch in range(1,1+self.epochs):
                epoch_output = self.train_epoch(epoch)
                train_loss, ag_res,ab_res = \
                    epoch_output.loss, epoch_output.ag_res, epoch_output.ab_res

                if ag_res!=None:
                    ag_acc, ag_auprc, ag_roc_auc, ag_f1_s, ag_mcc, ag_precision, ag_recall = \
                        ag_res.acc, ag_res.auprc, ag_res.roc_auc, ag_res.f1_s, ag_res.mcc, ag_res.precision, ag_res.recall
                    ab_acc, ab_auprc, ab_roc_auc, ab_f1_s, ab_mcc, ab_precision, ab_recall = \
                        ab_res.acc, ab_res.auprc, ab_res.roc_auc, ab_res.f1_s, ab_res.mcc, ab_res.precision, ab_res.recall

                    metric_list = ["epoch " + str(epoch)] + list(
                        map(float2str, [ag_acc, ag_auprc, ag_roc_auc, ag_f1_s, ag_mcc, ag_precision, ag_recall,
                                        ab_acc, ab_auprc, ab_f1_s, ab_mcc, ab_precision, ab_recall, ab_roc_auc]))
                    self.train_table.add_row(metric_list)
                test_ag_res,test_ab_res = self.print_res(epoch)
                if test_ag_res != None:
                    metrics = test_ag_res['auprc']
                    # self.scheduler.step(metrics, epoch)

                    if metrics>best_auprc:
                        best_auprc = metrics

                        torch.save(self.model.state_dict(), os.path.join(self.result_path, f"best_model_{fold}.pth"))
                        test_prettytable_file = os.path.join(self.result_path, f"test_markdowntable_{fold}.csv")
                        train_prettytable_file = os.path.join(self.result_path, f"train_markdowntable_{fold}.csv")

                        with open(test_prettytable_file, "w") as fp:
                            fp.write(self.test_table.get_csv_string())

                        with open(train_prettytable_file, "w") as fp:
                            fp.write(self.train_table.get_csv_string())
            fold+=1

    def test(self):
        num_batches = len(self.test_dataloader)
        pbar = tqdm(enumerate(self.test_dataloader), total=num_batches)
        iil_ab_metric = EvaMetric(task='cls', device=self.device)
        iil_ag_metric = EvaMetric(task='cls', device=self.device)

        sil_ab_metric = EvaMetric(task='cls', device=self.device)
        sil_ag_metric = EvaMetric(task='cls', device=self.device)
        loss_epoch = 0
        res_dict = {}
        with torch.no_grad():
            self.model.eval()
            for i, (batch, key_id) in pbar:
                batch = Munch({k: v.to(self.device)
                               for k, v in batch.items()})

                inputs = Munch(
                    batch=batch,
                    device=self.device,
                    # is_mixup=False,
                    mode='test',
                    stage=self.stage
                )

                output = self.model(inputs)

                iil_ab_pred = output.iil_ab_pred
                iil_ag_pred = output.iil_ag_pred
                sil_ab_pred = output.sil_ab_pred
                sil_ag_pred = output.sil_ag_pred

                iil_ab_n = F.softmax(iil_ab_pred, dim=1)[:, 1]
                iil_ag_n = F.softmax(iil_ag_pred, dim=1)[:, 1]

                sil_ab_n = F.softmax(sil_ab_pred, dim=1)[:, 1]
                sil_ag_n = F.softmax(sil_ag_pred, dim=1)[:, 1]

                iil_ab_metric.update(iil_ab_n, output.ab_label.long())
                iil_ag_metric.update(iil_ag_n, output.ag_label.long())

                sil_ab_metric.update(sil_ab_n, output.ab_label.long())
                sil_ag_metric.update(sil_ag_n, output.ag_label.long())

                if self.stage != 1:
                    ab_pred = output.ab_score
                    ag_pred = output.ag_score
                    ab_n = F.softmax(ab_pred, dim=1)[:, 1]
                    ag_n = F.softmax(ag_pred, dim=1)[:, 1]

                    self.ag_metrics.update(ag_n, output.ag_label.long())
                    self.ab_metrics.update(ab_n, output.ab_label.long())


            if self.stage != 1:
                ag_res = self.ag_metrics.get_metric()
                ab_res = self.ab_metrics.get_metric()
                self.ag_metrics.reset()
                self.ab_metrics.reset()
                # draw_calibration_curve(y_label, y_pred, iil_y_pred, sil_y_pred, self.save_file_path,
                #                        self.current_epoch)
            else:
                ag_res,ab_res = None,None

            res_dict['IIL_Antigen'] = iil_ag_metric.get_metric()
            res_dict['IIL_Antibody'] = iil_ab_metric.get_metric()

            res_dict['SIL_Antigen'] = sil_ag_metric.get_metric()
            res_dict['SIL_Antibody'] = sil_ab_metric.get_metric()

            iil_ag_metric.reset()
            iil_ab_metric.reset()

            sil_ag_metric.reset()
            sil_ab_metric.reset()

            result = Munch(
                res_dict=res_dict,
                ag_res=ag_res,
                ab_res=ab_res,
            )
        return result

    def print_res(self, epoch):
        float2str = lambda x: '%0.4f' % x

        results = self.test()
        # bcn_q_features, y_label, y_pred, loss=results.bcn_q_features, results.y_label, results.y_pred, results.loss
        ag_res,ab_res,res_dict= results.ag_res, results.ab_res, results.res_dict

        if ag_res != None:
            ag_acc, ag_auprc, ag_roc_auc, ag_f1_s, ag_mcc, ag_precision, ag_recall = \
                ag_res.acc, ag_res.auprc, ag_res.roc_auc, ag_res.f1_s, ag_res.mcc, ag_res.precision, ag_res.recall
            ab_acc, ab_auprc, ab_roc_auc, ab_f1_s, ab_mcc, ab_precision, ab_recall = \
                ab_res.acc, ab_res.auprc, ab_res.roc_auc, ab_res.f1_s, ab_res.mcc, ab_res.precision, ab_res.recall

            metric_list = ["epoch " + str(epoch)] + list(
                map(float2str, [ag_acc, ag_auprc, ag_roc_auc, ag_f1_s, ag_mcc, ag_precision, ag_recall,
            ab_acc,ab_auprc,ab_roc_auc,ab_f1_s, ab_mcc, ab_precision, ab_recall]))
            self.test_table.add_row(metric_list)

            print(f'Epoch {epoch} Antigen',
                  " AUROC "+ str(ag_roc_auc) + " AUPRC " + str(ag_auprc) + " MCC " + str(ag_mcc) + " F1 " +
                  str(ag_f1_s) + " Accuracy " + str(ag_acc) + " Precision " + str(ag_precision) + " Recall " + str(ag_recall))
            print(f'Epoch {epoch} Antibody',
                  " AUROC " + str(ab_roc_auc) + " AUPRC " + str(ab_auprc) + " MCC " + str(ab_mcc) + " F1 " +
                  str(ab_f1_s) + " Accuracy " + str(ab_acc) + " Precision " + str(ab_precision) + " Recall " + str(
                      ab_recall))

        for model_name, block_res in res_dict.items():
            block_acc, block_auprc, block_f1_s, block_mcc, block_precision, block_recall, block_roc_auc = block_res.acc, block_res.auprc, block_res.f1_s, block_res.mcc, block_res.precision, block_res.recall, block_res.roc_auc
            print(f'{model_name} at Epoch ' + str(epoch)," AUROC "+ str(block_roc_auc) + " AUPRC " + str(block_auprc) + " MCC " + str(block_mcc) + " F1 " +
                  str(block_f1_s) + " Accuracy " + str(block_acc) + " Precision " + str(
                      block_precision) + " Recall " + str(block_recall))

        return ag_res,ab_res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="antibody-antigen binding affinity prediction")
    # setting
    parser.add_argument('--data_path', default=f'{project_dir}/data/SAbDab/', type=str, metavar='TASK',
                        help='data path')
    parser.add_argument('--result_path', default=f'{project_dir}/result/train_site/', type=str, metavar='TASK',
                        help='data path')
    parser.add_argument('--gpu', default=0, type=int, metavar='S', help='run GPU number')
    parser.add_argument('--batch_size', default=32, type=int, metavar='S', help='run GPU number')
    parser.add_argument('--seed', default=0, type=int, metavar='S', help='run GPU number')
    parser.add_argument('--mode', default='test', type=str, metavar='S', help='run GPU number')
    parser.add_argument('--lr', default=1e-3, type=float, metavar='S', help='run GPU number')
    parser.add_argument('--epoch', default=300, type=int, metavar='S', help='run GPU number')
    parser.add_argument('--end_epoch', default=150, type=int, metavar='S', help='run GPU number')
    parser.add_argument('--alpha', default=0.4, type=float, metavar='S', help='run GPU number')

    args = parser.parse_args()
    # model = load_model(model_name='DeepInterAware_AM', yml=f'./configs/SAbDab.yml',
    #                    model_path=f'./save_models/SAbDab_am.pth', gpu=args.gpu)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    trainer_parameter = Munch(
        lr=args.lr,
        seed=args.seed,
        gpu = args.gpu,
        data_path=args.data_path,
        result_path=args.result_path,
        batch_size = args.batch_size,
        epochs = args.epoch,
        end_epoch = args.end_epoch,
        alpha = args.alpha,
        device = device
    )

    SiteTrainer(trainer_parameter).train()

