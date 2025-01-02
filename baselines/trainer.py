import json
from prettytable import PrettyTable
import os
from tqdm import tqdm
import torch
from munch import Munch
import numpy as np
import sys
# 获取 baseline 目录的上级目录（即 project 目录）
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 将 project 目录添加到 sys.path
sys.path.append(project_dir)
from utools.comm_utils import EvaMetric, get_map_index_for_sub_arr
class ClsBaseTrainer(object):
    def __init__(self, trainer_parameter):
        # self.start_epoch = trainer_parameter.start_epoch
        self.model = trainer_parameter.model
        self.save_best = trainer_parameter.save_best
        self.metric_type = trainer_parameter.metric_type
        self.optim = trainer_parameter.opt
        self.device = trainer_parameter.device
        self.metrics = EvaMetric(task='cls',device=self.device)
        self.epochs = trainer_parameter.train_epoch
        self.current_epoch = trainer_parameter.current_epoch
        self.train_dataloader = trainer_parameter.train_dataloader
        self.val_dataloader = trainer_parameter.val_dataloader
        self.unseen_dataloader = trainer_parameter.unseen_dataloader
        cfg=trainer_parameter.cfg
        self.batch_size = trainer_parameter.batch_size

        self.best_epoch = None
        self.best_acc = 0
        output_dir = trainer_parameter.output_dir
        self.seed = trainer_parameter.seed
        self.cfg = cfg  # 超参数

        self.result_dict={}

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.train_da_loss_epoch = []
        self.val_loss_epoch, self.val_roc_auc_epoch = [], []
        self.test_metrics = {}
        self.model_name = cfg.set.model_name

        self.output_dir = output_dir + '/' + cfg.set.dataset + '/' + self.model_name + '/'
        self.save_file_path = self.output_dir + f'batch_{self.batch_size}_epoch_{self.epochs}_seed_{self.seed}'

        valid_header = ["# Epoch", "ROC_AUC", "AUPRC", 'MCC', 'Accuracy', 'F1', 'Precision', 'Recall']

        unseen_test_header = ["# Epoch", "ROC_AUC", "AUPRC", 'MCC', 'Accuracy', 'F1', 'Precision', 'Recall']
        train_header = ["# Epoch", "ROC_AUC", "AUPRC", 'MCC', 'Accuracy', 'F1', 'Precision', 'Recall']

        self.val_table = PrettyTable(valid_header)
        # self.test_table = PrettyTable(test_header)
        self.train_table = PrettyTable(train_header)
        self.unseen_test_table = PrettyTable(unseen_test_header)

    def test(self, dataloader="test"):
        """
        [test,unseen_test] load best model
        [val,test_val,unseen_test_val] load current model
        :param dataloader: test,unseen_test,val,test_val,unseen_test_val
        :return:
        """
        loss_epoch=0
        y_label,y_pred,ag_cluster_id = [],[],[]
        if dataloader == "val":
            data_loader = self.val_dataloader
        elif dataloader in ["unseen_test","unseen_test_val" ]:
            data_loader = self.unseen_dataloader
        elif dataloader == "train":
            data_loader = self.train_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        num_batches = len(data_loader)
        bcn_q_features=[]
        # fetcher = InputFetcher(data_loader, self.cfg, self.device)
        if self.cfg.set.model_name == 'DeepAAI':
            # unique_ab_kmer = self.train_dataloader.dataset.unique_ab_kmer
            # unique_ag_kmer = self.train_dataloader.dataset.unique_ag_kmer
            all_unique_ab_id = data_loader.dataset.all_unique_ab_id
            all_unique_ag_id = data_loader.dataset.all_unique_ag_id
            unique_ab_id = data_loader.dataset.unique_ab_id
            unique_ag_id = data_loader.dataset.unique_ag_id
            antibody_graph_map_arr = get_map_index_for_sub_arr(
                unique_ab_id, np.arange(0, all_unique_ab_id.shape[0]))
            antigen_graph_map_arr = get_map_index_for_sub_arr(
                unique_ag_id, np.arange(0, all_unique_ag_id.shape[0]))

            unique_ab_kmer = data_loader.dataset.unique_ab_kmer
            unique_ag_kmer = data_loader.dataset.unique_ag_kmer

            unique_ab_kmer = unique_ab_kmer.to(self.device)
            unique_ag_kmer = unique_ag_kmer.to(self.device)
            antibody_graph_map_arr = torch.from_numpy(antibody_graph_map_arr).to(self.device)
            antigen_graph_map_arr = torch.from_numpy(antigen_graph_map_arr).to(self.device)
        pbar = tqdm(enumerate(data_loader), total=num_batches)

        with torch.no_grad():
            self.model.eval()
            for i, batch in pbar:
                batch = Munch({k: v.to(self.device)
                               for k, v in batch.items()})
                if self.cfg.set.model_name == 'DeepAAI':
                    batch.update({'unique_ab_kmer': unique_ab_kmer, 'unique_ag_kmer': unique_ag_kmer})
                    batch.update({'antibody_graph_map_arr': antibody_graph_map_arr,
                                  'antigen_graph_map_arr': antigen_graph_map_arr})
                inputs = Munch(
                    batch=batch,
                    device=self.device,
                    mode='test'
                )
                output = self.model(inputs)
                n = output.score
                self.metrics.update(n,batch.label.long())
                y_label = y_label + batch.label.to("cpu").tolist()
                y_pred = y_pred + n.to("cpu").tolist()

            res=self.metrics.get_metric()
            self.metrics.reset()
            result=Munch(
                y_label=y_label,
                y_pred=y_pred,
                res=res,
            )
        return  result

    def print_res(self,print_type='val'):
        results= self.test(dataloader=print_type)
        float2str = lambda x: '%0.4f' % x
        y_label, res = results.y_label, results.res
        acc,auprc,f1_s,mcc,precision, recall,roc_auc = res.acc,res.auprc,res.f1_s,res.mcc,res.precision, res.recall,res.roc_auc

        if print_type=='test' or print_type=='unseen_test':
            metric_list = ["epoch " + str(self.best_epoch)] + list(
                map(float2str, [roc_auc, auprc, mcc, acc, f1_s,precision, recall]))
        else:
            metric_list = ["epoch " + str(self.current_epoch)] + list(
                map(float2str, [roc_auc, auprc,mcc, acc, f1_s, precision, recall]))

        if print_type=='unseen_test_val' or print_type=='unseen_test':
            title = 'Useen Test'
            self.unseen_test_table.add_row(metric_list)
        elif print_type=='train':
            title = 'train'
            self.train_table.add_row(metric_list)
        else:
            title = 'Validation'
            self.val_table.add_row(metric_list)

        if print_type not in ['test','unseen_test','train']:
            print(f'{title} at current Epoch ' + str(self.current_epoch),
                  " AUROC "
                  + str(roc_auc) + " AUPRC " + str(auprc) +" MCC "+ str(mcc)+" F1 " +
                  str(f1_s) + " Accuracy " + str(acc) + " Precision " + str(precision) + " Recall " + str(recall))
        else:
            print(f'{title} at best Epoch ' + str(self.best_epoch),
                  " AUROC "
                  + str(roc_auc) + " AUPRC " + str(auprc) + " MCC "+ str(mcc)+" F1 " +
                  str(f1_s) + " Accuracy " + str(acc) + " Precision " + str(precision) + " Recall " + str(recall))
        return res

    def save_result(self):
        val_prettytable_file = os.path.join(self.save_file_path, "val_markdowntable.csv")
        train_prettytable_file = os.path.join(self.save_file_path, "train_markdowntable.csv")
        unseen_prettytable_file = os.path.join(self.save_file_path, "unseen_markdowntable.csv")

        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_csv_string())

        with open(val_prettytable_file, "w") as fp:
            fp.write(self.val_table.get_csv_string())

        with open(unseen_prettytable_file, "w") as fp:
            fp.write(self.unseen_test_table.get_csv_string())

class Trainer(ClsBaseTrainer):
    def __init__(self, trainer_parameter):
        super(Trainer, self).__init__(trainer_parameter)

        if not os.path.exists(self.save_file_path):
            os.makedirs(self.save_file_path)
        # self.cfg.dump(open(self.save_file_path + '/cfg.yaml', "w"))
        with open(self.save_file_path+'/cfg.json', "w") as f:
            json.dump(dict(self.cfg), f, indent=4)
        with open(os.path.join(self.save_file_path, "model_architecture.txt"), "w") as wf:
            wf.write(str(self.model))

    def train(self):
        float2str = lambda x: '%0.4f' % x

        best_metrics = 0
        for i in range(1, self.epochs + 1):
            self.current_epoch += 1
            epoch_output = self.train_epoch()
            train_loss, bcn_q_features, y_label, y_pred,res  =\
                epoch_output.loss, epoch_output.feature_s, epoch_output.y_label, epoch_output.y_pred,epoch_output.res
            # roc_auc, auprc,  mcc, acc, f1_s, precision, recall = evaluate_classification(y_pred, y_label)
            acc,auprc,f1_s,mcc,precision, recall,roc_auc = res.acc,res.auprc,res.f1_s,res.mcc,res.precision, res.recall,res.roc_auc
            train_lst = ["epoch " + str(self.current_epoch)] + list(
                map(float2str,
                    [roc_auc, auprc,  mcc, acc, f1_s, precision, recall]))

            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)
            print('Train at Epoch ' + str(self.current_epoch) + ' with sup loss ' + str(train_loss))

            # acc,auprc,f1_s,mcc,precision, recall,roc_auc, loss = self.print_res('val') #验证集
            val_res = self.print_res('val') #验证集
            if self.unseen_dataloader !=None:
                us_res = self.print_res('unseen_test_val')  # unseen
            else:
                us_res = None

            metrics = val_res[self.metric_type]
            if metrics>best_metrics:
                torch.save(self.model.state_dict(), os.path.join(self.save_file_path, f"best_model.pth"))

                val_prettytable_file = os.path.join(self.save_file_path, "val_markdowntable.csv")

                train_prettytable_file = os.path.join(self.save_file_path, "train_markdowntable.csv")

                unseen_prettytable_file = os.path.join(self.save_file_path, "unseen_markdowntable.csv")

                with open(train_prettytable_file, "w") as fp:
                    fp.write(self.train_table.get_csv_string())

                with open(val_prettytable_file, "w") as fp:
                    fp.write(self.val_table.get_csv_string())

                with open(unseen_prettytable_file, "w") as fp:
                    fp.write(self.unseen_test_table.get_csv_string())

        self.save_result()

    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)
        if self.cfg.set.model_name == 'DeepAAI':
            # unique_ab_kmer = self.train_dataloader.dataset.unique_ab_kmer
            # unique_ag_kmer = self.train_dataloader.dataset.unique_ag_kmer
            all_unique_ab_id= self.train_dataloader.dataset.all_unique_ab_id
            all_unique_ag_id = self.train_dataloader.dataset.all_unique_ag_id
            unique_ab_id = self.train_dataloader.dataset.unique_ab_id
            unique_ag_id = self.train_dataloader.dataset.unique_ag_id
            antibody_graph_map_arr = get_map_index_for_sub_arr(
                unique_ab_id, np.arange(0, all_unique_ab_id.shape[0]))
            antigen_graph_map_arr = get_map_index_for_sub_arr(
                unique_ag_id, np.arange(0, all_unique_ag_id.shape[0]))

            unique_ab_kmer=self.train_dataloader.dataset.unique_ab_kmer
            unique_ag_kmer=self.train_dataloader.dataset.unique_ag_kmer

            unique_ab_kmer = unique_ab_kmer.to(self.device)
            unique_ag_kmer = unique_ag_kmer.to(self.device)
            antibody_graph_map_arr = torch.from_numpy(antibody_graph_map_arr).to(self.device)
            antigen_graph_map_arr = torch.from_numpy(antigen_graph_map_arr).to(self.device)

        f_features, y_label, y_pred = [], [], []
        pbar = tqdm(enumerate(self.train_dataloader), total=num_batches)

        for i, batch in pbar:
            batch = Munch({k: v.to(self.device)
                           for k, v in batch.items()})
            if self.cfg.set.model_name == 'DeepAAI':
                batch.update({'unique_ab_kmer':unique_ab_kmer,'unique_ag_kmer':unique_ag_kmer})
                batch.update({'antibody_graph_map_arr':antibody_graph_map_arr,'antigen_graph_map_arr':antigen_graph_map_arr})

            inputs=Munch(
                batch=batch,
                device=self.device,
                mode = 'train'
            )
            self.optim.zero_grad()
            output = self.model(inputs)
            loss = output.loss
            n= output.score
            loss.backward()
            self.optim.step()
            y_label = y_label + batch.label.to("cpu").tolist()
            self.metrics.update(n, batch.label.long())
            y_pred = y_pred + n.to("cpu").tolist()

            loss_epoch += loss.item()
            lr = self.optim.state_dict()['param_groups'][0]['lr']
            pbar.set_description(f"train loss {loss.item()} lr {lr}")

        loss_epoch = loss_epoch / num_batches
        res = self.metrics.get_metric()
        self.metrics.reset()
        epoch_output=Munch(
            loss=loss_epoch,
            feature_s=f_features,
            y_label=y_label,
            y_pred=y_pred,
            res=res
        )
        return epoch_output

