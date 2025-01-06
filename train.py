from utools.comm_utils import TopKModel,ReduceLROnPlateau,EvaMetric,return_loss
from prettytable import PrettyTable
import torch
from tqdm import tqdm
from munch import Munch
import numpy as np
import os
import json
import torch.nn.functional as F
class ClsBaseTrainer(object):
    def __init__(self, trainer_parameter):
        # self.start_epoch = trainer_parameter.start_epoch
        self.end_epoch = trainer_parameter.end_epoch
        self.model = trainer_parameter.model
        self.save_best = trainer_parameter.save_best
        self.metric_type = trainer_parameter.metric_type
        self.optim = trainer_parameter.opt

        self.scheduler=ReduceLROnPlateau(self.optim,mode='max',patience=3)
        self.device = trainer_parameter.device
        self.metrics = EvaMetric(task='cls',device=self.device)
        self.epochs = trainer_parameter.cfg.solver.train_epoch
        self.init_epoch=0
        self.current_epoch = trainer_parameter.current_epoch
        self.train_dataloader = trainer_parameter.train_dataloader
        self.val_dataloader = trainer_parameter.val_dataloader
        self.unseen_dataloader = trainer_parameter.unseen_dataloader
        cfg=trainer_parameter.cfg
        self.batch_size = cfg.solver.batch_size

        self.best_epoch = None
        self.best_acc = 0
        self.max_epoch = cfg.solver.train_epoch
        self.seed = cfg.solver.seed
        self.cfg = cfg  # 超参数
        self.topk=TopKModel(20)
        self.result_dict={}

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.train_da_loss_epoch = []
        self.val_loss_epoch, self.val_roc_auc_epoch = [], []
        self.test_metrics = {}
        self.model_name = cfg.set.model_name

        self.output_dir = cfg.result.output_dir + '/' + cfg.set.dataset + '/' + self.model_name + '/'
        self.save_file_path = self.output_dir + f'batch_{self.batch_size}_epoch_{self.max_epoch}_seed_{self.seed}'

        valid_header = ["# Epoch", "ROC_AUC", "AUPRC", 'MCC', 'Accuracy', 'F1', 'Precision', 'Recall',
                        "Val_loss"]

        unseen_test_header = ["# Epoch", "ROC_AUC", "AUPRC", 'MCC', 'Accuracy', 'F1', 'Precision', 'Recall',
                              "Test_loss"]
        train_header = ["# Epoch", "ROC_AUC", "AUPRC", 'MCC', 'Accuracy', 'F1', 'Precision', 'Recall',
                        "Train_loss"]

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

        pbar = tqdm(enumerate(data_loader), total=num_batches)

        with torch.no_grad():
            self.model.eval()
            for i, batch in pbar:
                batch = Munch({k: v.to(self.device)
                               for k, v in batch.items()})

                inputs = Munch(
                    batch=batch,
                    device=self.device,
                    mode='test',
                )

                output = self.model(inputs)
                out_loss = return_loss(output.score, batch.label)
                n = out_loss.n
                loss = out_loss.loss

                bcn_q_features.append(output.feature.detach().cpu().numpy())
                self.metrics.update(n,batch.label.long())
                y_label = y_label + batch.label.to("cpu").tolist()
                y_pred = y_pred + n.to("cpu").tolist()
                loss_epoch += loss.item()

            res=self.metrics.get_metric()
            self.metrics.reset()
            loss_epoch = loss_epoch / num_batches
            result=Munch(
                bcn_q_features=np.concatenate(bcn_q_features),
                y_label=y_label,
                y_pred=y_pred,
                res=res,
                loss=loss_epoch,
            )
        return  result

    def print_res(self,print_type='val'):
        results= self.test(dataloader=print_type)
        float2str = lambda x: '%0.4f' % x
        bcn_q_features, y_label, res, loss=results.bcn_q_features, results.y_label, results.res, results.loss
        acc,auprc,f1_s,mcc,precision, recall,roc_auc = res.acc,res.auprc,res.f1_s,res.mcc,res.precision, res.recall,res.roc_auc

        if print_type=='test' or print_type=='unseen_test':
            metric_list = ["epoch " + str(self.best_epoch)] + list(
                map(float2str, [roc_auc, auprc, mcc, acc, f1_s,precision, recall, loss]))

        else:
            metric_list = ["epoch " + str(self.current_epoch)] + list(
                map(float2str, [roc_auc, auprc,mcc, acc, f1_s, precision, recall,loss]))

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
            print(f'{title} at current Epoch ' + str(self.current_epoch) + ' with loss ' + str(loss),
                  " AUROC "
                  + str(roc_auc) + " AUPRC " + str(auprc) +" MCC "+ str(mcc)+" F1 " +
                  str(f1_s) + " Accuracy " + str(acc) + " Precision " + str(precision) + " Recall " + str(recall))

        else:
            print(f'{title} at best Epoch ' + str(self.best_epoch) + ' with loss ' + str(loss),
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

class DeepInterAwareTrainer(ClsBaseTrainer):
    def __init__(self, trainer_parameter):
        super(DeepInterAwareTrainer, self).__init__(trainer_parameter)
        # self.start_epoch = trainer_parameter.start_epoch
        self.end_epoch = trainer_parameter.end_epoch
        self.alpha = self.cfg.set.alpha
        self.save_file_path = self.output_dir + f'batch_{self.batch_size}_epoch_{self.max_epoch}_seed_{self.seed}_{self.cfg.set.unseen_task}_{self.cfg.set.ab_model}'
        if self.cfg.predict.simple:
            self.save_file_path = self.save_file_path + '_simple'
        if self.cfg.set.phy:
            self.save_file_path = self.save_file_path + f'_phy'
        if self.cfg.protein.use_bn:
            self.save_file_path = self.save_file_path + '_bn'

        if trainer_parameter.sampler:
            self.save_file_path = self.save_file_path + '_sampler'


        if not os.path.exists(self.save_file_path):
            os.makedirs(self.save_file_path)
        else:
            print(self.save_file_path + " is empty")

        with open(self.save_file_path+'/cfg.json', "w") as f:
            json.dump(dict(self.cfg), f, indent=4)

    def train(self):
        float2str = lambda x: '%0.4f' % x

        for i in range(1, self.epochs + 1):
            self.current_epoch += 1

            epoch_output = self.train_epoch()
            train_loss, bcn_q_features, y_label, y_pred,res =\
                epoch_output.loss, epoch_output.feature_s, epoch_output.y_label, epoch_output.y_pred,epoch_output.res

            if res!=None:
                acc,auprc,f1_s,mcc,precision, recall,roc_auc = res.acc,res.auprc,res.f1_s,res.mcc,res.precision, res.recall,res.roc_auc
                train_lst = ["epoch " + str(self.current_epoch)] + list(
                    map(float2str,
                        [roc_auc, auprc,  mcc, acc, f1_s, precision, recall, train_loss]))

                self.train_table.add_row(train_lst)
                self.train_loss_epoch.append(train_loss)
                print('Train at Epoch ' + str(self.current_epoch) + ' with sup loss ' + str(train_loss))

            val_res = self.print_res('val')  # 验证集
            if self.unseen_dataloader !=None:
                us_res = self.print_res('unseen_test_val')  # unseen
            else:
                us_res = None

            if val_res!=None:
                metrics = val_res[self.metric_type]
                self.scheduler.step(metrics,self.current_epoch)
                topk=self.topk.update(metrics)

                if topk!=-1:
                    self.best_epoch = self.current_epoch
                    checkpoint = {
                        'model_state_dict': self.model.state_dict(),
                        'epoch': self.current_epoch
                    }
                    if self.save_best:
                        torch.save(checkpoint, os.path.join(self.save_file_path, f"best_model.pth"))
                    else:
                        torch.save(checkpoint, os.path.join(self.save_file_path, f"best_model_{topk}.pth"))
                    value_dict = {"epoch": self.current_epoch, f'valid {self.metric_type}': metrics,}
                    if us_res!=None:
                        value_dict[f'unseen {self.metric_type}']=us_res[self.metric_type]
                    self.result_dict[topk] = value_dict

                    result_file = os.path.join(self.save_file_path, "result.json")
                    with open(result_file, "w") as f:
                        f.write(json.dumps(self.result_dict, ensure_ascii=False, indent=4, separators=(',', ':')))

                    val_prettytable_file = os.path.join(self.save_file_path, "val_markdowntable.csv")
                    # test_prettytable_file = os.path.join(self.save_file_path, "test_markdowntable.csv")
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
        float2str = lambda x: '%0.4f' % x
        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)
        f_features, y_label, y_pred = [], [], []

        pbar = tqdm(enumerate(self.train_dataloader), total=num_batches)
        step=0
        for i, batch in pbar:

            batch = Munch({k: v.to(self.device)
                           for k, v in batch.items()})

            step += 1
            if 0 <= self.current_epoch <= self.end_epoch:
                stage = 1
            else:
                stage = 2
            inputs = Munch(
                batch=batch,
                device=self.device,
                stage=stage
            )
            self.optim.zero_grad()
            output = self.model(inputs)

            f_features.append(output.feature.detach().cpu().numpy())
            iil_loss = output.iil_loss
            sil_loss = output.sil_loss

            if inputs.stage == 1:
                loss = iil_loss + sil_loss
                loss.backward()
                loss_info = f'iil_loss {float2str(iil_loss.item())} sil_loss {float2str(sil_loss.item())} '
            else:
                # loss = (iil_loss + sil_loss) * 0.5 + output.jcl_loss * 2
                loss = (iil_loss + sil_loss) * self.alpha + output.jcl_loss * (1-self.alpha)
                loss.backward()
                loss_info = f'loss {float2str(loss.item())} iil_loss {float2str(iil_loss.item())} sil_loss {float2str(sil_loss.item())} '
            # loss.backward()
            self.optim.step()

            if stage != 1:
                f_features.append(output.feature.detach().cpu().numpy())
                n = F.softmax(output.score, dim=1)[:, 1]
                y_label = y_label + batch.label.to("cpu").tolist()
                self.metrics.update(n, batch.label.long())
                y_pred = y_pred + n.to("cpu").tolist()
            else:
                y_label = None
                y_pred = None

            loss_epoch += loss.item()
            lr = self.optim.state_dict()['param_groups'][0]['lr']

            pbar.set_description(f"{loss_info} lr {lr} ")
            if stage != 1:
                res = self.metrics.get_metric()
            else:
                y_label = None
                y_pred = None
                res = None
            loss_epoch = loss_epoch / num_batches

        self.metrics.reset()
        epoch_output = Munch(
            loss=loss_epoch,
            feature_s=f_features,
            y_label=y_label,
            y_pred=y_pred,
            res=res
        )
        return epoch_output

    def test(self, dataloader="test"):
        """
        [test,unseen_test] load best model
        [val,test_val,unseen_test_val] load current model
        :param dataloader: test,unseen_test,val,test_val,unseen_test_val
        :return:
        """
        loss_epoch=0
        y_label,y_pred,softmax_ent = [],[],[]
        iil_y_pred,sil_y_pred = [],[]
        if dataloader == "val":
            data_loader = self.val_dataloader
        elif dataloader in ["unseen_test","unseen_test_val" ]:
            data_loader = self.unseen_dataloader
        elif dataloader == "train":
            data_loader = self.train_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        num_batches = len(data_loader)
        bcn_q_features,res_dict=[],{}
        ag_cluster_id=[]
        if 0 <= self.current_epoch <= self.end_epoch:
            stage = 1
        else:
            stage = 2
        if dataloader in ["test", "unseen_test", 'train']:
            if self.save_best:
                state_dict = torch.load(os.path.join(self.save_file_path, f"best_model.pth"))
            else:
                topk = sorted(self.result_dict)[-1]
                print(f"load best_model_{topk}.pth")
                state_dict = torch.load(os.path.join(self.save_file_path, f"best_model_{topk}.pth"))

            self.model.load_state_dict(state_dict['model_state_dict'])
        pbar = tqdm(enumerate(data_loader), total=num_batches)
        iil_metric = EvaMetric(task='cls', device=self.device)
        sil_metric = EvaMetric(task='cls', device=self.device)

        with torch.no_grad():
            self.model.eval()
            for i, batch in pbar:
                batch = Munch({k: v.to(self.device)
                               for k, v in batch.items()})
                inputs = Munch(
                    batch=batch,
                    device=self.device,
                    stage=stage
                )

                output = self.model(inputs)
                iil_n = F.softmax(output.iil_pred, dim=1)[:, 1]
                sil_n = F.softmax(output.sil_pred, dim=1)[:, 1]
                iil_metric.update(iil_n, batch.label.long())
                sil_metric.update(sil_n, batch.label.long())

                if stage!=1:
                    out_loss = return_loss(output.score, batch.label)
                    n = out_loss.n
                    self.metrics.update(n, batch.label.long())
                    loss = out_loss.loss
                    bcn_q_features.append(output.feature.detach().cpu().numpy())
                    y_label = y_label + batch.label.to("cpu").tolist()
                    y_pred = y_pred + n.to("cpu").tolist()
                    iil_y_pred = iil_y_pred + iil_n.to("cpu").tolist()
                    sil_y_pred = sil_y_pred + sil_n.to("cpu").tolist()

                    if self.cfg.set.dataset == 'ppi' and (dataloader in ["unseen_test", "unseen_test_val"]):
                        ag_cluster_id = ag_cluster_id + batch.ag_cluster_id.to("cpu").tolist()
                    loss_epoch += loss.item()
                else:
                    bcn_q_features=None
            if stage != 1:
                res=self.metrics.get_metric()
                self.metrics.reset()
            else:
                res =None

            res_dict['IIL']=iil_metric.get_metric()
            res_dict['SIL']=sil_metric.get_metric()
            iil_metric.reset()
            sil_metric.reset()
            loss_epoch = loss_epoch / num_batches
            if bcn_q_features!=None:
                bcn_q_features=np.concatenate(bcn_q_features)
            result=Munch(
                bcn_q_features=bcn_q_features,
                y_label=y_label,
                res_dict=res_dict,
                res=res,
                loss=loss_epoch,
            )
        return  result

    def print_res(self,print_type='val'):
        float2str = lambda x: '%0.4f' % x
        results= self.test(dataloader=print_type)
        # bcn_q_features, y_label, y_pred, loss=results.bcn_q_features, results.y_label, results.y_pred, results.loss
        bcn_q_features, y_label, res, loss ,res_dict = results.bcn_q_features, results.y_label, results.res, results.loss,results.res_dict
        if res !=None:
            acc,auprc,f1_s,mcc,precision, recall,roc_auc = res.acc,res.auprc,res.f1_s,res.mcc,res.precision, res.recall,res.roc_auc

            if print_type=='test' or print_type=='unseen_test':
                metric_list = ["epoch " + str(self.best_epoch)] + list(
                    map(float2str, [roc_auc, auprc, mcc, acc, f1_s,precision, recall, loss]))
            elif print_type=='train':
                metric_list = ["epoch " + str(self.best_epoch)] + list(
                    map(float2str, [roc_auc, auprc, mcc, acc, f1_s, precision, recall, loss]))
            else:
                metric_list = ["epoch " + str(self.current_epoch)] + list(map(float2str, [roc_auc, auprc,mcc, acc, f1_s, precision, recall,loss]))

            if print_type=='unseen_test_val' or print_type=='unseen_test':
                title = 'Useen Test'
                self.unseen_test_table.add_row(metric_list)
            elif print_type=='train':
                title = 'train'
                self.train_table.add_row(metric_list)
            else:
                title = 'Validation'
                self.val_table.add_row(metric_list)

            if print_type not in ['test', 'unseen_test', 'train']:
                type = 'current'
            else:
                type = 'best'

            print(f'Total {title} at {type} Epoch ' + str(self.current_epoch) + ' with loss ' + str(loss),
                  " AUROC "
                  + str(roc_auc) + " AUPRC " + str(auprc) + " MCC " + str(mcc) + " F1 " +
                  str(f1_s) + " Accuracy " + str(acc) + " Precision " + str(precision) + " Recall " + str(recall))

        for model_name, block_res in res_dict.items():
            block_acc, block_auprc, block_f1_s, block_mcc, block_precision, block_recall, block_roc_auc = block_res.acc, block_res.auprc, block_res.f1_s, block_res.mcc, block_res.precision, block_res.recall, block_res.roc_auc
            print(f'{model_name} {print_type} at current Epoch ' + str(self.current_epoch) + ' with loss ' + str(loss),
                  " AUROC "
                  + str(block_roc_auc) + " AUPRC " + str(block_auprc) + " MCC " + str(block_mcc) + " F1 " +
                  str(block_f1_s) + " Accuracy " + str(block_acc) + " Precision " + str(block_precision) + " Recall " + str(block_recall))


        return res
    
class Trainer(object):
    def __init__(self, trainer_parameter):
        # self.start_epoch = trainer_parameter.start_epoch
        self.end_epoch = trainer_parameter.end_epoch
        self.model = trainer_parameter.model
        self.save_best = trainer_parameter.save_best
        # self.print_test = trainer_parameter.print_test
        self.metric_type = trainer_parameter.metric_type
        self.optim = trainer_parameter.opt
        self.scheduler = ReduceLROnPlateau(self.optim, mode='max', patience=3)

        self.device = trainer_parameter.device
        self.metrics = EvaMetric(task='cls', device=self.device)
        self.epochs = trainer_parameter.cfg.solver.train_epoch
        self.init_epoch = trainer_parameter.current_epoch
        self.class_weight = 0
        self.current_epoch = trainer_parameter.current_epoch
        self.dataloader = {"train":trainer_parameter.train_dataloader,
                           "val":trainer_parameter.val_dataloader,
                           "unseen_test":trainer_parameter.unseen_dataloader}
        # self.train_dataloader = trainer_parameter.train_dataloader
        # self.test_dataloader = trainer_parameter.unseen_dataloader
        # self.val_dataloader = trainer_parameter.val_dataloader
        self.site_test = trainer_parameter.site_test
        # self.multi_dataloader = trainer_parameter.multi_dataloader
        cfg = trainer_parameter.cfg
        self.alpha = cfg.set.alpha
        self.batch_size = trainer_parameter.batch_size
        self.step = 0
        self.sil_step = 0
        self.best_epoch = None
        self.best_acc = 0
        self.max_epoch = cfg.solver.train_epoch
        self.seed = cfg.solver.seed
        self.cfg = cfg  # 超参数
        self.topk = TopKModel(20)
        self.result_dict = {}

        self.model_name = cfg.set.model_name

        self.output_dir = cfg.result.output_dir + '/' + cfg.set.dataset + '/' + self.model_name + '/'
        self.save_file_path = self.output_dir + f'batch_{self.batch_size}_epoch_{self.max_epoch}_seed_{cfg.solver.seed}_fold_{cfg.set.ab_model}_{cfg.set.unseen_task}_alpha_{self.alpha}'
        if self.cfg.protein.use_bn:
            self.save_file_path = self.save_file_path + '_bn'

        val_header = ["# Epoch", "ROC_AUC", "AUPRC", 'MCC', 'Accuracy', 'F1', 'Precision', 'Recall',
                        "Train_loss"]

        test_header = ["# Epoch", "ROC_AUC", "AUPRC", 'MCC', 'Accuracy', 'F1', 'Precision', 'Recall',
                       "Test_loss"]

        train_header = ["# Epoch", "ROC_AUC", "AUPRC", 'MCC', 'Accuracy', 'F1', 'Precision', 'Recall',
                        "Train_loss"]


        if self.cfg.set.phy:
            self.save_file_path = self.save_file_path + '_phy'

        if self.cfg.predict.simple:
            self.save_file_path = self.save_file_path + '_simple'

        if not os.path.exists(self.save_file_path):
            os.makedirs(self.save_file_path)
        # elif os.path.getsize(self.save_file_path) != 0:
        #     # 存在但是文件夹不为空
        #     self.save_file_path = self.save_file_path + f"_{self.cfg.set.version}"
        #     os.makedirs(self.save_file_path)
        else:
            print(self.save_file_path + " is empty")

        # self.cfg.dump(open(self.save_file_path + '/cfg.yaml', "w"))

        with open(self.save_file_path+'/cfg.json', "w") as f:
            json.dump(dict(self.cfg), f, indent=4)

        self.test_table = PrettyTable(test_header)
        self.val_table = PrettyTable(val_header)
        self.train_table = PrettyTable(train_header)
        self.stage = 1

    def train(self):
        float2str = lambda x: '%0.4f' % x
        for i in range(self.init_epoch + 1, self.epochs + 1):
            self.current_epoch += 1
            epoch_output = self.train_epoch()
            train_loss, bcn_q_features, y_label, y_pred,res  =\
                epoch_output.loss, epoch_output.feature_s, epoch_output.y_label, epoch_output.y_pred,epoch_output.res
            # roc_auc, auprc,  mcc, acc, f1_s, precision, recall = evaluate_classification(y_pred, y_label)
            if res != None:
                acc,auprc,f1_s,mcc,precision, recall,roc_auc = res.acc,res.auprc,res.f1_s,res.mcc,res.precision, res.recall,res.roc_auc
                train_lst = ["epoch " + str(self.current_epoch)] + list(
                    map(float2str,
                        [roc_auc, auprc,  mcc, acc, f1_s, precision, recall, train_loss]))

                self.train_table.add_row(train_lst)

                print('Train at Epoch ' + str(self.current_epoch) + ' with sup loss ' + str(train_loss))

            val_res = self.print_res('val')
            if self.dataloader['unseen_test'] !=None:
                unseen_res = self.print_res('unseen_test')
            if val_res !=None:
                metrics = val_res[self.metric_type]
                self.scheduler.step(metrics,self.current_epoch)

                topk=self.topk.update(metrics)
                if topk!=-1:
                    self.best_epoch = self.current_epoch
                    # checkpoint = {
                    #     'model_state_dict': self.model.state_dict(),
                    #     'optimizer_state_dict': self.optim.state_dict(),
                    #     # 'scheduler_state_dict': scheduler.state_dict(),
                    #     'epoch': self.current_epoch
                    # }
                    # # torch.save(self.model.state_dict(), os.path.join(self.save_file_path, f"best_model_{topk}.pth"))
                    if self.save_best:
                        torch.save(self.model.state_dict(), os.path.join(self.save_file_path, f"best_model.pth"))
                    else:
                        torch.save(self.model.state_dict(), os.path.join(self.save_file_path, f"best_model_{topk}.pth"))
                    # torch.save(self.model.state_dict(), os.path.join(self.save_file_path, f"best_model_{topk}.pth"))
                    self.result_dict[topk] = {"epoch":self.current_epoch,f'test {self.metric_type}':metrics}

                    result_file = os.path.join(self.save_file_path, "result.json")
                    with open(result_file, "w") as f:
                        f.write(json.dumps(self.result_dict, ensure_ascii=False, indent=4, separators=(',', ':')))

                    test_prettytable_file = os.path.join(self.save_file_path, "test_markdowntable.csv")
                    train_prettytable_file = os.path.join(self.save_file_path, "train_markdowntable.csv")
                    val_prettytable_file = os.path.join(self.save_file_path, "val_markdowntable.csv")

                    with open(val_prettytable_file, "w") as fp:
                        fp.write(self.val_table.get_csv_string())

                    with open(train_prettytable_file, "w") as fp:
                        fp.write(self.train_table.get_csv_string())

                    with open(test_prettytable_file, "w") as fp:
                        fp.write(self.test_table.get_csv_string())

        self.save_result()

    def train_epoch(self,):
        float2str = lambda x: '%0.4f' % x
        self.model.train()
        loss_epoch = 0
        data_loader = self.dataloader['train']
        num_batches = len(data_loader)
        f_features, y_label, y_pred = [], [], []

        pbar = tqdm(enumerate(data_loader), total=num_batches)

        for i, batch in pbar:
            batch = Munch({k: v.to(self.device)
                           for k, v in batch.items()})
            self.step += 1
            if self.current_epoch > self.end_epoch:
                self.stage = 2
            inputs=Munch(
                batch=batch,
                device=self.device,
                stage = self.stage
            )
            self.optim.zero_grad()
            output = self.model(inputs)

            f_features.append(output.feature.detach().cpu().numpy())

            iil_loss = output.iil_loss
            sil_loss = output.sil_loss

            if inputs.stage ==1:
                loss = iil_loss + sil_loss
                loss.backward()
                # sil_loss.backward()
                loss_info = f'iil_loss {float2str(iil_loss.item())} sil_loss {float2str(sil_loss.item())} '
                loss = iil_loss+sil_loss
            else:
                # iil_loss = output.iil_loss
                # sil_loss = output.sil_loss

                loss = (iil_loss + sil_loss)*self.alpha + output.jcl_loss*(1-self.alpha)
                loss.backward()
                loss_info = f'loss {float2str(loss.item())} iil_loss {float2str(iil_loss.item())} sil_loss {float2str(sil_loss.item())} '

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 8)
            self.optim.step()
            # adjust_lr(self.optim,self.step,1e-4,max_iter=5000)


            if self.stage != 1:
                # if output.score.shape[-1] == 2:
                #     n = F.softmax(output.score, dim=1)[:, 1]
                # else:
                #     n = output.score
                f_features.append(output.feature.detach().cpu().numpy())
                n = F.softmax(output.score, dim=1)[:, 1]
                y_label = y_label + batch.label.to("cpu").tolist()
                self.metrics.update(n, batch.label.long())
                y_pred = y_pred + n.to("cpu").tolist()
            else:
                y_label = None
                y_pred = None

            loss_epoch += loss.item()
            # if self.experiment:
            #     self.experiment.log_metric("train_step model loss", loss.item(), step=self.step)
            lr = self.optim.state_dict()['param_groups'][0]['lr']

            # w2 = torch.exp(self.model.w[1]) / torch.sum(torch.exp(self.model.w))
            # pbar.set_description(f"{loss_info} weight {self.class_weight} {str_lr}")
            # pbar.set_description(f"{loss_info} lr {lr} ratio {output.ratio}")
            pbar.set_description(f"{loss_info} lr {lr} ")
            # else:
            #     pbar.set_description(f"train loss {loss.item()} lr {lr} class_weight {self.class_weight}")
            # self.scheduler.step(acc,self.current_epoch)

        if self.stage != 1:
            res = self.metrics.get_metric()
        else:
            y_label = None
            y_pred = None
            res = None

        loss_epoch = loss_epoch / num_batches

        self.metrics.reset()
        epoch_output = Munch(
            loss=loss_epoch,
            feature_s=f_features,
            y_label=y_label,
            y_pred=y_pred,
            res=res
        )
        return epoch_output

    def test(self, dataloader="test"):
        """
        [test,unseen_test] load best model
        [val,test_val,unseen_test_val] load current model
        :param dataloader: test,unseen_test,val,test_val,unseen_test_val
        :return:
        """
        loss_epoch = 0
        y_label, y_pred, softmax_ent = [], [], []
        iil_y_pred,sil_y_pred=[],[]
        data_loader = self.dataloader[dataloader]
        num_batches = len(data_loader)
        bcn_q_features, res_dict = [], {}
        ag_cluster_id = []
        # if dataloader in ["test", "unseen_test", 'train']:
        #     if self.save_best:
        #         state_dict = torch.load(os.path.join(self.save_file_path, f"best_model.pth"))
        #     else:
        #         topk = sorted(self.result_dict)[-1]
        #         print(f"load best_model_{topk}.pth")
        #         state_dict = torch.load(os.path.join(self.save_file_path, f"best_model_{topk}.pth"))
        #     # if self.cfg.set.mixup:
        #     #     self.model.net.load_state_dict(state_dict)
        #     #     # torch.save(self.model.net.state_dict(), os.path.join(self.save_file_path, f"best_model.pth"))
        #     # else:
        #     # torch.save(self.model.state_dict(), os.path.join(self.save_file_path, f"best_model.pth"))
        #     self.model.load_state_dict(state_dict['model_state_dict'])
        pbar = tqdm(enumerate(data_loader), total=num_batches)
        iil_metric = EvaMetric(task='cls', device=self.device)
        sil_metric = EvaMetric(task='cls', device=self.device)

        with torch.no_grad():
            self.model.eval()
            for i, batch in pbar:
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
                iil_n = F.softmax(output.iil_pred, dim=1)[:, 1]
                sil_n = F.softmax(output.sil_pred, dim=1)[:, 1]
                iil_metric.update(iil_n, batch.label.long())
                sil_metric.update(sil_n, batch.label.long())

                if self.stage != 1:
                    out_loss = return_loss(output.score, batch.label)
                    n = out_loss.n
                    self.metrics.update(n, batch.label.long())
                    loss = out_loss.loss

                    bcn_q_features.append(output.feature.detach().cpu().numpy())

                    y_label = y_label + batch.label.to("cpu").tolist()
                    y_pred = y_pred + n.to("cpu").tolist()
                    iil_y_pred = iil_y_pred + iil_n.to("cpu").tolist()
                    sil_y_pred = sil_y_pred + sil_n.to("cpu").tolist()

                    if self.cfg.set.dataset == 'ppi' and (dataloader in ["unseen_test", "unseen_test_val"]):
                        ag_cluster_id = ag_cluster_id + batch.ag_cluster_id.to("cpu").tolist()
                    loss_epoch += loss.item()
                   
                else:
                    bcn_q_features = None

            if self.stage != 1:
                res = self.metrics.get_metric()
                self.metrics.reset()
                # draw_calibration_curve(y_label, y_pred, iil_y_pred, sil_y_pred, self.save_file_path,
                #                        self.current_epoch)
            else:
                res = None

            res_dict['IIL'] = iil_metric.get_metric()
            res_dict['SIL'] = sil_metric.get_metric()
            iil_metric.reset()
            sil_metric.reset()
            loss_epoch = loss_epoch / num_batches
            if bcn_q_features != None:
                bcn_q_features = np.concatenate(bcn_q_features)
            result = Munch(
                bcn_q_features=bcn_q_features,
                y_label=y_label,
                res_dict=res_dict,
                res=res,
           
                loss=loss_epoch,
            )
        return result

    def print_res(self, print_type='val'):
        float2str = lambda x: '%0.4f' % x

        results = self.test(dataloader=print_type)
        # bcn_q_features, y_label, y_pred, loss=results.bcn_q_features, results.y_label, results.y_pred, results.loss
        bcn_q_features, y_label, res, loss, res_dict= results.bcn_q_features, results.y_label, results.res, results.loss, results.res_dict

        if res != None:
            acc, auprc, f1_s, mcc, precision, recall, roc_auc = res.acc, res.auprc, res.f1_s, res.mcc, res.precision, res.recall, res.roc_auc
            if print_type == 'test' or print_type == 'unseen_test':
                metric_list = ["epoch " + str(self.best_epoch)] + list(
                    map(float2str, [roc_auc, auprc, mcc, acc, f1_s, precision, recall, loss]))
            elif print_type == 'train':
                metric_list = ["epoch " + str(self.best_epoch)] + list(
                    map(float2str, [roc_auc, auprc, mcc, acc, f1_s, precision, recall, loss]))
            else:
                # print(metric_topk)
                metric_list = ["epoch " + str(self.current_epoch)] + list(
                    map(float2str, [roc_auc, auprc, mcc, acc, f1_s, precision, recall, loss]))

            if print_type == 'unseen_test' :
                title = 'Useen Test'
                self.test_table.add_row(metric_list)
            elif print_type == 'val':
                title = 'val'
                self.val_table.add_row(metric_list)
            else:
                title = 'train'
                self.train_table.add_row(metric_list)


            if print_type not in ['test', 'val', 'train']:
                type = 'current'
            else:
                type = 'best'

            print(f'Total {title} at {type} Epoch ' + str(self.current_epoch) + ' with loss ' + str(loss),
                  " AUROC "
                  + str(roc_auc) + " AUPRC " + str(auprc) + " MCC " + str(mcc) + " F1 " +
                  str(f1_s) + " Accuracy " + str(acc) + " Precision " + str(precision) + " Recall " + str(recall))

        for model_name, block_res in res_dict.items():
            block_acc, block_auprc, block_f1_s, block_mcc, block_precision, block_recall, block_roc_auc = block_res.acc, block_res.auprc, block_res.f1_s, block_res.mcc, block_res.precision, block_res.recall, block_res.roc_auc
            print(f'{model_name} {print_type} at current Epoch ' + str(self.current_epoch) + ' with loss ' + str(loss),
                  " AUROC "
                  + str(block_roc_auc) + " AUPRC " + str(block_auprc) + " MCC " + str(block_mcc) + " F1 " +
                  str(block_f1_s) + " Accuracy " + str(block_acc) + " Precision " + str(
                      block_precision) + " Recall " + str(block_recall))

        return res

    def save_result(self):
        # print(f"Best Epoch {self.best_epoch} Best ACC {self.best_acc}")
        test_prettytable_file = os.path.join(self.save_file_path, "test_markdowntable.csv")
        train_prettytable_file = os.path.join(self.save_file_path, "train_markdowntable.csv")

        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_csv_string())

        with open(test_prettytable_file, "w") as fp:
            fp.write(self.test_table.get_csv_string())

        # res=self.print_res('unseen_test')


class SiteTrainer(object):
    def __init__(self, trainer_parameter):
        # self.start_epoch = trainer_parameter.start_epoch
        self.end_epoch = trainer_parameter.end_epoch
        self.model = trainer_parameter.model
        self.save_best = trainer_parameter.save_best
        # self.print_test = trainer_parameter.print_test
        self.metric_type = trainer_parameter.metric_type
        self.optim = trainer_parameter.opt
        self.scheduler = ReduceLROnPlateau(self.optim, mode='max', patience=3)

        self.device = trainer_parameter.device
        self.metrics = EvaMetric(task='cls', device=self.device)
        self.epochs = trainer_parameter.cfg.solver.train_epoch
        self.init_epoch = trainer_parameter.current_epoch
        self.class_weight = 0
        self.current_epoch = trainer_parameter.current_epoch
        self.dataloader = {"train": trainer_parameter.train_dataloader,
                           "val": trainer_parameter.val_dataloader,
                           "unseen_test": trainer_parameter.unseen_dataloader}
        # self.train_dataloader = trainer_parameter.train_dataloader
        # self.test_dataloader = trainer_parameter.unseen_dataloader
        # self.val_dataloader = trainer_parameter.val_dataloader
        self.site_test = trainer_parameter.site_test
        # self.multi_dataloader = trainer_parameter.multi_dataloader
        cfg = trainer_parameter.cfg
        self.alpha = cfg.set.alpha
        self.batch_size = cfg.solver.batch_size
        self.step = 0
        self.sil_step = 0
        self.best_epoch = None
        self.best_acc = 0
        self.max_epoch = cfg.solver.train_epoch
        self.seed = cfg.solver.seed
        self.cfg = cfg  # 超参数
        self.topk = TopKModel(20)
        self.result_dict = {}

        self.model_name = cfg.set.model_name

        self.output_dir = cfg.result.output_dir + '/' + cfg.set.dataset + '/' + self.model_name + '/'
        self.save_file_path = self.output_dir + f'batch_{self.batch_size}_epoch_{self.max_epoch}_seed_{cfg.solver.seed}_fold_{cfg.set.ab_model}_{cfg.set.unseen_task}_alpha_{self.alpha}'
        if self.cfg.protein.use_bn:
            self.save_file_path = self.save_file_path + '_bn'

        train_header = ["#PDB_id", "Ag ACC", "Ag AUPRC", "Ag ROC_AUC", "Ag F1_Score", 'Ag MCC','Ag Precision', 'Ag Recall',
                  "Ab ACC", "Ab AUPRC", "Ab ROC_AUC", "Ab F1_Score", 'Ab MCC', 'Ab Precision', 'Ab Recall']

        test_header = ["#PDB_id", "Ag ACC", "Ag AUPRC", "Ag ROC_AUC", "Ag F1_Score", 'Ag MCC','Ag Precision', 'Ag Recall',
                  "Ab ACC", "Ab AUPRC", "Ab ROC_AUC", "Ab F1_Score", 'Ab MCC', 'Ab Precision', 'Ab Recall']

        val_header = ["#PDB_id", "Ag ACC", "Ag AUPRC", "Ag ROC_AUC", "Ag F1_Score", 'Ag MCC','Ag Precision', 'Ag Recall',
                  "Ab ACC", "Ab AUPRC", "Ab ROC_AUC", "Ab F1_Score", 'Ab MCC', 'Ab Precision', 'Ab Recall']

        if self.cfg.set.phy:
            self.save_file_path = self.save_file_path + '_phy'

        if self.cfg.predict.simple:
            self.save_file_path = self.save_file_path + '_simple'

        if not os.path.exists(self.save_file_path):
            os.makedirs(self.save_file_path)
        # elif os.path.getsize(self.save_file_path) != 0:
        #     # 存在但是文件夹不为空
        #     self.save_file_path = self.save_file_path + f"_{self.cfg.set.version}"
        #     os.makedirs(self.save_file_path)
        else:
            print(self.save_file_path + " is empty")

        # self.cfg.dump(open(self.save_file_path + '/cfg.yaml', "w"))

        with open(self.save_file_path + '/cfg.json', "w") as f:
            json.dump(dict(self.cfg), f, indent=4)

        self.test_table = PrettyTable(test_header)
        self.val_table = PrettyTable(val_header)
        self.train_table = PrettyTable(train_header)
        self.stage = 1

    def train(self):
        float2str = lambda x: '%0.4f' % x
        for i in range(self.init_epoch + 1, self.epochs + 1):
            self.current_epoch += 1
            epoch_output = self.train_epoch()
            train_loss, bcn_q_features, y_label, y_pred, res = \
                epoch_output.loss, epoch_output.feature_s, epoch_output.y_label, epoch_output.y_pred, epoch_output.res
            # roc_auc, auprc,  mcc, acc, f1_s, precision, recall = evaluate_classification(y_pred, y_label)
            if res != None:
                acc, auprc, f1_s, mcc, precision, recall, roc_auc = res.acc, res.auprc, res.f1_s, res.mcc, res.precision, res.recall, res.roc_auc
                train_lst = ["epoch " + str(self.current_epoch)] + list(
                    map(float2str,
                        [roc_auc, auprc, mcc, acc, f1_s, precision, recall, train_loss]))

                self.train_table.add_row(train_lst)

                print('Train at Epoch ' + str(self.current_epoch) + ' with sup loss ' + str(train_loss))

            val_res = self.print_res('val')
            if self.dataloader['unseen_test'] != None:
                unseen_res = self.print_res('unseen_test')
            if val_res != None:
                metrics = val_res[self.metric_type]
                self.scheduler.step(metrics, self.current_epoch)
                # self.scheduler.step(metrics=acc,epoch=self.current_epoch)

                topk = self.topk.update(metrics)
                if topk != -1:
                    self.best_epoch = self.current_epoch
                    checkpoint = {
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optim.state_dict(),
                        # 'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': self.current_epoch
                    }
                    # torch.save(self.model.state_dict(), os.path.join(self.save_file_path, f"best_model_{topk}.pth"))
                    if self.save_best:
                        torch.save(checkpoint, os.path.join(self.save_file_path, f"best_model.pth"))
                    else:
                        torch.save(checkpoint, os.path.join(self.save_file_path, f"best_model_{topk}.pth"))

                    # torch.save(self.model.state_dict(), os.path.join(self.save_file_path, f"best_model_{topk}.pth"))
                    self.result_dict[topk] = {"epoch": self.current_epoch, f'test {self.metric_type}': metrics}

                    result_file = os.path.join(self.save_file_path, "result.json")
                    with open(result_file, "w") as f:
                        f.write(json.dumps(self.result_dict, ensure_ascii=False, indent=4, separators=(',', ':')))

                    test_prettytable_file = os.path.join(self.save_file_path, "test_markdowntable.csv")
                    train_prettytable_file = os.path.join(self.save_file_path, "train_markdowntable.csv")
                    val_prettytable_file = os.path.join(self.save_file_path, "val_markdowntable.csv")

                    with open(val_prettytable_file, "w") as fp:
                        fp.write(self.val_table.get_csv_string())

                    with open(train_prettytable_file, "w") as fp:
                        fp.write(self.train_table.get_csv_string())

                    with open(test_prettytable_file, "w") as fp:
                        fp.write(self.test_table.get_csv_string())

        self.save_result()

    def train_epoch(self, ):
        float2str = lambda x: '%0.4f' % x
        self.model.train()
        loss_epoch = 0
        data_loader = self.dataloader['train']
        num_batches = len(data_loader)
        f_features, y_label, y_pred = [], [], []

        pbar = tqdm(enumerate(data_loader), total=num_batches)

        for i, batch in pbar:
            batch = Munch({k: v.to(self.device)
                           for k, v in batch.items()})
            self.step += 1
            if self.current_epoch>self.end_epoch:
                self.stage = 2
            inputs = Munch(
                batch=batch,
                device=self.device,
                stage=self.stage
            )
            self.optim.zero_grad()
            output = self.model(inputs)

            f_features.append(output.feature.detach().cpu().numpy())

            iil_loss = output.iil_loss
            sil_loss = output.sil_loss

            if inputs.stage == 1:
                loss = iil_loss + sil_loss
                loss.backward()
                # sil_loss.backward()
                loss_info = f'iil_loss {float2str(iil_loss.item())} sil_loss {float2str(sil_loss.item())} '
                loss = iil_loss + sil_loss
            else:
                # iil_loss = output.iil_loss
                # sil_loss = output.sil_loss

                loss = (iil_loss + sil_loss) * self.alpha + output.jcl_loss * (1 - self.alpha)
                loss.backward()
                loss_info = f'loss {float2str(loss.item())} iil_loss {float2str(iil_loss.item())} sil_loss {float2str(sil_loss.item())} '

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 8)
            self.optim.step()
            # adjust_lr(self.optim,self.step,1e-4,max_iter=5000)

            if self.stage != 1:
                # if output.score.shape[-1] == 2:
                #     n = F.softmax(output.score, dim=1)[:, 1]
                # else:
                #     n = output.score
                f_features.append(output.feature.detach().cpu().numpy())
                n = F.softmax(output.score, dim=1)[:, 1]
                y_label = y_label + batch.label.to("cpu").tolist()
                self.metrics.update(n, batch.label.long())
                y_pred = y_pred + n.to("cpu").tolist()
            else:
                y_label = None
                y_pred = None

            loss_epoch += loss.item()
            # if self.experiment:
            #     self.experiment.log_metric("train_step model loss", loss.item(), step=self.step)
            lr = self.optim.state_dict()['param_groups'][0]['lr']

            # w2 = torch.exp(self.model.w[1]) / torch.sum(torch.exp(self.model.w))
            # pbar.set_description(f"{loss_info} weight {self.class_weight} {str_lr}")
            # pbar.set_description(f"{loss_info} lr {lr} ratio {output.ratio}")
            pbar.set_description(f"{loss_info} lr {lr} ")
            # else:
            #     pbar.set_description(f"train loss {loss.item()} lr {lr} class_weight {self.class_weight}")
            # self.scheduler.step(acc,self.current_epoch)

        if self.stage != 1:
            res = self.metrics.get_metric()
        else:
            y_label = None
            y_pred = None
            res = None

        loss_epoch = loss_epoch / num_batches

        self.metrics.reset()
        epoch_output = Munch(
            loss=loss_epoch,
            feature_s=f_features,
            y_label=y_label,
            y_pred=y_pred,
            res=res
        )
        return epoch_output

    def test(self, dataloader="test"):
        """
        [test,unseen_test] load best model
        [val,test_val,unseen_test_val] load current model
        :param dataloader: test,unseen_test,val,test_val,unseen_test_val
        :return:
        """
        loss_epoch = 0
        y_label, y_pred, softmax_ent = [], [], []
        iil_y_pred, sil_y_pred = [], []
        data_loader = self.dataloader[dataloader]
        num_batches = len(data_loader)
        bcn_q_features, res_dict = [], {}
        ag_cluster_id = []
        if 0 <= self.current_epoch <= self.end_epoch:
            stage = 1
        else:
            stage = 2

        pbar = tqdm(enumerate(data_loader), total=num_batches)
        iil_ab_metric = EvaMetric(task='cls', device=self.device)
        iil_ag_metric = EvaMetric(task='cls', device=self.device)

        sil_ab_metric = EvaMetric(task='cls', device=self.device)
        sil_ag_metric = EvaMetric(task='cls', device=self.device)

        with torch.no_grad():
            self.model.eval()
            for i, batch in pbar:
                batch = Munch({k: v.to(self.device)
                               for k, v in batch.items()})

                inputs = Munch(
                    batch=batch,
                    device=self.device,
                    # is_mixup=False,
                    mode='test',
                    stage=stage
                )

                output = self.model(inputs)
                iil_ab_n = F.softmax(output.iil_ab_pred, dim=1)[:, 1]
                iil_ag_n = F.softmax(output.iil_ag_pred, dim=1)[:, 1]

                sil_ab_n = F.softmax(output.sil_ab_pred, dim=1)[:, 1]
                sil_ag_n = F.softmax(output.sil_ag_pred, dim=1)[:, 1]

                iil_ab_metric.update(iil_ab_n, batch.ab_label.long())
                iil_ag_metric.update(iil_ag_n, batch.ag_label.long())

                sil_ab_metric.update(sil_ab_n, batch.ab_label.long())
                sil_ag_metric.update(sil_ag_n, batch.ag_label.long())

                if stage != 1:
                    out_loss = return_loss(output.score, batch.label)
                    n = out_loss.n
                    self.metrics.update(n, batch.label.long())
                    loss = out_loss.loss

                    y_label = y_label + batch.label.to("cpu").tolist()
                    y_pred = y_pred + n.to("cpu").tolist()
                    iil_ab_y_pred = iil_ab_y_pred + iil_ab_n.to("cpu").tolist()
                    iil_ag_y_pred = iil_ag_y_pred + iil_ag_n.to("cpu").tolist()

                    sil_ab_y_pred = sil_ab_y_pred + sil_ab_n.to("cpu").tolist()
                    sil_ag_y_pred = sil_ag_y_pred + sil_ag_n.to("cpu").tolist()

                    loss_epoch += loss.item()


            if stage != 1:
                res = self.metrics.get_metric()
                self.metrics.reset()
            else:
                res = None

            res_dict['IIL_ab'] = iil_ab_metric.get_metric()
            res_dict['IIL_ag'] = iil_ag_metric.get_metric()

            res_dict['SIL_ab'] = sil_ab_metric.get_metric()
            res_dict['SIL_ag'] = sil_ag_metric.get_metric()

            iil_ab_metric.reset()
            iil_ag_metric.reset()

            sil_ab_metric.reset()
            sil_ag_metric.reset()

            loss_epoch = loss_epoch / num_batches

            result = Munch(

                y_label=y_label,
                res_dict=res_dict,
                res=res,

                loss=loss_epoch,
            )
        return result

    def print_res(self, print_type='val'):
        float2str = lambda x: '%0.4f' % x

        results = self.test(dataloader=print_type)
        # bcn_q_features, y_label, y_pred, loss=results.bcn_q_features, results.y_label, results.y_pred, results.loss
        bcn_q_features, y_label, res, loss, res_dict = results.bcn_q_features, results.y_label, results.res, results.loss, results.res_dict

        if res != None:
            acc, auprc, f1_s, mcc, precision, recall, roc_auc = res.acc, res.auprc, res.f1_s, res.mcc, res.precision, res.recall, res.roc_auc
            if print_type == 'test' or print_type == 'unseen_test':
                metric_list = ["epoch " + str(self.best_epoch)] + list(
                    map(float2str, [roc_auc, auprc, mcc, acc, f1_s, precision, recall, loss]))
            elif print_type == 'train':
                metric_list = ["epoch " + str(self.best_epoch)] + list(
                    map(float2str, [roc_auc, auprc, mcc, acc, f1_s, precision, recall, loss]))
            else:
                # print(metric_topk)
                metric_list = ["epoch " + str(self.current_epoch)] + list(
                    map(float2str, [roc_auc, auprc, mcc, acc, f1_s, precision, recall, loss]))

            if print_type == 'unseen_test':
                title = 'Useen Test'
                self.test_table.add_row(metric_list)
            elif print_type == 'val':
                title = 'val'
                self.val_table.add_row(metric_list)
            else:
                title = 'train'
                self.train_table.add_row(metric_list)

            if print_type not in ['test', 'val', 'train']:
                type = 'current'
            else:
                type = 'best'

            print(f'Total {title} at {type} Epoch ' + str(self.current_epoch) + ' with loss ' + str(loss),
                  " AUROC "
                  + str(roc_auc) + " AUPRC " + str(auprc) + " MCC " + str(mcc) + " F1 " +
                  str(f1_s) + " Accuracy " + str(acc) + " Precision " + str(precision) + " Recall " + str(recall))

        for model_name, block_res in res_dict.items():
            block_acc, block_auprc, block_f1_s, block_mcc, block_precision, block_recall, block_roc_auc = block_res.acc, block_res.auprc, block_res.f1_s, block_res.mcc, block_res.precision, block_res.recall, block_res.roc_auc
            print(f'{model_name} {print_type} at current Epoch ' + str(self.current_epoch) + ' with loss ' + str(loss),
                  " AUROC "
                  + str(block_roc_auc) + " AUPRC " + str(block_auprc) + " MCC " + str(block_mcc) + " F1 " +
                  str(block_f1_s) + " Accuracy " + str(block_acc) + " Precision " + str(
                      block_precision) + " Recall " + str(block_recall))

        return res

    def save_result(self):
        # print(f"Best Epoch {self.best_epoch} Best ACC {self.best_acc}")
        test_prettytable_file = os.path.join(self.save_file_path, "test_markdowntable.csv")
        train_prettytable_file = os.path.join(self.save_file_path, "train_markdowntable.csv")

        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_csv_string())

        with open(test_prettytable_file, "w") as fp:
            fp.write(self.test_table.get_csv_string())

        # res=self.print_res('unseen_test')