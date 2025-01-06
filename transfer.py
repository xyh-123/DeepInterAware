import pandas as pd
from prettytable import PrettyTable

from configs import get_cfg_defaults
from dataloader import PairDataset, return_dataset
from torch.utils.data import DataLoader
import os
import argparse
import torch
from networks.models import DeepInterAware
from train import DeepInterAwareTrainer
from utools.comm_utils import set_seed, EvaMetric, return_loss
from munch import Munch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
project_dir = os.path.dirname(os.path.abspath(__file__))


def load_finetune_dataset(cfg,seed):
    dataFolder = f'{os.getcwd()}/data/CoVAbDab/'
    # antibody = pd.read_csv(dataFolder+'antibody.csv')
    # antigen = pd.read_csv(dataFolder+'antigen.csv')
    train_data = pd.read_csv(dataFolder+f'finetune_train.csv')
    test_data = pd.read_csv(dataFolder+f'finetune_test.csv')

    train_dataset=PairDataset(train_data,dataFolder,cfg)
    test_dataset=PairDataset(test_data,dataFolder,cfg)
    train_loader = DataLoader(train_dataset,shuffle=False, num_workers=0,drop_last=False,batch_size=256)
    test_loader = DataLoader(test_dataset,shuffle=False, num_workers=0,drop_last=False,batch_size=256)
    return train_loader,test_loader

def train(loader,model,optim,stage,alpha):
    float2str = lambda x: '%0.4f' % x
    model.train()
    loss_epoch = 0
    num_batches = len(loader)
    f_features, y_label, y_pred ,ag_list=[],[],[],[]
    metrics = EvaMetric(task='cls', device=device)
    pbar = tqdm(enumerate(loader), total=num_batches)
    step=0
    # stage = 2
    for i, batch in pbar:
        batch = Munch({k: v.to(device)
                       for k, v in batch.items()})
        step += 1
        inputs = Munch(
            batch=batch,
            device=device,
            stage=stage
        )
        # optim.zero_grad()
        optim.opt.zero_grad()
        output = model(inputs)
        f_features.append(output.feature.detach().cpu().numpy())
        iil_loss = output.iil_loss
        sil_loss = output.sil_loss

        if inputs.stage == 1:
            loss = sil_loss+iil_loss
            loss.backward()
            loss_info = f'iil_loss {float2str(output.iil_loss.item())} sil_loss {float2str(output.sil_loss.item())} '
        else:
            loss = (iil_loss + sil_loss) * alpha + output.jcl_loss * (1-alpha)
            loss.backward()
            loss_info = f'loss {float2str(loss.item())} iil_loss {float2str(iil_loss.item())} sil_loss {float2str(sil_loss.item())} '
        # loss.backward()
        optim.opt.step()
        if stage != 1:
            if output.score.shape[-1] == 2:
                n = F.softmax(output.score, dim=1)[:, 1]
            else:
                n = output.score
            f_features.append(output.feature.detach().cpu().numpy())
            n = F.softmax(output.score, dim=1)[:, 1]
            y_label = y_label + batch.label.to("cpu").tolist()
            metrics.update(n, batch.label.long())
            y_pred = y_pred + n.to("cpu").tolist()

        else:
            y_label = None
            y_pred = None

        lr = optim.opt.state_dict()['param_groups'][0]['lr']
        pbar.set_description(f"{loss_info} lr {lr}")
        # else:
        #     pbar.set_description(f"train loss {loss.item()} lr {lr} class_weight {self.class_weight}")
    # scheduler.step()
    # topk=metric_top_k(y_pred,y_label,[10,50,100,500],ag_list)

    if stage != 1:
        res = metrics.get_metric()
        acc, auprc, f1_s, mcc, precision, recall, roc_auc = res.acc, res.auprc, res.f1_s, res.mcc, res.precision, res.recall, res.roc_auc
        print(f"Train" + f" AUROC " + str(roc_auc) + " AUPRC " + str(auprc) + " MCC " + str(mcc) + " F1 " + str(
            f1_s) + " Accuracy " + str(acc) + " Precision " + str(precision) + " Recall " + str(recall))
        metrics.reset()
    else:
        y_label = None
        y_pred = None
        res = None

    loss_epoch = loss_epoch / num_batches
    # res =metrics.get_metric()
    epoch_output=Munch(
        loss=loss_epoch,
        feature_s=f_features,
        y_label=y_label,
        y_pred=y_pred,
        # topk=topk,
        res=res
    )
    return epoch_output

def test(loader,model):
    """
    [test,unseen_test] load best model
    [val,test_val,unseen_test_val] load current model
    :param dataloader: test,unseen_test,val,test_val,unseen_test_val
    :return:
    """

    num_batches = len(loader)
    bcn_q_features,res_dict=[],{}
    ag_cluster_id=[]

    pbar = tqdm(enumerate(loader), total=num_batches)
    metrics = EvaMetric(task='cls', device=device)
    iil_metric = EvaMetric(task='cls', device=device)
    sil_metric = EvaMetric(task='cls', device=device)
    stage =2
    y_pred,y_label,ag_list,iil_pred,sil_pred=[],[],[],[],[]
    with torch.no_grad():
        model.eval()
        for i, batch in pbar:
            batch = Munch({k: v.to(device)
                           for k, v in batch.items()})
            inputs = Munch(
                batch=batch,
                device=device,
                # is_mixup=False,
                mode='test',
                stage=stage
            )

            output = model(inputs)
            iil_n = F.softmax(output.iil_pred, dim=1)[:, 1]
            sil_n = F.softmax(output.sil_pred, dim=1)[:, 1]
            iil_metric.update(iil_n, batch.label.long())
            sil_metric.update(sil_n, batch.label.long())
            out_loss = return_loss(output.score, batch.label)
            n = out_loss.n
            metrics.update(n, batch.label.long())
            loss = out_loss.loss

            bcn_q_features.append(output.feature.detach().cpu().numpy())

            y_label = y_label + batch.label.to("cpu").tolist()
            y_pred = y_pred + n.to("cpu").tolist()
            iil_pred = iil_pred + iil_n.to("cpu").tolist()
            sil_pred = sil_pred + sil_n.to("cpu").tolist()

            ag_list=ag_list+batch.ag_id.to("cpu").tolist()



        res1=metrics.get_metric()
        metrics.reset()

        res_dict['IILModule']=iil_metric.get_metric()
        res_dict['SILModule']=sil_metric.get_metric()
        iil_metric.reset()
        sil_metric.reset()

        features=np.concatenate(bcn_q_features)
        iil_pred=np.array(iil_pred)
        sil_pred=np.array(sil_pred)
        acc1, auprc1, f1_s1, mcc1, precision1, recall1, roc_auc1 = res1.acc, res1.auprc, res1.f1_s, res1.mcc, res1.precision, res1.recall, res1.roc_auc
        print(f"Test AUROC " + str(roc_auc1) + " AUPRC " + str(auprc1) + " MCC " + str(
            mcc1) + " F1 " + str(f1_s1) + " Accuracy " + str(acc1) + " Precision " + str(precision1) + " Recall " + str(
            recall1))



        for model_name, block_res in res_dict.items():
            block_acc, block_auprc, block_f1_s, block_mcc, block_precision, block_recall, block_roc_auc = block_res.acc, block_res.auprc, block_res.f1_s, block_res.mcc, block_res.precision, block_res.recall, block_res.roc_auc
            print(f'{model_name} '+" AUROC "
                  + str(block_roc_auc) + " AUPRC " + str(block_auprc) + " MCC " + str(block_mcc) + " F1 " +
                  str(block_f1_s) + " Accuracy " + str(block_acc) + " Precision " + str(block_precision) + " Recall " + str(block_recall))
        return features,np.array(y_pred),np.array(y_label),np.array(ag_list),iil_pred,sil_pred,res1

def merge_finetune(train_loader, test_loader, model, save_file_path, freeze, alpha):
        best_roc_auc = 0
        header = ["Epoch", "ROC_AUC", "AUPRC", 'MCC', 'Accuracy', 'F1', 'Precision', 'Recall']
        result_table = PrettyTable(header)
        float2str = lambda x: '%0.4f' % x
        if freeze:
            optims = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
        else:
            optims = Munch()
            parameters = [{'params': model.parameters()}]
            opt = torch.optim.Adam(parameters)
            optims['opt'] = opt
        # scheduler = CosineAnnealingLR(optim, T_max=50, eta_min=1e-5)
        for epoch in range(150):
            print(f"train epoch {epoch}")
            if 0 <= epoch <= 75:
                stage = 1
            else:
                stage = 2
            # epoch_output = new_merge_train(train_loader, model, optims,stage,alpha)
            epoch_output = train(train_loader, model, optims, stage, alpha)
            # scheduler.step()
            features, y_pred, y_label, ag_list, y_pred2, y_pred3, test_res = test(test_loader,model)
            if stage == 2:
                # res = epoch_output.res

                acc, auprc, f1_s, mcc, precision, recall, roc_auc = test_res.acc, test_res.auprc, test_res.f1_s, test_res.mcc, test_res.precision, test_res.recall, test_res.roc_auc
                metric_list = [str(epoch)] + list(
                    map(float2str, [roc_auc, auprc, mcc, acc, f1_s, precision, recall]))
                result_table.add_row(metric_list)
                if roc_auc > best_roc_auc:
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'epoch': epoch
                    }
                    # torch.save(self.model.state_dict(), os.path.join(self.save_file_path, f"best_model_{topk}.pth"))

                    torch.save(checkpoint, os.path.join(save_file_path, f"best_finetune_model.pth"))

                    best_roc_auc = roc_auc
                    # metric_list = [str(epoch)] + list(
                    #         map(float2str, [roc_auc, auprc, mcc, acc, f1_s,precision, recall]))+result_list
                    # result_table.add_row(metric_list)
        return result_table

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="antibody-antigen binding affinity prediction")
    parser.add_argument('--config', default=f'{project_dir}/configs/HIV.yml', type=str, metavar='S', help='pretrain model')
    parser.add_argument('--model_path', default=f'{project_dir}/save_models/', type=str, metavar='S', help='pretrain model')
    parser.add_argument('--gpu', default=0, type=int, metavar='S', help='run GPU number')
    parser.add_argument('--alpha', default=0.4, type=float, metavar='S', help='run GPU number')
    parser.add_argument('--data', default='./data', type=str, metavar='TASK',help='data path')
    parser.add_argument('--unseen_task', default='transfer', type=str, metavar='TASK',help='data path')
    parser.add_argument('--freeze', action='store_true', help='freeze model')
    parser.add_argument('--train_epoch',default=50, type=int, help='freeze model')
    parser.add_argument('--finetune_epoch',default=100, type=int, help='freeze model')
    parser.add_argument('--end_epoch', default=30, type=int, metavar='S', help='dataset')
    parser.add_argument('--save_best', action='store_true')
    parser.add_argument('--metric_type', default='roc_auc', type=str, metavar='S', help='dataset')
    parser.add_argument('--batch_size', default=32, type=int, metavar='S', help='dataset')

    args = parser.parse_args()
    # model_name =args.model_name
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    # dataFolder = f'{args.data}/data/'
    dataFolder = f'{os.getcwd()}/data/HIV/'

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.set.unseen_task = args.unseen_task
    cfg.set.alpha = args.alpha

    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    for seed in range(0,5):
        set_seed(seed)
        model = DeepInterAware(cfg)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.solver.lr, weight_decay=cfg.solver.weight_decay)
        model = model.to(device)
        scheduler = CosineAnnealingLR(opt, T_max=50, eta_min=1e-5)

        # if not os.path.exists(args.model_path+'HIV.pth'):
        #
        #     cfg.solver.seed = seed
        #     train_dataset, val_dataset, unseen_dataset= return_dataset(cfg, dataFolder)
        #     params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': cfg.solver.num_workers,
        #               'drop_last': True}
        #
        #
        #
        #     train_dataloader = DataLoader(train_dataset, sampler=None, **params)
        #     params['shuffle'] = True
        #     params['drop_last'] = False
        #     val_dataloader = DataLoader(val_dataset, **params)
        #     if unseen_dataset != None:
        #         unseen_dataloader = DataLoader(unseen_dataset, sampler=None, **params)
        #     else:
        #         unseen_dataloader = None
        #
        #
        #     trainer_parameter = Munch(
        #         device=device,
        #         current_epoch=0,
        #         save_best=args.save_best,
        #         metric_type=args.metric_type,
        #         sampler=False,
        #         # start_epoch=args.start_epoch,
        #         end_epoch=args.end_epoch,
        #         model=model,
        #         opt=opt,
        #         scheduler=scheduler,
        #         cfg=cfg,
        #         # cfg=cfg
        #     )
        #     trainer_parameter.train_dataloader = train_dataloader
        #     trainer_parameter.val_dataloader = val_dataloader
        #     trainer_parameter.unseen_dataloader = unseen_dataloader
        #
        #     trainer = DeepInterAwareTrainer(trainer_parameter)
        #     print()
        #     print(f"Directory for saving result: {trainer.save_file_path}")
        #     result = trainer.train()
        #
        #     with open(os.path.join(trainer.save_file_path, "model_architecture.txt"), "w") as wf:
        #         wf.write(str(model))

        print("Finetune on the CoVAbDab dataset")

        train_loader, test_loader = load_finetune_dataset(cfg,seed)
        state_dict = torch.load(args.model_path+'HIV.pth')

        # model = DeepInterAware(cfg)
        # print(state_dict['model_state_dict'].keys())
        model.load_state_dict(state_dict)
        if args.freeze:
            for name, param in model.named_parameters():
                if 'out_linear2' not in name :
                    param.requires_grad = False

        # 检查参数是否冻结
        # for name, param in model.named_parameters():
        #     print(name, param.requires_grad)

        model = model.to(device)
        result_table = merge_finetune(train_loader, test_loader, model, args.model_path, args.freeze, args.alpha)
        file = os.path.join(args.model_path, f"CoVAbDab_{seed}_{args.alpha}.csv")


        with open(file, "w") as fp:
            fp.write(result_table.get_csv_string())