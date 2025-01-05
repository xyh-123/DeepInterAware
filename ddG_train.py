import pandas as pd
from prettytable import PrettyTable
from dataloader import MuteDataset
from utools.comm_utils import set_seed, ReduceLROnPlateau
from networks.models import load_AM_model
import torch
from tqdm import tqdm
from munch import Munch
import numpy as np
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
from sklearn.model_selection import KFold
import scipy
import os
from torch.utils.data import DataLoader
import argparse
project_dir = os.path.dirname(os.path.abspath(__file__))

def calculate_delta_G(Kd):
    R = 1.987
    T = 298
    delta_G = R * T * np.log(Kd)
    return delta_G/1000

def mute_test(model,dataloader,device):
    model.eval()
    num_batches = len(dataloader)
    pbar = tqdm(enumerate(dataloader), total=num_batches)
    y_label,y_pred,key_ids = [],[],[]
    with torch.no_grad():
        for i, (wt_inputs, mu_inputs, label, key_id) in pbar:
            # print(label)
            wt_inputs = Munch({k: v.to(device)
                               for k, v in wt_inputs.items()})
            mu_inputs = Munch({k: v.to(device)
                               for k, v in mu_inputs.items()})
            label = label.to(device)
            output = model(wt_inputs, mu_inputs, label)
            n = output.score

            y_label = y_label + label.to("cpu").tolist()
            y_pred = y_pred + n.to("cpu").tolist()
            key_ids = key_ids + key_id.tolist()

        y_label = np.concatenate(y_label)
        y_pred = np.concatenate(y_pred)
        key_ids = np.concatenate(key_ids)
        # print(y_label)
        # print(y_pred)
        RMSD = np.sqrt(mean_squared_error(y_label, y_pred))
        PCC = scipy.stats.pearsonr(y_label, y_pred)[0]
        MAE = mean_absolute_error(y_label, y_pred)
        R2 = r2_score(y_label, y_pred)

    return RMSD,PCC,MAE,R2,y_label,y_pred,key_ids

def mute_train(model,train_dataloader,device,opt):
    float2str = lambda x: '%0.4f' % x
    num_batches = len(train_dataloader)
    pbar = tqdm(enumerate(train_dataloader), total=num_batches)
    y_label,y_pred,key_ids = [], [], []
    model.train()
    loss_list = []
    for i, (wt_inputs, mu_inputs, label, key_id) in pbar:
        wt_inputs = Munch({k: v.to(device)
                           for k, v in wt_inputs.items()})
        mu_inputs = Munch({k: v.to(device)
                           for k, v in mu_inputs.items()})
        label = label.to(device)
        opt.zero_grad()
        output = model(wt_inputs, mu_inputs, label)
        loss = output.loss
        loss.backward()
        opt.step()
        loss_info = f'loss {float2str(loss.item())}'
        loss_list.append(loss.item())

        n = output.score

        y_label = y_label + label.to("cpu").tolist()
        y_pred = y_pred + n.to("cpu").tolist()
        key_ids = key_ids + key_id.to("cpu").tolist()

        lr = opt.state_dict()['param_groups'][0]['lr']
        pbar.set_description(f"{loss_info} lr {lr} ")

    y_label = np.concatenate(y_label)
    y_pred = np.concatenate(y_pred)
    # print(y_label)
    # print(y_pred)

    RMSE = np.sqrt(mean_squared_error(y_label, y_pred))
    PCC = scipy.stats.pearsonr(y_label, y_pred)[0]
    MAE = mean_absolute_error(y_label, y_pred)
    R2 = r2_score(y_label, y_pred)
    return RMSE,PCC,MAE,R2,np.mean(loss_list)

def affinity_mutation(trainer_parameter):
    seed = trainer_parameter.seed
    lr = trainer_parameter.lr
    gpu = trainer_parameter.gpu
    data_path = trainer_parameter.data_path
    model_path = trainer_parameter.model_path
    batch_size = trainer_parameter.batch_size
    dataset = trainer_parameter.dataset
    result_path = os.path.join(trainer_parameter.result_path, dataset)
    epoch = trainer_parameter.epoch
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    label = torch.load(f'{data_path}/{dataset}/label.pt')
    print(result_path)
    # key_id = torch.load(f'{path}/key_id.pt')
    # result_path = f'/root/deepinteraware/result/mutant_result/DeepInterAware'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    params = {'batch_size':batch_size,'shuffle': True,'drop_last': False}
    index = range(label.shape[0])
    # print(label.shape[0])

    float2str = lambda x: '%0.4f' % x
    set_seed(seed)
    # pbar = tqdm(seqs.keys(),total=len(seqs))
    RMSE_whole = []
    PCC_whole = []
    MAE_whole = []
    R2_whole = []
    kf = KFold(n_splits=10, shuffle=True)
    fold_header = ["Fold", "PCC", "RMSE", 'MAE', 'R2']
    fold_table = PrettyTable(fold_header)
    fold = 1
    for train_index, test_index in kf.split(index):
        model = load_AM_model(f'{os.getcwd()}/configs/SAbDab.yml',
                                  model_path=f'{model_path}/SAbDab.pth', gpu=gpu)
        print(model)
        for name, param in model.named_parameters():
            if name == 'pair_encoder':
                param.requires_grad = False

        opt = torch.optim.Adam(model.predict_layer.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(opt, mode='max', patience=3)
        # scheduler = CosineAnnealingLR(opt, T_max=50, eta_min=1e-5)
        best_RMSE = float('inf')
        best_PCC = -1
        best_MAE = float('inf')
        best_R2 = -1
        print(f"train on the {fold} fold")
        train_dataset = MuteDataset(train_index,dataset,data_path=data_path)
        test_dataset = MuteDataset(test_index,dataset,data_path=data_path)
        train_dataloader = DataLoader(train_dataset, **params)
        test_dataloader = DataLoader(test_dataset, **params)
        header = ["#Epoch", "PCC", "RMSE", 'MAE', 'R2']
        table = PrettyTable(header)
        for epoch in range(1, epoch):
            train_RMSE,train_PCC,train_MAE,train_R2,train_loss = mute_train(model,train_dataloader,device,opt)
            scheduler.step(train_RMSE, epoch)
            # scheduler.step()

            RMSE, PCC, MAE,R2,y_label,y_pred,key_ids = mute_test(model, test_dataloader, device)
            print(f"Fold:{fold}  Epoch: {epoch}")
            print(f"Train loss:{train_loss} RMSE:{train_RMSE} MAE:{train_MAE} PCC:{train_PCC} R2:{train_R2}")
            print(f"Test RMSE:{RMSE} MAE:{MAE} PCC:{PCC} R2:{R2}")
            if PCC>best_PCC:
                best_RMSE = RMSE
                best_PCC = PCC
                best_MAE = MAE
                best_R2 = R2
                # draw(y_pred, y_label, path, fold, seed)
                data = pd.DataFrame()
                data.insert(loc=0, column='key_id', value=key_ids)
                data.insert(loc=1, column='y_label', value=y_label)
                data.insert(loc=2, column='y_pred', value=y_pred)
                data.to_csv(f'{result_path}/seed_{seed}_fold_{fold}_result.csv', index=False)

                table.add_row([epoch,PCC,RMSE,MAE,R2])
                torch.save(model.state_dict(), os.path.join(result_path, f"{seed}_best_model_{fold}.pth"))

        PCC_whole.append(best_PCC)
        RMSE_whole.append(best_RMSE)
        MAE_whole.append(best_MAE)
        R2_whole.append(best_R2)

        fold_table.add_row([fold,best_PCC, best_RMSE, best_MAE, best_R2])
        with open(f'{result_path}/seed_{seed}_{fold}_metric_result.csv', "w") as fp:
            fp.write(table.get_csv_string())

        file_out = open(f'{result_path}/seed_{seed}_{fold}_mean_result.txt', "wb")
        content = f"{seed} {fold} Fold RMSD:{best_RMSE} PCC:{best_PCC} MAE:{best_MAE} R2:{best_R2}"
        file_out.write(content.encode('utf-8'))
        file_out.close()
        fold += 1


    mean_RMSE= np.mean(RMSE_whole)
    mean_MAE = np.mean(MAE_whole)
    mean_PCC = np.mean(PCC_whole)
    mean_R2= np.mean(R2_whole)

    file_out = open(f'{result_path}/seed_{seed}_{fold}_mean_result.txt', "wb")
    content = f"{seed} {fold} Fold RMSE:{mean_RMSE} PCC:{mean_PCC} MAE:{mean_MAE} R2:{mean_R2}"
    file_out.write(content.encode('utf-8'))
    file_out.close()
    fold += 1
    # max_table.add_row([fold,max_PCC,max_PCC,max_RMSE,max_MAE,max_R2])
    print(f"Mean RMSE:{mean_RMSE} MAE:{mean_MAE} PCC:{mean_PCC} R2:{mean_R2}")

    with open(f'{result_path}/seed_{seed}_all_fold_metric_result_{lr}_{batch_size}.csv', "w") as fp:
        fp.write(fold_table.get_csv_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="antibody-antigen binding affinity prediction")
    parser.add_argument('--model_path', default=f'{project_dir}/save_models/', type=str, metavar='TASK',
                        help='data path')
    parser.add_argument('--data_path', default=f'{project_dir}/data/', type=str, metavar='TASK',
                        help='data path')
    parser.add_argument('--result_path', default=f'{project_dir}/result/mutant_result/', type=str, metavar='TASK',
                        help='data path')
    parser.add_argument('--dataset', default='AB-Bind', type=str,
                        metavar='TASK',
                        help='data path')
    parser.add_argument('--gpu', default=0, type=int, metavar='S', help='run GPU number')
    parser.add_argument('--batch_size', default=32, type=int, metavar='S', help='run GPU number')
    parser.add_argument('--seed', default=0, type=int, metavar='S', help='run GPU number')
    parser.add_argument('--lr', default=5e-4, type=float, metavar='S', help='run GPU number')
    parser.add_argument('--epoch', default=50, type=int, metavar='S', help='run GPU number')
    args = parser.parse_args()
    trainer_parameter = Munch(
        lr=args.lr,
        seed=args.seed,
        gpu = args.gpu,
        model_name = args.model_name,
        ab_name = args.ab_name,
        data_path=args.data_path,
        model_path=args.model_path,
        result_path=args.result_path,
        batch_size = args.batch_size,
        dataset = args.dataset,
        operation =args.operation,
        epoch = args.epoch
    )
    affinity_mutation(trainer_parameter)