import argparse
import os
import sys
import torch
import pandas as pd
import os
import torch.nn.functional as F

# project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# # 将 project 目录添加到 sys.path
# sys.path.append(project_dir)
from utools.cdr_extract import extract_CDR
from utools.feature_encoder import get_mask,getAAfeature
from networks.models import load_model


project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def get_args():
    parser = argparse.ArgumentParser(description="antibody-antigen binding affinity prediction")
    parser.add_argument('--pair_file', default=f'{project_dir}/data/example/ag_ab_pair.csv', type=str, metavar='TASK',help='model train path')
    parser.add_argument('--model_path', default=f'./save_models/', type=str, metavar='S',
                        help='training dataset')
    parser.add_argument('--result_path', default=f'./result/', type=str, metavar='S',
                        help='training dataset')
    parser.add_argument('--wt', default=f'{project_dir}/data/example/wt.csv', type=str, metavar='TASK',
                        help='data path')
    parser.add_argument('--mu', default=f'{project_dir}/data/example/mu.csv', type=str, metavar='TASK',
                        help='data path')
    parser.add_argument('--gpu', default=0, type=int, metavar='S', help='run GPU number')
    parser.add_argument('--task', default='binding_site', type=str, metavar='S', help='run GPU number')
    args=parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    if args.task in ['binding_site','binding','neutralization']:
        data = pd.read_csv(args.pair_file)
        ag_list = data['antigen'].to_list()
        heavy = data['heavy'].to_list()
        light = data['light'].to_list()
        if args.task == 'binding':
            max_ab_len = 110
            max_ag_len = 800
            ab_list = []
            for res_H, res_L in zip(heavy, light):
                ab_info, ab_cdr = extract_CDR(res_H, res_L)
                ab_list.append((ab_info['H_cdr'], ab_info['L_cdr']))
        elif args.task == 'binding_site':
            max_ab_len = 105
            max_ag_len = 1024
            ab_list = zip(heavy,light)
        else:
            max_ab_len = 676
            max_ag_len = 912
            ab_list = zip(heavy,light)

        ag_token, ab_token,_,_ = getAAfeature(ag_list, ab_list, gpu=args.gpu,max_ag_len=max_ag_len,max_ab_len=max_ab_len)
        ag_mask, ab_mask, ag_len, ab_len = get_mask(ag_list,ab_list, args.gpu,max_ag_len=max_ag_len,max_ab_len=max_ab_len)

        if args.task == 'binding_site':
            model= load_model('DeepInterAwareSite',f'./configs/SAbDab.yml',
                              model_path=os.path.join(args.model_path,'bindingsite.pth'),gpu=args.gpu)
            model.eval()
            with torch.no_grad():
                output = model.inference(ag_token, ab_token,ag_mask, ab_mask)
                ab_predict = output.ab_score
                ag_predict = output.ag_score
                device = ab_predict.device
                ab_n = F.softmax(ab_predict, dim=1)[:, 1]
                ag_n = F.softmax(ag_predict, dim=1)[:, 1]
            print("Antibody site prediction")
            print(ab_n)
            print("Antigen site prediction")
            print(ag_n)
        else:
            if args.task == 'binding':
                model_name = 'SAbDab'
            else:
                model_name = 'HIV'
            model = load_model('DeepInterAware', f'./configs/{model_name}.yml',
                               model_path=os.path.join(args.model_path, f'{model_name}.pth'), gpu=args.gpu)
            model.eval()
            with torch.no_grad():
                output = model.inference(ag_token, ab_token, ag_mask, ab_mask)
                predict = output.score
            print(f"Predicted Score:")
            print(predict)

    else:
        model = load_model(model_name='DeepInterAware_AM', yml=f'./configs/SAbDab.yml',
                           model_path=os.path.join(args.model_path,'ddG.pth'),
                           gpu=args.gpu)
        wt_data = pd.read_csv(f"{args.wt}", encoding='ISO-8859-1')
        mu_data = pd.read_csv(f"{args.mu}", encoding='ISO-8859-1')
        wt_ag_list = wt_data['antigen'].to_list()
        wt_ab_list = list(map(tuple, zip(wt_data['heavy'], wt_data['light'])))

        mu_ag_list = mu_data['antigen'].to_list()
        mu_ab_list = list(map(tuple, zip(mu_data['heavy'], mu_data['light'])))

        wt_ag_token_ft, wt_ab_token_ft = getAAfeature(wt_ag_list, wt_ab_list, gpu=args.gpu, max_ag_len=800,
                                                      max_ab_len=908)
        wt_ag_mask, wt_ab_mask, wt_ag_len, wt_ab_len = get_mask(wt_ag_list, wt_ab_list, gpu=args.gpu, max_ag_len=800,
                                                                max_ab_len=908)
        mu_ag_token_ft, mu_ab_token_ft = getAAfeature(mu_ag_list, mu_ab_list, gpu=args.gpu, max_ag_len=800,
                                                      max_ab_len=908)
        mu_ag_mask, mu_ab_mask, mu_ag_len, mu_ab_len = get_mask(mu_ag_list, mu_ab_list, gpu=args.gpu, max_ag_len=800,
                                                                max_ab_len=908)
        wt = wt_ag_token_ft, wt_ab_token_ft, wt_ag_mask, wt_ab_mask
        mu = mu_ag_token_ft, mu_ab_token_ft, mu_ag_mask, mu_ab_mask

        model.eval()
        with torch.no_grad():
            output = model.inference(wt, mu)
        print(output.score)
        result = pd.DataFrame()
        result.insert(loc=0, column='ddG', value=output.score.squeeze(-1).tolist())
        result.to_csv(f"{args.result_path}/result.csv", index=False)
