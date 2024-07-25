from utils.binding_site import draw_site_map, get_binding_site
from utils.feature_encoder import get_mask,getAAfeature
from models import load_model
import torch
import sys
import os
# sys.path.append(os.getcwd())
# print(os.getcwd())
# ag_list = [
#         'DSFVCFEHKGFDISQCPKIGGHGSKKCTGDAAFCSAYECTAQYANAYCSHA',
#         'SILDIKQGPKESFRDYVDRFFKTLRAEQCTQDVKNWMTDTLLVQNANPDCKTILRALGPGATLEEMMTACQGV'
# ]
#
# ab_list = [
#     ('TFSNYWMNWVLEWVAEIRLKSNNYATHYAYCTRGNGNYRAMDYW',None),
#     ('YAMIWVVEYIGIINTGGSASYAFCARTRGVNDAYEHAFDPW',None)
# ]
# ag_token_ft,ab_token_ft = getAAfeature(ag_list, ab_list, gpu=0)
# ag_mask,ab_mask = get_mask(ag_list, ab_list,gpu=0)
# # model= load_model(f'{os.getcwd()}/configs/SAbDab.yml',model_path=f'{os.getcwd()}/save_model/SAbDab.pth',gpu=0)
# model= load_model(f'./configs/SAbDab.yml',model_path=f'./save_models/SAbDab.pth',gpu=0)
# model.eval()
# with torch.no_grad():
#     output = model.inference(ag_token_ft,ab_token_ft,ag_mask,ab_mask)
# print(output.score)

full_seq, full_label, seq_dict, ab_info, label_dict = get_binding_site('6i9i','H','L','D')
ag_list = [seq_dict['ag_seq']]
ab_list = [(seq_dict['H_cdr'],seq_dict['L_cdr'])]
ag_token_ft,ab_token_ft = getAAfeature(ag_list, ab_list, gpu=1)
ag_mask,ab_mask,ag_len,ab_len = get_mask(ag_list, ab_list,gpu=1)
for seed in range(0,5):
    # model= load_model(f'{os.getcwd()}/configs/SAbDab.yml',model_path=f'{os.getcwd()}/save_model/SAbDab.pth',gpu=0)
    model= load_model(f'./configs/SAbDab.yml',model_path=f'./save_models/new_SAbDab_{seed}.pth',gpu=1)
    model.eval()
    with torch.no_grad():
        output = model.inference(ag_token_ft,ab_token_ft,ag_mask,ab_mask)
    print(output.score)
    ag_recall,ab_recall,pair_recall = draw_site_map('6i9i',output,ag_len,ab_len,label_dict,threshold=0.5)