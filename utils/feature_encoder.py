import ablang
import torch
import pandas as pd
import numpy as np
from munch import Munch
import re
from transformers import EsmTokenizer, EsmModel
import math
import sys
import os

from models import load_model
# from models import DeepInterAwar
def process_ablang_token_ft(heavy,light,model,max_ab_len=None):
    # device = torch.device('cuda:1')
    # antiberty = AntiBERTyRunner()
    torch.manual_seed(42)
    padding_tensor = (torch.rand((768)) * 2 - 1) * 1e-2
    seq = ''

    with torch.no_grad():
        if not pd.isnull(heavy):
            # if len(heavy) > 157:
            h_feature = []
            for i in range(math.ceil(len(heavy) / 157)):
                right_bound = min((i + 1) * 157, len(heavy) + 1)
                h = heavy[i * 157: right_bound]
                h_token_ft = model.heavy_ablang(h, mode='rescoding')
                h_token_ft = np.stack(h_token_ft)
                h_token_ft = torch.tensor(h_token_ft).squeeze(0)
                h_feature.append(h_token_ft)
            heavy_token_ft = torch.cat(h_feature, dim=0)

            # heavy_token_ft = model.heavy_ablang(heavy, mode='rescoding')
            # heavy_token_ft = np.stack(heavy_token_ft)
            # heavy_token_ft = torch.tensor(heavy_token_ft).squeeze(0)
            seq += heavy

        if not pd.isnull(light):
            l_feature = []
            for i in range(math.ceil(len(heavy) / 157)):
                right_bound = min((i + 1) * 157, len(light) + 1)
                l = light[i * 157: right_bound]
                l_token_ft = model.light_ablang(l, mode='rescoding')
                l_token_ft = np.stack(l_token_ft)
                l_token_ft = torch.tensor(l_token_ft).squeeze(0)
                l_feature.append(l_token_ft)
            light_token_ft = torch.cat(l_feature, dim=0)
            # light_token_ft = model.light_ablang(light, mode='rescoding')
            # light_token_ft = np.stack(light_token_ft)
            # light_token_ft = torch.tensor(light_token_ft).squeeze(0)
            seq += light

        if (not pd.isnull(light)) and (not pd.isnull(heavy)):
            ab_token_ft = torch.cat([heavy_token_ft, light_token_ft])
        elif (not pd.isnull(light)):
            ab_token_ft = light_token_ft
        elif (not pd.isnull(heavy)):
            ab_token_ft = heavy_token_ft
        else:
            print("All None")

    length = len(seq)
    if length < max_ab_len:
        padding = [padding_tensor for i in range(max_ab_len - length)]
        padding = torch.stack(padding)
        ab_token_ft = torch.cat([ab_token_ft, padding], dim=0).to('cpu')
    else:
        ab_token_ft = ab_token_ft.to('cpu')

    return ab_token_ft

def process_ag_token_ft(seq,encoder,tokenizer,device,max_ag_length):
    # device = torch.device('cuda:1')
    torch.manual_seed(42)
    padding_tensor = (torch.rand((480)) * 2 - 1) * 1e-2
    seq = re.sub(r"[XUZOBJ?*]", "", seq)
    length = len(seq)

    with torch.no_grad():
        encoder.eval()
        ag_feature = []
        for j in range(math.ceil(len(seq) / 510)):
            right_bound = min((j + 1) * 510, len(seq) + 1)
            l = seq[j * 510: right_bound]
            ag = tokenizer(l, truncation=True, padding="max_length", return_tensors="pt")
            ag['input_ids'] = ag['input_ids'].squeeze(1)  # bsz,length
            ag['attention_mask'] = ag['attention_mask'].squeeze(1)
            ag = {k: v.to(device) for k, v in ag.items()}

            # print(encoder(**ab).last_hidden_state.shape)
            ag_feature.append(((encoder(**ag)).last_hidden_state)[:, 1:len(l) + 1, :].to('cpu').squeeze(0))

        ag_token_ft = torch.cat(ag_feature, dim=0)

    padding = [padding_tensor for _ in range(max_ag_length - length)]

    if len(padding)>0:
        padding = torch.stack(padding)
        ag_token_ft = torch.cat([ag_token_ft, padding], dim=0).to('cpu')
    return ag_token_ft

def getAAfeature(ag_list,ab_list,gpu=0,max_ag_len=800, max_ab_len=110):
    """
    :param ab_list: [(heavy,light)]
    :param ag_list: [ag_seq]
    :return: 
    """
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    heavy_ablang = ablang.pretrained("heavy", device=device)
    heavy_ablang.freeze()
    light_ablang = ablang.pretrained("light", device=device)
    light_ablang.freeze()
    ab_encoder = Munch(
        heavy_ablang = heavy_ablang,
        light_ablang = light_ablang,
    )

    ag_encoder = EsmModel.from_pretrained("/home/u/data/xyh/project/interface_aware/networks/pretrained-ESM2")
    ag_encoder = ag_encoder.to(device)
    ab_token_ft = []
    ag_token_ft = []
    if max_ag_len==None:
        sequences = []
        for i in ag_list:
            i = re.sub(r"[XUZOBJ?*]", "", i)
            sequences.append(i)
        lengths = [len(i) for i in sequences]
        max_ag_len = max(lengths)
    if max_ag_len > 510:
        model_max_length = 512
    else:
        model_max_length = max_ag_len + 2
    tokenizer = EsmTokenizer.from_pretrained("/home/u/data/xyh/project/interface_aware/networks/pretrained-ESM2/vocab.txt", do_lower_case=False,
                                             model_max_length=model_max_length)
    if max_ab_len == None:
        sequences = []
        for heavy,light in ab_list:
            seq = ""
            if not pd.isnull(heavy):
                heavy = re.sub(r"[XUZOBJ?*]", "", heavy)
                seq+=heavy
            if not pd.isnull(light):
                light = re.sub(r"[XUZOBJ?*]", "", light)
                seq += light
            sequences.append(seq)
        lengths = [len(i) for i in sequences]
        max_ab_len = max(lengths)

    for heavy,light in ab_list:
        ab_token = process_ablang_token_ft(heavy,light,ab_encoder,max_ab_len=max_ab_len)
        ab_token_ft.append(ab_token)
    ab_token_ft = torch.stack(ab_token_ft)


    for seq in ag_list:
        ag_token = process_ag_token_ft(seq,ag_encoder,tokenizer,device,max_ag_length=max_ag_len)
        ag_token_ft.append(ag_token)
    ag_token_ft = torch.stack(ag_token_ft)

    return ag_token_ft.to(device),ab_token_ft.to(device)

def get_mask(ag_list,ab_list,gpu,max_ag_len=800, max_ab_len=110):
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    ag_masks,ab_masks = [],[]
    ag_len,ab_len = [],[]
    for ag_seq in ag_list:
        ag_mask = torch.zeros(max_ag_len)
        ag_len.append(len(ag_seq))
        ag_mask[:len(ag_seq)] = 1
        ag_masks.append(ag_mask)
    for heavy,light in ab_list:
        ab_seq = ""
        if not pd.isnull(heavy):
            heavy = re.sub(r"[XUZOBJ?*]", "", heavy)
            ab_seq += heavy
        if not pd.isnull(light):
            light = re.sub(r"[XUZOBJ?*]", "", light)
            ab_seq += light
        ab_mask = torch.zeros(max_ab_len)
        ab_len.append(len(ab_seq))
        ab_mask[:len(ab_seq)] = 1
        ab_masks.append(ab_mask)
    ag_masks = torch.stack(ag_masks,dim=0)
    ab_masks = torch.stack(ab_masks,dim=0)
    return ag_masks.to(device),ab_masks.to(device),ag_len,ab_len

if __name__ == '__main__':
    ag_list = [
        'AIVNAQPKCNPNLHYWTTQDEGAAIGLAWIPYFGPAAEGIYTEGLMHNQDGLICGLRQLANETTQALQLFLRATTELRTFSILNRKAIDFLLQRWGGTCHILGPDCCIEPHDW',
        'IDPYKEFGATVELLSFLPSDFFPSVRDLLDTAAALYRDALESPEHASPHHTALRQAILCWGDLMTLATWVGNTNVGLKFRQLLWFHISALTFGRETVLEYLVSFGVWIRTPPAYRPPNAPI'
    ]
    ab_list = [
        ('TFSNYWMNWVLEWVAEIRLKSNNYATHYAYCTRGNGNYRAMDYW','NIYSSLAWYQQPQLLVYSATILADGYCQHFWGTPYTF'),
        ('YAMIWVVEYIGIINTGGSASYAFCARTRGVNDAYEHAFDPW','SIGNALAWYQQPKLLISAGSNLASGYCLGTYSAIDRAF')
    ]
    # ab_list = [
    #     ('TFSNYWMNWVLEWVAEIRLKSNNYATHYAYCTRGNGNYRAMDYW',None),
    #     ('YAMIWVVEYIGIINTGGSASYAFCARTRGVNDAYEHAFDPW',None)
    # ]
    ag_token_ft,ab_token_ft = getAAfeature(ag_list, ab_list, gpu=0)
    ag_mask,ab_mask,ag_len,ab_len = get_mask(ag_list, ab_list,gpu=0)
    # device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

    model= load_model('./configs/SAbDab.yml',model_path='./save_model/new_SAbDab_0.pt',gpu=0)
    output = model.inference(ag_token_ft,ab_token_ft,ag_mask,ab_mask)
    print(output.score)










    
