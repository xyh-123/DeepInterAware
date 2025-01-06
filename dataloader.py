from functools import reduce

import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utools.comm_utils import amino_seq_to_one_hot
import json
import re
from torch.utils.data.sampler import WeightedRandomSampler

def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]

    return WeightedRandomSampler(weights, len(weights))

max_antibody_len={
    'HIV': 676, #676 for baseline
    'AVIDa_hIL6':66,
    'CoVAbDab': 676, #441
    'SAbDab': 110
}
max_antigen_len = {
    'HIV': 912,
    'AVIDa_hIL6':218,
    'CoVAbDab': 912,
    'SAbDab': 800
}

def return_dataset(cfg,dataFolder):
    cfg.protein.max_antibody_len=max_antibody_len[cfg.set.dataset]
    cfg.protein.max_antigen_len=max_antigen_len[cfg.set.dataset]

    if cfg.set.dataset!='SAbDab':
        dataset={
            'HIV':load_dataset,
            'AVIDa_hIL6':load_dataset,
            'CoVAbDab':load_dataset,
        }
        #
        return dataset[cfg.set.dataset](cfg, dataFolder)

def k_fold_dataset(cfg, dataFolder,k):
    """
    neutralising/ab_ag_pair.csv unique ab 7496 ,unique ag 49, positive 10564 ,negtive 13493
    binding/ab_ag_pair.csv      unique ab 11263,unique ag 72, positive 24795 ,negtive 3821
    :param cfg:
    :param dataFolder:
    :return:
    """
    pair_data = pd.read_csv((os.path.join(dataFolder, 'ab_ag_pair.csv')))
    train_data = pd.read_csv((os.path.join(dataFolder,'kfold', f'train_fold_{k}.csv')))
    test_data = pd.read_csv((os.path.join(dataFolder,'kfold', f'test_fold_{k}.csv')))
    unique_instance_label = pair_data.instance_label.unique()
    train_instance_label = train_data.instance_label.unique()
    print(f"train data {train_data.shape[0]}")
    print(f"test data {test_data.shape[0]}")

    test_instance_label = np.setdiff1d(unique_instance_label, train_instance_label)
    train_dataset = PairDataset(train_data, dataFolder, cfg)
    test_dataset = PairDataset(test_data, dataFolder, cfg)
    return train_dataset,test_dataset,train_instance_label,test_instance_label

def load_dataset(cfg, dataFolder):
    """
    neutralising/ab_ag_pair.csv unique ab 7496 ,unique ag 49, positive 10564 ,negtive 13493
    binding/ab_ag_pair.csv      unique ab 11263,unique ag 72, positive 24795 ,negtive 3821
    :param cfg:
    :param dataFolder:
    :return:
    """
    task = cfg.set.unseen_task
    pair_data = pd.read_csv(os.path.join(dataFolder, f'ab_ag_pair.csv'))
    unique_ab_id = pair_data.ab_id.unique()
    unique_ag_id = pair_data.ag_id.unique()
    train_data=pd.read_csv(os.path.join(dataFolder+f'{task}', f'train_seed_{cfg.solver.seed}.csv'))
    val_data=pd.read_csv(os.path.join(dataFolder+f'{task}', f'val_seed_{cfg.solver.seed}.csv'))
    # test_data=pd.read_csv(os.path.join(dataFolder+f'{task}', f'test_seed_{cfg.solver.seed}.csv'))
    print(f"Train number {train_data.shape[0]}")
    print(f"Valid number {val_data.shape[0]}")
    if task != 'transfer':
        unseen_data=pd.read_csv(os.path.join(dataFolder+f'{task}', f'unseen_seed_{cfg.solver.seed}.csv'))
        print(f"Unseen number {unseen_data.shape[0]}")
    else:
        unseen_data = None

    train_dataset = PairDataset(train_data, dataFolder, cfg)
    val_dataset = PairDataset(val_data, dataFolder, cfg)
    # test_dataset = PairDataset(test_data, dataFolder, cfg)
    if isinstance(unseen_data, pd.DataFrame):
        unseen_dataset = PairDataset(unseen_data, dataFolder, cfg)
    else:
        unseen_dataset = None

    return train_dataset, val_dataset, unseen_dataset

class PairDataset(Dataset):
    def __init__(self, pair_data,dataFolder,cfg):
        """
        :param pair_data:
        :param max_ab_length: HIV:271 ag20:155
        :param max_ag_length:  HIV:912 ag20:460
        """
        self.pair_data = pair_data
        self.cfg=cfg
        self.max_ab_length = cfg.protein.max_antibody_len
        self.max_ag_length = cfg.protein.max_antigen_len

        self.ag_id = list(set(pair_data['ag_id'].to_list()))
        self.label = pair_data['label'].to_list()
        self.ab_seq = pd.read_csv(dataFolder + 'antibody.csv')  # 1786462
        self.ag_seq = pd.read_csv(dataFolder + 'antigen.csv')  # 1786462

        self.ab_token_ft_mat = torch.load(dataFolder + f'{cfg.set.ab_model}_token_ft.pt')
        self.ag_token_ft_mat = torch.load(dataFolder + f'ag_token_ft.pt')


    def __len__(self):
        return self.pair_data.shape[0]

    def __getitem__(self, index):
        ab_id=self.pair_data.iloc[index]['ab_id']
        ag_id=self.pair_data.iloc[index]['ag_id']
        if self.cfg.set.dataset in ['SAbDab', 'AVIDa_hIL6']:
            key = 'ab_cdr'
        else:
            key = 'ab_seq'
        ab_seq=self.ab_seq.iloc[ab_id][key]
        ab_seq=ab_seq.replace(' ','')
        ab_seq = re.sub(r"[UZOBJ?*X]", "", ab_seq)

        ag_seq=self.ag_seq.iloc[ag_id]['ag_seq']
        ag_seq = ag_seq.replace(' ', '')
        ag_seq = re.sub(r"[UZOBJ?*X]", "", ag_seq)

        label = self.pair_data.iloc[index]['label']
        input_data = {'ab_id': ab_id, 'ag_id': ag_id, 'label': label,'ag_len':len(ag_seq),'ab_len':len(ab_seq)}

        ab_token_ft = self.ab_token_ft_mat[ab_id]
        ag_token_ft = self.ag_token_ft_mat[ag_id]
        ab_mask = torch.zeros(ab_token_ft.shape[0])
        ag_mask = torch.zeros(ag_token_ft.shape[0])
        ab_mask[:len(ab_seq)] = 1
        ag_mask[:len(ag_seq)] = 1
        attention_mask = torch.outer(ab_mask,ag_mask)
        input_data.update({'ab_token_ft': ab_token_ft, 'ag_token_ft': ag_token_ft,'ab_mask':ab_mask,'ag_mask':ag_mask,'attention_mask':attention_mask})

        return input_data

class SiteDataset(Dataset):
    def __init__(self, dataFolder,index):
        """
        :param pair_data:
        :param max_ab_length: HIV:271
        :param max_ag_length:  HIV:912
        """
        map_index = pd.read_csv(f'{dataFolder}/map_index.csv')
        self.map_index = map_index.iloc[index]
        self.index = index
        self.site_label = json.load(open(f'{dataFolder}/site_label.json', 'r'))

        self.site_ag_token = torch.load(f'{dataFolder}/site_ag_token.pt')[index]
        self.site_ab_token = torch.load(f'{dataFolder}/site_ab_token.pt')[index]

        self.site_ag_mask = torch.load(f'{dataFolder}/site_ag_mask.pt')[index]
        self.site_ab_mask = torch.load(f'{dataFolder}/site_ab_mask.pt')[index]

    def __len__(self):
        return len(self.map_index)

    def __getitem__(self, index):
        key = self.map_index.iloc[index]['key']
        ab_token_ft = self.site_ab_token[index]
        ag_token_ft = self.site_ag_token[index]

        ag_mask, ab_mask = self.site_ag_mask[index], self.site_ab_mask[index]
        max_ag_len = self.site_ag_mask.shape[-1]
        max_ab_len = self.site_ab_mask.shape[-1]

        site_label = self.site_label[key]
        epitope = site_label['epitope']
        ag_padding = [0 for _ in range(max_ag_len-len(epitope))]
        epitope += ag_padding

        paratope = site_label['paratope']
        ab_padding = [0 for _ in range(max_ab_len-len(paratope))]
        paratope+=ab_padding

        input_data = {'ab_token_ft': ab_token_ft, 'ag_token_ft': ag_token_ft,
                      'ab_mask':ab_mask,'ag_mask':ag_mask,'ag_label':torch.tensor(epitope),'ab_label':torch.tensor(paratope)}

        return input_data,key

class TestSiteDataset(Dataset):
    def __init__(self, dataFolder):
        """
        :param pair_data:
        :param max_ab_length: HIV:271
        :param max_ag_length:  HIV:912
        """
        self.map_index = json.load(open(f'{dataFolder}/map_index.json','r'))
        self.seq_data = json.load(open(f'{dataFolder}/all_seq.json', 'r'))

        self.site_ag_token = torch.load(f'{dataFolder}/site_ag_token.pt')
        self.site_ab_token = torch.load(f'{dataFolder}/site_ab_token.pt')

        self.site_ag_mask = torch.load(f'{dataFolder}/site_ag_mask.pt')
        self.site_ab_mask = torch.load(f'{dataFolder}/site_ab_mask.pt')

    def __len__(self):
        return len(self.map_index)

    def __getitem__(self, index):
        key = self.map_index[str(index)]

        ab_token_ft = self.site_ab_token[index]
        ag_token_ft = self.site_ag_token[index]

        ag_mask, ab_mask = self.site_ag_mask[index], self.site_ab_mask[index]
        seq_dict = self.seq_data[key]
        ag_seq = seq_dict['ag_seq']
        ab_seq = seq_dict['ab_cdr']
        ab_len = len(ab_seq)
        ag_len = len(ag_seq)
        input_data = {'ag_len':ag_len,'ab_len':ab_len,
                      'ab_token_ft': ab_token_ft, 'ag_token_ft': ag_token_ft,
                      'ab_mask':ab_mask,'ag_mask':ag_mask}

        return input_data,key

class MuteDataset(Dataset):
    def __init__(self, index,dataset,data_path=f'{os.getcwd()}/data'):
        """
        :param pair_data:
        :param max_ab_length: HIV:271 ag20:155
        :param max_ag_length:  HIV:912 ag20:460
        """
        path = f'{data_path}/{dataset}/'
        self.wt_ag_token_fts = torch.load(f'{path}/wt_ag_token_ft.pt')[index]
        self.wt_ab_token_fts = torch.load(f'{path}/wt_ab_token_ft.pt')[index]
        self.wt_ag_masks = torch.load(f'{path}/wt_ag_masks.pt')[index]
        self.wt_ab_masks = torch.load(f'{path}/wt_ab_masks.pt')[index]

        self.mu_ag_token_fts = torch.load(f'{path}/wt_ag_token_ft.pt')[index]
        self.mu_ab_token_fts = torch.load(f'{path}/wt_ab_token_ft.pt')[index]
        self.mu_ag_masks = torch.load(f'{path}/mu_ag_masks.pt')[index]
        self.mu_ab_masks = torch.load(f'{path}/mu_ab_masks.pt')[index]

        self.label = torch.load(f'{path}/label.pt')[index]
        self.key_id = torch.load(f'{path}/key_id.pt')[index]
    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, index):
        label = self.label[index]
        key_id = self.key_id[index]

        wt_ag_token_ft = self.wt_ag_token_fts[index]
        wt_ab_token_ft = self.wt_ab_token_fts[index]
        wt_ag_mask = self.wt_ag_masks[index]
        wt_ab_mask = self.wt_ab_masks[index]

        mu_ag_token_ft = self.mu_ag_token_fts[index]
        mu_ab_token_ft = self.mu_ab_token_fts[index]
        mu_ag_mask = self.mu_ag_masks[index]
        mu_ab_mask = self.mu_ab_masks[index]

        wt_input_data = {'ag_token_ft': wt_ag_token_ft, 'ab_token_ft': wt_ab_token_ft,'ag_mask': wt_ag_mask,'ab_mask': wt_ab_mask}
        mu_input_data = {'ag_token_ft': mu_ag_token_ft, 'ab_token_ft': mu_ab_token_ft,'ag_mask': mu_ag_mask,'ab_mask': mu_ab_mask}

        return wt_input_data,mu_input_data,label,key_id