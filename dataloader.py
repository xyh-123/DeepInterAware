from functools import reduce

import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.comm_utils import amino_seq_to_one_hot
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
    'original_data': 676,
    'AVIDa_hIL6':66,
    'sars_cov2/neutralising': 314, #441
    'SabDab': 110
}
max_antigen_len = {
    'HIV': 912,
    'AVIDa_hIL6':218,
    'sars_cov2/neutralising': 912,
    'SabDab': 800
}

def return_dataset(cfg,dataFolder):
    cfg.protein.max_antibody_len=max_antibody_len[cfg.set.dataset]
    cfg.protein.max_antigen_len=max_antigen_len[cfg.set.dataset]

    if cfg.set.dataset!='SabDab':
        dataset={
            'HIV':load_dataset,
            'AVIDa_hIL6':load_dataset,
            'sars_cov2/neutralising':load_dataset,
        }
        #
        return dataset[cfg.set.dataset](cfg, dataFolder)

class DeepAAIDataset(Dataset):
    def __init__(self, pair_data,dataFolder,cfg,unique_ab_id,unique_ag_id,all_unique_ab_id,all_unique_ag_id):
        """
        :param pair_data:
        :param max_ab_length: HIV:271 ag20:155
        :param max_ag_length:  HIV:912 ag20:460
        """
        self.pair_data = pair_data
        self.cfg=cfg
        self.max_ab_length = cfg.protein.max_antibody_len
        self.max_ag_length = cfg.protein.max_antigen_len
        self.unique_ab_id = unique_ab_id
        self.unique_ag_id = unique_ag_id
        self.all_unique_ab_id = all_unique_ab_id
        self.all_unique_ag_id = all_unique_ag_id
        self.ag_id = list(set(pair_data['ag_id'].to_list()))
        self.ab_seq = pd.read_csv(dataFolder + 'antibody.csv')  # 1786462
        self.ag_seq = pd.read_csv(dataFolder + 'antigen.csv')  # 1786462

        self.antibody_one_hot = amino_seq_to_one_hot(self.ab_seq.ab_seq.tolist(), 'antibody', antibody_max_len=self.max_ab_length,
                                                virus_max_len=self.max_ag_length)
        self.antigen_one_hot = amino_seq_to_one_hot(self.ag_seq.ag_seq.tolist(), 'virus', antibody_max_len=self.max_ab_length,
                                             virus_max_len=self.max_ag_length)

        if self.cfg.set.kmer:
            ab_kmer_mat = np.load(dataFolder + f'ab_kmer.npy')
            # print(ab_kmer_mat.shape)
            ag_kmer_mat = np.load(dataFolder + f'ag_kmer.npy')
            # print(ab_kmer_mat.shape)

            self.unique_ab_kmer = ab_kmer_mat[self.unique_ab_id].astype(np.float32)
            self.unique_ag_kmer = ag_kmer_mat[self.unique_ag_id].astype(np.float32)


    def __len__(self):
        return self.pair_data.shape[0]

    def __getitem__(self, index):
        ab_id=self.pair_data.iloc[index]['ab_id']
        ag_id=self.pair_data.iloc[index]['ag_id']
        ab_seq=self.ab_seq.iloc[ab_id]['ab_seq']
        ab_seq=ab_seq.replace(' ','')
        ab_seq = re.sub(r"[UZOBJ?*X]", "", ab_seq)
        # ab_cluster_id = self.ab_seq.iloc[ab_id]['cluster_id']


        ag_seq=self.ag_seq.iloc[ag_id]['ag_seq']
        ag_seq = ag_seq.replace(' ', '')
        ag_seq = re.sub(r"[UZOBJ?*X]", "", ag_seq)
        # ag_cluster_id = self.ag_seq.iloc[ag_id]['cluster_id']

        label = self.pair_data.iloc[index]['label']

        ab = self.antibody_one_hot[ab_id]
        ag = self.antigen_one_hot[ag_id]
        input_data = {'ab_id': ab_id, 'ag_id': ag_id, 'label': label, 'ab': ab, 'ag': ag,}
        if 'cluster_id' in self.ab_seq.columns:
            ab_cluster_id = self.ab_seq.iloc[ab_id]['cluster_id']
            input_data.update({'ab_cluster_id': ab_cluster_id})
        if 'CLS' in self.ag_seq.columns:
            ag_cluster_id = self.ag_seq.iloc[ag_id]['CLS']
            input_data.update({'ag_cluster_id': ag_cluster_id})
        elif 'cluster_id' in self.ag_seq.columns:
            ag_cluster_id = self.ag_seq.iloc[ag_id]['cluster_id']
            input_data.update({'ag_cluster_id': ag_cluster_id})

        return input_data

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
    if cfg.set.model_name == 'DeepAAI':
        seen_unique_ab_id = np.sort(train_data.ab_id.unique())
        seen_unique_ag_id = np.sort(train_data.ag_id.unique())
        unique_ab_id = np.sort(pair_data.ab_id.unique())
        unique_ag_id = np.sort(pair_data.ag_id.unique())
        train_dataset = DeepAAIDataset(train_data, dataFolder, cfg, seen_unique_ab_id, seen_unique_ag_id,
                                       unique_ab_id, unique_ag_id)
        test_dataset = DeepAAIDataset(test_data, dataFolder, cfg, unique_ab_id, unique_ag_id, unique_ab_id,
                                        unique_ag_id)
    else:
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

    if cfg.set.model_name == 'DeepAAI':

        seen_unique_ab_id = np.sort(train_data.ab_id.unique())
        seen_unique_ag_id = np.sort(train_data.ag_id.unique())

        # seen_unique_ab_id=reduce(np.union1d,[train_data.ab_id.unique(),val_data.ab_id.unique(),test_data.ab_id.unique()])
        # seen_unique_ag_id=reduce(np.union1d,[train_data.ag_id.unique(),val_data.ag_id.unique(),test_data.ag_id.unique()])
        seen_unique_ab_id=reduce(np.union1d,[train_data.ab_id.unique(),val_data.ab_id.unique()])
        seen_unique_ag_id=reduce(np.union1d,[train_data.ag_id.unique(),val_data.ag_id.unique()])
        # unseen_unique_ab_id = unseen_data.ab_id.unique()
        # unseen_unique_ag_id = unseen_data.ag_id.unique()
        unique_ab_id = np.sort(pair_data.ab_id.unique())
        unique_ag_id = np.sort(pair_data.ag_id.unique())
        train_dataset = DeepAAIDataset(train_data, dataFolder, cfg,seen_unique_ab_id,seen_unique_ag_id,unique_ab_id,unique_ag_id)
        val_dataset = DeepAAIDataset(val_data, dataFolder, cfg,seen_unique_ab_id,seen_unique_ag_id,unique_ab_id,unique_ag_id)
        # test_dataset = DeepAAIDataset(test_data, dataFolder, cfg,seen_unique_ab_id,seen_unique_ag_id,unique_ab_id,unique_ag_id)
        if isinstance(unseen_data, pd.DataFrame):
            unseen_dataset = DeepAAIDataset(unseen_data, dataFolder, cfg,unique_ab_id,unique_ag_id,unique_ab_id,unique_ag_id)
        else:
            unseen_dataset = None
    else:
        train_dataset = PairDataset(train_data, dataFolder, cfg)
        val_dataset = PairDataset(val_data, dataFolder, cfg)
        # test_dataset = PairDataset(test_data, dataFolder, cfg)
        if isinstance(unseen_data, pd.DataFrame):
            unseen_dataset = PairDataset(unseen_data, dataFolder, cfg)
        else:
            unseen_dataset = None

    return train_dataset, val_dataset, unseen_dataset

class BaselineDataset(Dataset):
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
        self.antibody_one_hot = amino_seq_to_one_hot(self.ab_seq.ab_seq.tolist(), self.max_ab_length)
        self.antigen_one_hot = amino_seq_to_one_hot(self.ag_seq.ag_seq.tolist(), self.max_ag_length)

        if self.cfg.set.CKSAAP:
            self.ab_CKSAAP_mat = np.load(dataFolder + f'ab_CKSAAP.npy')
            self.ag_CKSAAP_mat = np.load(dataFolder + f'ag_CKSAAP.npy')

        if self.cfg.set.phy:
            self.phy_id = json.load(open(dataFolder + 'phy_id.json', 'r'))
            self.ab_phy_mat = torch.load(dataFolder + f'ab_Phy.pt')
            self.ag_phy_mat = torch.load(dataFolder + f'ag_Phy.pt')

        if self.cfg.set.model_name in ['ESM2Ablang','SMAICFAblation','SMAICFAfter','SMAICF']:
            self.ab_token_ft_mat = torch.load(dataFolder + f'{cfg.set.ab_model}_token_ft.pt')
            self.ag_token_ft_mat = torch.load(dataFolder + f'ag_token_ft.pt')
        if self.cfg.set.model_name in ['ESM2AbLang','ESM2Antiberty']:
            self.ab_ft_mat = torch.load(dataFolder + f'{cfg.set.ab_model}_ft.pt')
            self.ag_ft_mat = torch.load(dataFolder + f'ag_ft.pt')

    def __len__(self):
        return self.pair_data.shape[0]

    def __getitem__(self, index):
        ab_id=self.pair_data.iloc[index]['ab_id']
        ag_id=self.pair_data.iloc[index]['ag_id']
        if self.cfg.set.dataset == 'SabDab':
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

        input_data = {'ab_id': ab_id, 'ag_id': ag_id, 'label': label,'ag_len':len(ag_seq),'ab_len':len(ab_seq),'idx':torch.LongTensor([index])}
        if 'cluster_id' in self.ab_seq.columns:
            ab_cluster_id = self.ab_seq.iloc[ab_id]['cluster_id']
            input_data.update({'ab_cluster_id': ab_cluster_id})
        if 'CLS' in self.ag_seq.columns:
            ag_cluster_id = self.ag_seq.iloc[ag_id]['CLS']
            input_data.update({'ag_cluster_id': ag_cluster_id})
        elif 'cluster_id' in self.ag_seq.columns:
            ag_cluster_id = self.ag_seq.iloc[ag_id]['cluster_id']
            input_data.update({'ag_cluster_id': ag_cluster_id})
        if self.cfg.set.model_name =='S3AI':
            input_data.update({'ab_seq': ab_seq, 'ag_seq': ag_seq})
        if self.cfg.set.model_name in ['ESM2AbLang']:
            ab_ft = self.ab_ft_mat[ab_id]
            ag_ft = self.ag_ft_mat[ag_id]
            input_data.update({'ab_ft': ab_ft, 'ag_ft': ag_ft})

        if self.cfg.set.model_name in ['ESM2Ablang','SMAICFAblation','SMAICFAfter','SMAICF']:
            ab_token_ft = self.ab_token_ft_mat[ab_id]
            ag_token_ft = self.ag_token_ft_mat[ag_id]
            ab_mask = torch.zeros(ab_token_ft.shape[0])
            ag_mask = torch.zeros(ag_token_ft.shape[0])
            ab_mask[:len(ab_seq)] = 1
            ag_mask[:len(ag_seq)] = 1
            attention_mask = torch.outer(ab_mask,ag_mask)
            input_data.update({'ab_token_ft': ab_token_ft, 'ag_token_ft': ag_token_ft,'ab_mask':ab_mask,'ag_mask':ag_mask,'attention_mask':attention_mask})
        else:
            ab = self.antibody_one_hot[ab_id]
            ag = self.antigen_one_hot[ag_id]
            input_data.update({ 'ab': ab, 'ag': ag,})

        if self.cfg.set.CKSAAP:
            ab_CKSAAP = self.ab_CKSAAP_mat[ab_id]
            ag_CKSAAP = self.ag_CKSAAP_mat[ag_id]
            ab_CKSAAP = ab_CKSAAP.astype(np.float32)
            ag_CKSAAP = ag_CKSAAP.astype(np.float32)
            input_data.update({'ab_CKSAAP': ab_CKSAAP, 'ag_CKSAAP': ag_CKSAAP, })

        if self.cfg.set.phy:
            ab_phy = self.ab_phy_mat[ab_id]  # antibody,ab_len * 5
            ag_phy = self.ag_phy_mat[ag_id]  # antigen,,ag_len * 5
            input_data.update({'ab_phy': ab_phy, 'ag_phy': ag_phy, })

        return input_data

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

        if self.cfg.set.phy:
            self.phy_id = json.load(open(dataFolder + 'phy_id.json', 'r'))
            self.ab_phy_mat = torch.load(dataFolder + f'ab_Phy.pt')
            self.ag_phy_mat = torch.load(dataFolder + f'ag_Phy.pt')

        if self.cfg.set.model_name in ['ESM2Ablang','SMAICFAblation','SMAICFAfter','SMAICF']:
            self.ab_token_ft_mat = torch.load(dataFolder + f'{cfg.set.ab_model}_token_ft.pt')
            self.ag_token_ft_mat = torch.load(dataFolder + f'ag_token_ft.pt')


    def __len__(self):
        return self.pair_data.shape[0]

    def __getitem__(self, index):
        ab_id=self.pair_data.iloc[index]['ab_id']
        ag_id=self.pair_data.iloc[index]['ag_id']
        if self.cfg.set.dataset == 'SabDab':
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

        if self.cfg.set.phy:
            ab_phy = self.ab_phy_mat[ab_id]  # antibody,ab_len * 5
            ag_phy = self.ag_phy_mat[ag_id]  # antigen,,ag_len * 5
            input_data.update({'ab_phy': ab_phy, 'ag_phy': ag_phy, })

        return input_data