import argparse

import numpy as np
import re
import torch
from itertools import chain, product
import pandas as pd
import os
from feature_encoder import get_esm2_Seq_feature,get_ablang_Seq_feature,get_antiberty_Seq_feature
from utools.cdr_extract import extract_CDR

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class kmer_featurization:

  def __init__(self, k):
    """
    seqs: a list of DNA sequences
    k: the "k" in k-mer
    """
    self.k = k
    self.letters =  ["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
    self.multiplyBy = 20 ** np.arange(k-1, -1, -1) # the multiplying number for each digit position in the k-number system
    self.n = 20**k # number of possible k-mers

  def obtain_kmer_feature_for_a_list_of_sequences(self, seqs, write_number_of_occurrences=False):
    """
    Given a list of m DNA sequences, return a 2-d array with shape (m, 4**k) for the 1-hot representation of the kmer features.

    Args:
      write_number_of_occurrences:
        a boolean. If False, then in the 1-hot representation, the percentage of the occurrence of a kmer will be recorded; otherwise the number of occurrences will be recorded. Default False.
    """
    kmer_features = []
    for seq in seqs:
      seq = re.sub(r"[-UZOBJ?*X]", "", seq)
      this_kmer_feature = self.obtain_kmer_feature_for_one_sequence(seq.upper(), write_number_of_occurrences=write_number_of_occurrences)
      kmer_features.append(this_kmer_feature)

    # print(kmer_features)
    # kmer_features = np.array(kmer_features)
    kmer_features = torch.stack(kmer_features)

    return kmer_features

  def obtain_kmer_feature_for_one_sequence(self, seq, write_number_of_occurrences=False):
    """
    Given a DNA sequence, return the 1-hot representation of its kmer feature.

    Args:
      seq:
        a string, a DNA sequence
      write_number_of_occurrences:
        a boolean. If False, then in the 1-hot representation, the percentage of the occurrence of a kmer will be recorded; otherwise the number of occurrences will be recorded. Default False.
    """
    number_of_kmers = len(seq) - self.k + 1

    kmer_feature = torch.zeros(self.n)

    for i in range(number_of_kmers):
      this_kmer = seq[i:(i+self.k)]
      this_numbering = self.kmer_numbering_for_one_kmer(this_kmer)
      kmer_feature[this_numbering] += 1

    if not write_number_of_occurrences:
      kmer_feature = kmer_feature / number_of_kmers

    return kmer_feature

  def kmer_numbering_for_one_kmer(self, kmer):
    """
    Given a k-mer, return its numbering (the 0-based position in 1-hot representation)
    """
    digits = []
    for letter in kmer:
      digits.append(self.letters.index(letter))

    digits = np.array(digits)

    numbering = (digits * self.multiplyBy).sum()

    return numbering

AA = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
DP = list(product(AA, AA))
DP_list = []
for i in DP:
    DP_list.append(str(i[0]) + str(i[1]))

def returnCKSAAPcode(query_seq, k):
    code_final = []
    for turns in range(k + 1):
        DP_dic = {}
        code = []
        # code_order = []
        for i in DP_list:
            DP_dic[i] = 0
        for i in range(len(query_seq) - turns - 1):
            tmp_dp_1 = query_seq[i]
            tmp_dp_2 = query_seq[i + turns + 1]
            tmp_dp = tmp_dp_1 + tmp_dp_2
            if tmp_dp in DP_dic.keys():
                DP_dic[tmp_dp] += 1
            else:
                DP_dic[tmp_dp] = 1
        for i, j in DP_dic.items():
            code.append(j / (len(query_seq) - turns - 1))
        # for i in AAindex_list:
        #     code_order.append(code[DP_list.index(i)])
        # code_final+=code_order
        code_final += code
    return code_final

def gen_CKSAAP(seq_list):
    cksaap=[]
    for i in seq_list:
        i = i.replace(' ','')
        i = re.sub(r"[XUZOBJ?*]", "", i)
        try:
            i_cksaap=np.array(returnCKSAAPcode(i,3)).reshape(4,20,20)
            cksaap.append(i_cksaap)
        except:
            print('cksaap')
            print(i)
    cksaap = np.stack(cksaap)
    return cksaap

def process_CKSAAP(data_path):
    antibody=pd.read_csv(data_path+'antibody.csv')
    antigen = pd.read_csv(data_path + 'antigen.csv')
    ab_cksaap,ag_cksaap=[],[]
    for i in antibody.ab_seq.tolist():
        i = re.sub(r"[XUZOBJ?*]", "", i)
        try:
            i_cksaap=np.array(returnCKSAAPcode(i,3)).reshape(4,20,20)
            ab_cksaap.append(i_cksaap)
        except:
            print(i)

    ab_cksaap=np.stack(ab_cksaap)
    for i in antigen.ag_seq.tolist():
        i = re.sub(r"[XUZOBJ?*]", "", i)
        try:
            i_cksaap=np.array(returnCKSAAPcode(i,3)).reshape(4,20,20)
            ag_cksaap.append(i_cksaap)
        except:
            print(i)

    ag_cksaap = np.stack(ag_cksaap)
    print(ab_cksaap.shape)
    print(ag_cksaap.shape)
    np.save(data_path+'ab_CKSAAP.npy',ab_cksaap)
    np.save(data_path+'ag_CKSAAP.npy',ag_cksaap)

def process_Kmer(datafolder):
    ab=pd.read_csv(datafolder+'antibody.csv')
    ag=pd.read_csv(datafolder+'antigen.csv')
    ab_list=ab['ab_seq'].to_list()
    ag_list=ag['ag_seq'].to_list()
    ab_length=[len(i) for i in ab_list]
    ag_length=[len(i) for i in ag_list]
    max_ab_length = np.array(ab_length).max()
    max_ag_length = np.array(ag_length).max()
    print(f"max ab length {max_ab_length}")
    print(f"max ag length {max_ag_length}")
    ab_number = len(ab_list)
    ag_number = len(ag_list)
    # with open(datafolder+'ab_id_seq.json','r') as f:
    #     antibody=json.load(f)
    # with open(datafolder+'ag_id_seq.json','r') as f:
    #     antigen=json.load(f)
    ab_kmer_mat,ag_kmer_mat = [],[]
    for k in range(1,4):
        obj = kmer_featurization(k)  # initialize a kmer_featurization object
        ab_kmer=obj.obtain_kmer_feature_for_a_list_of_sequences(ab_list, write_number_of_occurrences=False)
        ag_kmer=obj.obtain_kmer_feature_for_a_list_of_sequences(ag_list, write_number_of_occurrences=False)
        # print(ab_kmer.shape)
        ab_kmer_mat.append(ab_kmer)
        ag_kmer_mat.append(ag_kmer)

    ab_kmer_mat = torch.cat(ab_kmer_mat,dim=-1)
    ag_kmer_mat = torch.cat(ag_kmer_mat,dim=-1)

    torch.save(ab_kmer_mat,datafolder+'ab_kmer.pt')
    torch.save(ag_kmer_mat,datafolder+'ag_kmer.pt')

    print(f"ab number {ab_number}") #310034
    print(f"ag number {ag_number}") #50
    print(ab_kmer_mat.shape)  # 296
    print(ag_kmer_mat.shape)  # 296

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="antibody-antigen binding affinity prediction")
    parser.add_argument('--data_path', default=f'{project_dir}/data/', type=str, metavar='TASK',
                        help='data path')
    parser.add_argument('--dataset', default=f'HIV', type=str, metavar='TASK',
                        help='data path')
    parser.add_argument('--use_cdr', action='store_true')
    parser.add_argument('--gpu', default=0, type=int, metavar='S', help='run GPU number')
    args = parser.parse_args()

    # project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datafolder = os.path.join(args.data_path,'HIV/')
    # process_Kmer(datafolder)
    # process_CKSAAP(datafolder)

    print(args.dataset)
    antibody = pd.read_csv(os.path.join(args.data_path, args.dataset, 'antibody.csv'))
    antigen = pd.read_csv(os.path.join(args.data_path, args.dataset, 'antigen.csv'))
    if args.dataset!='AVIDa_hIL6':
        heavy = antibody['heavy'].to_list()
        light = antibody['light'].to_list()
    else:
        heavy = antibody['ab_seq'].to_list()
        light = [np.nan for i in range(len(heavy))]

    ab_list = []
    if args.use_cdr:
        heavy = antibody['heavy_cdr'].to_list()
        light = antibody['light_cdr'].to_list()

    ab_list = zip(heavy, light)
    ag_list = antigen['ag_seq'].to_list()

    ag_ft=get_esm2_Seq_feature(ag_list,args.gpu)

    # print(ab_list)
    antiberty_ft = get_antiberty_Seq_feature(ab_list,args.gpu)
    ab_list = zip(heavy, light)
    ablang_ft = get_ablang_Seq_feature(ab_list,args.gpu)

    print(ag_ft.shape,ablang_ft.shape,antiberty_ft.shape)
    # print(antiberty_ft.shape)
    torch.save(ag_ft, datafolder + 'ag_ft.pt')
    torch.save(ablang_ft, datafolder + 'ablang_ft.pt')
    torch.save(antiberty_ft, datafolder + 'antiberty_ft.pt')