from biopandas.pdb import PandasPdb
import requests
import numpy as np
from utils.cdr_extract import extract_CDR
import torch
import torch.nn.functional as F
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torchmetrics.classification import BinaryRecall

res_codes = {
    'ALA':'A','CYS':'C','ASP':'D','GLU':'E',
    'PHE':'F','GLY':'G','HIS':'H','LYS':'K',
    'ILE':'I','LEU':'L','MET':'M','ASN':'N',
    'PRO':'P','GLN':'Q','ARG':'R','SER':'S',
    'THR':'T','VAL':'V','TYR':'Y','TRP':'W'}

def dis_pairs(coord_1,coord_2):
    coord_1_x = coord_1[-3]
    coord_1_y = coord_1[-2]
    coord_1_z = coord_1[-1]
    coord_2_x = coord_2[-3]
    coord_2_y = coord_2[-2]
    coord_2_z = coord_2[-1]
    distance = np.sqrt((float(coord_1_x) - float(coord_2_x)) ** 2 + (float(coord_1_y) - float(coord_2_y)) ** 2 + (float(coord_1_z) - float(coord_2_z)) ** 2)
    return distance

#标记label
def get_labels(coord_AG,coord_H,coord_L,coord_H_res_id,coord_L_res_id,coord_AG_res_id):
    label_AG = [0 for i in range(len(coord_AG_res_id))]
    label_H = [0 for i in range(len(coord_H_res_id))]
    label_L = [0 for i in range(len(coord_L_res_id))]
    source_AGH = []
    target_AGH = []
    source_AGL = []
    target_AGL = []
    for i in range(len(coord_AG)):
        for j in range(len(coord_H)):
            # if(dis_pairs(coord_AG[i],coord_H[j]) <= 4.5):
            if(dis_pairs(coord_AG[i],coord_H[j]) <= 5.0):
                label_AG[coord_AG_res_id.index(coord_AG[i][0])] = 1
                label_H[coord_H_res_id.index(coord_H[j][0])] = 1
                if(coord_AG_res_id.index(coord_AG[i][0]) in source_AGH and coord_H_res_id.index(coord_H[j][0]) in target_AGH):
                    continue
                else:
                    source_AGH.append(coord_AG_res_id.index(coord_AG[i][0]))
                    target_AGH.append(coord_H_res_id.index(coord_H[j][0]))
        for k in range(len(coord_L)):
            # if(dis_pairs(coord_AG[i],coord_L[k]) <= 4.5):
            if(dis_pairs(coord_AG[i],coord_L[k]) <= 5.0):
                label_AG[coord_AG_res_id.index(coord_AG[i][0])] = 1
                label_L[coord_L_res_id.index(coord_L[k][0])] = 1
                if (coord_AG_res_id.index(coord_AG[i][0]) in source_AGL and coord_L_res_id.index(coord_L[k][0]) in target_AGL):
                    continue
                else:
                    source_AGL.append(coord_AG_res_id.index(coord_AG[i][0]))
                    target_AGL.append(coord_L_res_id.index(coord_L[k][0]))
    label_AGH = [source_AGH, target_AGH]
    label_AGL = [source_AGL, target_AGL]

    return label_AG,label_H,label_L,label_AGH,label_AGL

def coord(ATOM,Hchain,Lchain,AGchain):

    coord_H = []    # 存储原子坐标
    coord_L = []
    coord_AG = []
    coord_H_res_id=["FIRST"]
    coord_L_res_id=["FIRST"]
    coord_AG_res_id=["FIRST"]
    coord_H_res=[]
    coord_L_res =[]
    coord_AG_res =[]
    res_id_before="none"
    # 提前记录所有残基的去重id
    res_id_H=[]
    res_id_L=[]
    res_id_AG=[]
    for row in range(ATOM.shape[0]):
        """遍历所有原子标记原子氨基酸"""
        row_info = (np.array(ATOM.iloc[row, :]).tolist())
        res_id = row_info[7] + str(row_info[8]) + str(row_info[5])  # 该残基的独一无二标记
        # ['ATOM', 4148, '', 'OXT', '', 'VAL', '', 'H', 225, '', '', 87.753, 61.481, -11.977, 1.0, 87.78, '', '', 'O', nan, 4650]
        if (row_info[7] == Hchain):
            if (row_info[3] == "CA" or row_info[3] == "CB"):
                res_id_H.append(res_id)
        elif (row_info[7] == Lchain):
            if (row_info[3] == "CA" or row_info[3] == "CB"):
                res_id_L.append(res_id)
        elif (row_info[7] in AGchain):
            if (row_info[3] == "CA" or row_info[3] == "CB"):
                res_id_AG.append(res_id)
    res_id_H = list(set(res_id_H))
    res_id_L = list(set(res_id_L))
    res_id_AG = list(set(res_id_AG))
    # print("res_id_L:",res_id_L)

    for row in range(ATOM.shape[0]):
        row_info=(np.array(ATOM.iloc[row,:]).tolist())
        # ['ATOM', 4148, '', 'OXT', '', 'VAL', '', 'H', 225, '', '', 87.753, 61.481, -11.977, 1.0, 87.78, '', '', 'O', nan, 4650]
        res_id = row_info[7] + str(row_info[8]) + str(row_info[5])  # 该残基的独一无二标记
        res_x = row_info[11]
        res_y = row_info[12]
        res_z = row_info[13]
        tag_coord_H_res_i = 0  # 标注
        tag_coord_L_res_i = 0
        tag_coord_AG_res_i = 0

        if (row_info[7] == Hchain):
            # 记录基本信息
            res_n = res_codes[row_info[5]]  # 残基名称
            # 属于重链
            if(res_id in res_id_H):
                coord_H.append([res_id,res_n, res_x, res_y, res_z]) # 原子坐标
                # if(res_id not in coord_H_res_id and coord_H_res_id[-1] != res_id):
                if (coord_H_res_id[-1] != res_id):
                    coord_H_res_id.append(res_id)
                if (row_info[3] == "CA" or row_info[3] == "CB"):
                    coord_H_res_i=[res_x, res_y, res_z]
                    tag_coord_H_res_i = 1
        if (row_info[7] == Lchain):
            # 记录基本信息
            res_n = res_codes[row_info[5]]  # 残基名称
            # 属于轻链
            if (res_id in res_id_L):
                coord_L.append([res_id,res_n, res_x, res_y, res_z])
                # if (res_id not in coord_L_res_id):
                if (coord_L_res_id[-1] != res_id):
                    coord_L_res_id.append(res_id)
                if (row_info[3] == "CA" or row_info[3] == "CB"):
                    coord_L_res_i = [res_x, res_y, res_z]
                    tag_coord_L_res_i = 1
        if (row_info[7] in AGchain):
            # 记录基本信息
            res_n = res_codes[row_info[5]]  # 残基名称
            # 属于抗原
            if (res_id in res_id_AG):
                coord_AG.append([res_id,res_n, res_x, res_y, res_z])
                # if (res_id not in coord_AG_res_id and coord_AG_res_id[-1] != res_id):
                if (coord_AG_res_id[-1] != res_id):
                    coord_AG_res_id.append(res_id)
                if (row_info[3] == "CA" or row_info[3] == "CB"):
                    coord_AG_res_i = [res_x, res_y, res_z]
                    tag_coord_AG_res_i = 1
        #更新res坐标
        if(res_id_before != res_id):
            if (tag_coord_AG_res_i == 1):
                coord_AG_res.append(coord_AG_res_i)
                res_id_before = res_id
            elif (tag_coord_H_res_i == 1):
                # print(res_id)
                coord_H_res.append(coord_H_res_i)
                res_id_before = res_id
            elif (tag_coord_L_res_i == 1):
                coord_L_res.append(coord_L_res_i)
                res_id_before = res_id
                # print(res_id)

    coord_H_res_id=coord_H_res_id[1:]
    coord_L_res_id = coord_L_res_id[1:]
    coord_AG_res_id = coord_AG_res_id[1:]

    if(len(coord_H_res) != len(coord_H_res_id)):
        print("complex_dict['coord_H'] ERROR!")
    if (len(coord_L_res) != len(coord_L_res_id)):
        print("complex_dict['coord_L'] ERROR!")
    if (len(coord_AG_res) != len(coord_AG_res_id)):
        print("complex_dict['coord_AG'] ERROR!")
    # print("===============coord_L_res_id===============")
    # for i in range(len(coord_L_res_id)):
    #     print(coord_L_res_id[i])

    return coord_H,coord_L,coord_AG,coord_H_res_id,coord_L_res_id,coord_AG_res_id,coord_H_res,coord_L_res,coord_AG_res

def id2fasta2dict(pdb_id,Hchain,Lchain,AGchain):
    try:
        ATOM=PandasPdb().read_pdb(f'/mnt/xyh/project/gaai/data/SabDab/pdb/{pdb_id}.pdb').df['ATOM']
    except:
        pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

        # 发起HTTP请求并下载PDB文件
        response = requests.get(pdb_url)

        # 检查请求是否成功
        if response.status_code == 200:
            # 构建PDB文件的本地路径
            pdb_filename = f"/mnt/xyh/project/gaai/data/SabDab/pdb/{pdb_id}.pdb"

            # 写入PDB文件到本地
            with open(pdb_filename, 'wb') as f:
                f.write(response.content)

            print(f"Downloaded {pdb_id}.pdb")
            ATOM = PandasPdb().read_pdb(f'/mnt/xyh/project/gaai/data/SabDab/pdb/{pdb_id}.pdb').df['ATOM']
        else:
            pdb_url = f"https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/pdb/{pdb_id}/?raw=true"

            # 发起HTTP请求并下载PDB文件
            response = requests.get(pdb_url)
            if response.status_code == 200:
                pdb_filename = f"/mnt/xyh/project/gaai/data/SabDab/pdb/{pdb_id}.pdb"

                # 写入PDB文件到本地
                with open(pdb_filename, 'wb') as f:
                    f.write(response.content)

                print(f"Downloaded {pdb_id}.pdb")
                ATOM = PandasPdb().read_pdb(f'/mnt/xyh/project/gaai/data/SabDab/pdb/{pdb_id}.pdb').df['ATOM']
            else:
                print(f"Downloaded {pdb_id} error")
                ppdb = PandasPdb().fetch_pdb(pdb_id)
                ATOM=ppdb.df['ATOM']
    Hchain=Hchain
    Lchain=Lchain
    AGchain=AGchain
    coord_H, coord_L, coord_AG, coord_H_res_id, coord_L_res_id, coord_AG_res_id, coord_H_res, coord_L_res, coord_AG_res=coord(ATOM,Hchain,Lchain,AGchain)
    res_H=list(map(lambda x: res_codes[x[-3:]],coord_H_res_id))
    res_L=list(map(lambda x: res_codes[x[-3:]],coord_L_res_id))
    res_AG=list(map(lambda x: res_codes[x[-3:]],coord_AG_res_id))
    res_H=''.join(res_H)
    res_L=''.join(res_L)
    res_AG=''.join(res_AG)
    label_AG,label_H,label_L,label_AGH,label_AGL=get_labels(coord_AG,coord_H,coord_L,coord_H_res_id,coord_L_res_id,coord_AG_res_id)
    #创建字典
    full_seq = {"res_H":res_H,"res_L":res_L,"res_AG":res_AG}
    full_label = {"label_H":label_H,"label_L":label_L,"label_AG":label_AG,'label_AGH':label_AGH,'label_AGL':label_AGL}

    return full_seq,full_label

def site_process(res_H,label_H,res_L,label_L):
    sites = []
    ab_info,ab_cdr = extract_CDR(res_H,res_L)
    if res_H != None:
        h_site = []
        for i in range(1,4):
            cdr_index, cdr_range = ab_info[f'H_cdr{i}_range']
            h_site += label_H[cdr_index:cdr_range]
        sites += h_site

    if res_L != None:
        l_site = []
        for i in range(1,4):
            cdr_index, cdr_range = ab_info[f'L_cdr{i}_range']
            l_site += label_L[cdr_index:cdr_range]
        sites += l_site

    return sites,ab_info,ab_cdr

def process_interaction_pattern(ab_info,label_AGH,label_AGL):
    len_h_cdr = len(ab_info['H_cdr'])
    def process_chain(chain):
        if chain == 'H':
            pair_label = label_AGH
        else:
            pair_label = label_AGL
        new_pair_label = []
        cdr1_start, cdr1_end = ab_info[f'{chain}_cdr1_range']
        cdr2_start, cdr2_end = ab_info[f'{chain}_cdr2_range']
        cdr3_start, cdr3_end = ab_info[f'{chain}_cdr3_range']

        cdr1_len = cdr1_end - cdr1_start
        cdr2_len = cdr2_end - cdr2_start
        # cdr3_len = cdr3_end - cdr3_start

        for i, j in zip(pair_label[0], pair_label[1]):
            if j >= cdr1_start and j <= cdr1_end:
                j = j - cdr1_start
            elif j >= cdr2_start and j <= cdr2_end:
                j = j - cdr2_start+cdr1_len
            elif j >= cdr3_start and j <= cdr3_end:
                j = j - cdr3_start+cdr1_len+cdr2_len
            else:
                #不在CDR区
                continue
            if chain == 'H':
                new_pair_label.append((j, i))
            else:
                new_pair_label.append((len_h_cdr+j, i))
        return new_pair_label

    new_label_AGH = process_chain('H')
    if 'L_cdr' in ab_info:
        new_label_AGL = process_chain('L')
        pair_label = new_label_AGH + new_label_AGL
    else:
        pair_label = new_label_AGH

    pair_label.sort(key=lambda x:x[0])

    # h_cdr1_range = ab_info['H_cdr1_range']
    # h_cdr2_range = ab_info['H_cdr2_range']
    # h_cdr3_range = ab_info['H_cdr3_range']
    #
    # l_cdr1_range = ab_info['L_cdr1_range']
    # l_cdr2_range = ab_info['L_cdr2_range']
    # l_cdr3_range = ab_info['L_cdr3_range']

    # h_cdr1_start,h_cdr1_end = ab_info['H_cdr1_range']
    # h_cdr2_start,h_cdr2_end = ab_info['H_cdr2_range']
    # h_cdr3_start,h_cdr3_end = ab_info['H_cdr3_range']
    #
    # l_cdr1_start, l_cdr1_end = ab_info['L_cdr1_range']
    # l_cdr2_start, l_cdr2_end = ab_info['L_cdr2_range']
    # l_cdr3_start, l_cdr3_end = ab_info['L_cdr3_range']
    #
    # h_cdr1_len = h_cdr1_end-h_cdr1_start
    # h_cdr2_len = h_cdr2_end-h_cdr2_start
    # h_cdr3_len = h_cdr3_end-h_cdr3_start
    #
    # l_cdr1_len = l_cdr1_end-l_cdr1_start
    # l_cdr2_len = l_cdr2_end-l_cdr2_start
    # l_cdr3_len = l_cdr3_end-l_cdr3_start
    #
    # len_h_cdr =  h_cdr1_len+ h_cdr2_len+ h_cdr3_len
    #
    # print(f"h_cdr {h_cdr1_len,h_cdr2_len,h_cdr3_len}")
    #
    #
    # def h_mapping_func(index):
    #     if index<h_cdr1_start or index>h_cdr3_end:
    #         return -1
    #     if index>=h_cdr1_start and index<=h_cdr1_end:
    #         #h_cdr1
    #         return index-h_cdr1_start
    #     elif index>=h_cdr2_start and index<=h_cdr2_end:
    #         return index-h_cdr2_start+h_cdr1_len
    #     else:
    #         return index-h_cdr3_start+h_cdr1_len+h_cdr2_len
    #
    # def l_mapping_func(index):
    #     if index<l_cdr1_start or index>l_cdr3_end:
    #         return -1
    #     if index>=l_cdr1_start and index<=l_cdr1_end:
    #         #h_cdr1
    #         return index-l_cdr1_start
    #     elif index>=l_cdr2_start and index<=l_cdr2_end:
    #         return index-l_cdr2_start+l_cdr1_len
    #     else:
    #         return index-l_cdr3_start+l_cdr1_len+l_cdr2_len
    #
    # source_AGH, target_AGH = label_AGH[0],label_AGH[1]
    # source_AGL, target_AGL = label_AGL[0],label_AGL[1]
    #
    # for index in range(len(source_AGH)):
    #     target = h_mapping_func(target_AGH[index])
    #     if target==-1:
    #         source_AGH[index] = target
    #
    #     target_AGH[index] = target
    #
    # for index in range(len(source_AGL)):
    #     target = l_mapping_func(target_AGL[index])
    #     if target==-1:
    #         source_AGL[index] = target
    #
    #     target_AGL[index] = len_h_cdr+target
    #
    # source_AGH = list(filter(lambda x: x != -1, source_AGH))
    # target_AGH = list(filter(lambda x: x != -1, target_AGH))
    #
    # source_AGL = list(filter(lambda x: x != -1, source_AGL))
    # target_AGL = list(filter(lambda x: x != -1, target_AGL))

    # new_label_AGH = zip(target_AGH,source_AGH)
    # new_label_AGL = zip(target_AGL,source_AGL)

    # new_label_AGH = zip(source_AGH,target_AGH)
    # new_label_AGL = zip(source_AGL,target_AGL)
    #
    # print(list(new_label_AGH))
    # print(list(new_label_AGL))
    #
    # new_pair_label = list(new_label_AGH) + list(new_label_AGL)
    # new_pair_label.sort(key=lambda x: x[0])
    # print(new_pair_label)
    return pair_label

def get_binding_site(pdb_id,heavy_chain,light_chain,antigen_chain):
    full_seq,full_label = id2fasta2dict(pdb_id,heavy_chain,light_chain,antigen_chain) #(Ag,H),(Ag,L),坐标从0开始
    # print(full_seq)
    # print(full_label)
    ab_sites,ab_info,ab_cdr = site_process(full_seq['res_H'],full_label['label_H'],full_seq['res_L'],full_label['label_L'])
    # print(ab_info)
    # print(sites)
    pair_label = process_interaction_pattern(ab_info, full_label['label_AGH'], full_label['label_AGL'])
    label_dict = {
        'epitope':full_label['label_AG'],
        'paratope':ab_sites,
        'paratope-epitope':pair_label
    }
    seq_dict = {
        'H_cdr':ab_info['H_cdr'],
        'ag_seq':full_seq['res_AG'],
        'ab_cdr':ab_cdr
    }
    if 'L_cdr' in ab_info:
        seq_dict['L_cdr']=ab_info['L_cdr']

    # print(seq_dict)
    # print(label_dict)
    return full_seq,full_label,seq_dict,ab_info,label_dict



def norm(att):
    att_mean = torch.mean(att, dim=-1, keepdim=True)
    att_std  = torch.std(att, dim=-1, keepdim=True)
    att_norm = (att - att_mean) / att_std
    return att_norm

def calculate_binding_pair_recall(filter_att,pair_label,ag_len,ab_len,threshold):
    att_label = torch.zeros(ab_len,ag_len)
    for i, j in pair_label:
        att_label[i][j] = 1

    att = []
    for i in filter_att:
        att.append(norm(i))
    filter_att = torch.stack(att)
    filter_att = torch.sigmoid(filter_att).detach().to('cpu')
    filter_att = torch.ceil(filter_att - threshold).long()

    # 计算 True Positives (TP) 和 False Negatives (FN)
    TP = torch.sum((att_label == 1) & (filter_att == 1))
    FN = torch.sum((att_label == 1) & (filter_att == 0))

    # 计算召回率
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return recall.item()

def draw(data,path,vmax = 0.35):
    # color = ["#3399CC", "#FFFAF9", "#FF0000"]
    # color = ["#3399CC","#8ADD7A",  "#FF0000"]
    cmap = colors.LinearSegmentedColormap.from_list("brw", ["#3399CC","#FFFAF9",  "#FF0000"], N=256)
    # nodes = [0.0, 0.5, 1]
    # cmap = colors.LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, color)), N=256)
    #
    plt.figure(figsize=(80, 20))
    ax = sns.heatmap(data, cmap=cmap, cbar=False, square=True, linewidths=0.3, annot=False,
                     annot_kws={"fontsize": 24},
                     vmin=0,
                     vmax=1,
                     # mask = mask
                     )

    plt.title(None)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()

def draw_site_map(pdb_id,output,ag_lens,ab_lens,label_dict,threshold=0.5):
    all_att = output.iil_out.att
    device =all_att.device
    # w_ab = output.local_out.w_ab.view(-1)
    # w_ag = output.local_out.w_ag.view(-1)

    # att = output.att
    score = output.score

    for i,(ag_len,ab_len) in enumerate(zip(ag_lens,ab_lens)):
        att = all_att[i]
        # w_ab = w_ab[:ab_len]
        # w_ag = w_ag[:ag_len]

        # w_ab = torch.sigmoid(norm(w_ab))
        # w_ag = torch.sigmoid(norm(w_ag))

        # n = F.softmax(score, dim=1)[:, 1]
        # pred = n.detach().cpu()

        att = att.reshape(110, 800)
        filter_att = att[:ab_len, :ag_len]

        ag_att = filter_att.mean(0)
        ab_att = filter_att.mean(1)
        # antibody
        norm_ab_att = norm(ab_att)
        norm_ag_att = norm(ag_att)

        pred_ab_label = torch.ceil(torch.sigmoid(norm_ab_att) - threshold)
        pred_ag_label = torch.ceil(torch.sigmoid(norm_ag_att) - threshold)

        recall_metric = BinaryRecall().to(device)


        label_Ag = torch.tensor(label_dict['epitope']).to(device)
        label_Ab = torch.tensor(label_dict['paratope']).to(device)
        pair_label = label_dict['paratope-epitope']

        # print(pred_ab_label)
        # print(label_Ab)

        ag_recall = recall_metric(pred_ag_label.long(), label_Ag).item()
        ab_recall = recall_metric(pred_ab_label.long(), label_Ab).item()

        pair_recall = calculate_binding_pair_recall(filter_att, pair_label, ag_len, ab_len, threshold)

        ab_padding = torch.zeros(int(ab_len / 40 + 1) * 40 - ab_len).to(att.device)
        ag_padding = torch.zeros(int(ag_len / 40 + 1) * 40 - ag_len).to(att.device)

        norm_ab_att = torch.cat([norm_ab_att, ab_padding]).to('cpu')
        norm_ag_att = torch.cat([norm_ag_att, ag_padding]).to('cpu')
        norm_ab_att = torch.sigmoid(norm_ab_att)
        norm_ag_att = torch.sigmoid(norm_ag_att)

        draw(norm_ag_att.reshape(-1, 40), f'{os.getcwd()}/figs/{pdb_id}_ag.svg')
        draw(norm_ab_att.reshape(-1, 40), f'{os.getcwd()}/figs/{pdb_id}_ab.svg')

        print(ag_recall,ab_recall,pair_recall)

        return ag_recall,ab_recall,pair_recall

        # draw(ab_label.reshape(-1, 40),path + f'{key_name}_ab_label_{seed}.svg')
        # draw(ag_label.reshape(-1, 40),path + f'{key_name}_ag_label_{seed}.svg')
        # draw2(ab_color.reshape(-1, 40), path + f'{key_name}_ab_color_{seed}.svg')
        # draw2(ag_color.reshape(-1, 40), path + f'{key_name}_ag_color_{seed}.svg')
        # draw(filter_att,path + f'{pdb_id}.svg')

full_seq,full_label,seq_dict,ab_info,label_dict = get_binding_site('3vrl','H','L','C')