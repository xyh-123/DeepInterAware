# Author: Pietro Sormanni
## FUNCTIONS FOR SEQUENCE PROCESSING OF INPUTS ####
import json
import sys
import re
import pandas as pd
from munch import Munch
import argparse
import os
sys.path.append(os.getcwd())
# from bindingsite import id2fasta2dict, SAbDab_site2
from tqdm import tqdm
NUM_EXTRA_RESIDUES=2

def get_CDR_simple(sequence ,allow=set(["H", "K", "L"]),scheme='chothia',seqname='' \
                   ,cdr1_scheme={'H':range(30-NUM_EXTRA_RESIDUES,36+NUM_EXTRA_RESIDUES),'L':range(30-NUM_EXTRA_RESIDUES,37+NUM_EXTRA_RESIDUES)} \
                   ,cdr2_scheme={'H':range(47-NUM_EXTRA_RESIDUES,59+NUM_EXTRA_RESIDUES),'L':range(46-NUM_EXTRA_RESIDUES,56+NUM_EXTRA_RESIDUES)} \
                   ,cdr3_scheme={'H':range(93-NUM_EXTRA_RESIDUES,102+NUM_EXTRA_RESIDUES),'L':range(89-NUM_EXTRA_RESIDUES,97+NUM_EXTRA_RESIDUES)}) :
# def get_CDR_simple(sequence, allow=set(["H", "K", "L"]), scheme='chothia', seqname='' \
#                    , cdr1_scheme={'H': range(26 - NUM_EXTRA_RESIDUES, 33 + NUM_EXTRA_RESIDUES),
#                                   'L': range(24 - NUM_EXTRA_RESIDUES, 35 + NUM_EXTRA_RESIDUES)} \
#                    , cdr2_scheme={'H': range(52 - NUM_EXTRA_RESIDUES, 57 + NUM_EXTRA_RESIDUES),
#                                   'L': range(50 - NUM_EXTRA_RESIDUES, 57 + NUM_EXTRA_RESIDUES)} \
#                    , cdr3_scheme={'H': range(95 - NUM_EXTRA_RESIDUES, 103 + NUM_EXTRA_RESIDUES),
#                                   'L': range(89 - NUM_EXTRA_RESIDUES, 98 + NUM_EXTRA_RESIDUES)}):
    '''
    From a VH or VL amino acid sequences returns the three CDR sequences as determined from the input numbering (scheme) and the given ranges.
    default ranges are Chothia CDRs +/- NUM_EXTRA_RESIDUES residues per side.
      requires the python module anarci - Available from http://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/ANARCI.php

    For other numbering schemes see also http://www.bioinf.org.uk/abs/#cdrdef
    Loop    Kabat          AbM    Chothia1    Contact2
    L1    L24--L34    L24--L34    L24--L34    L30--L36
    L2    L50--L56    L50--L56    L50--L56    L46--L55
    L3    L89--L97    L89--L97    L89--L97    L89--L96
    H1    H31--H35B   H26--H35B   H26--H32..34  H30--H35B
    H1    H31--H35    H26--H35    H26--H32    H30--H35
    H2    H50--H65    H50--H58    H52--H56    H47--H58
    H3    H95--H102   H95--H102   H95--H102   H93--H101

    For generic Chothia identification can set auto_detect_chain_type=True and use:
    cdr1_scheme={'H':range(26,34),'L':range(24,34)}
    cdr2_scheme={'H':range(52,56),'L':range(50,56)}
    cdr3_scheme={'H':range(95,102),'L':range(89,97)}
    '''
    try :
        import anarci
    except ImportError :
        raise Exception("\n**ImportError** function get_CDR_simple() requires the python module anarci\n Available from http://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/ANARCI.php\n\n")

    # res_num_all=anarci.number(sequence, scheme=scheme, allow=allow)
    res_num_all=anarci.number(sequence, scheme=scheme,allowed_species=['human','mouse','rat','rabbit','rhesus','pig','alpaca'])

    # print("===========res_num_all===========")
    # print(res_num_all)

    if not hasattr(res_num_all[0], '__len__') :
        sys.stderr.write( "*ERROR* in get_CDR_simple() anarci failed on %s -returned %s chaintype=%s\n" % (seqname,str(res_num_all[0]),str(res_num_all[1])))
        return None
    cdr1,cdr2,cdr3='','',''
    chain_type=res_num_all[1]
    # sys.stdout.write( '%s chain_type= %s\n'%(seqname,chain_type))
    if hasattr(cdr1_scheme, 'keys') : # supports dictionary or OrderedDict as input type - assume all cdr ranges are like this
        if chain_type=='K' and chain_type not in cdr1_scheme : chain_type='L' # Kappa light chain to Lambda light chain for this purpose
        if chain_type not in cdr1_scheme :
            raise Exception("\n chain_type %s not in input cdr1_scheme\n" % (chain_type))
        cdr1_scheme=cdr1_scheme[chain_type]
        cdr2_scheme=cdr2_scheme[chain_type]
        cdr3_scheme=cdr3_scheme[chain_type]
    # extract CDR sequences
    for num_tuple,res in res_num_all[0] :
        if num_tuple[0] in cdr1_scheme: cdr1+=res # num_tuple[1] may be an insertion code, (e.g. 111B)
        elif num_tuple[0] in cdr2_scheme: cdr2+=res
        elif num_tuple[0] in cdr3_scheme: cdr3+=res

    # put in parapred formta
    cdrs={'cdr1':cdr1,'cdr2':cdr2,'cdr3':cdr3}
    cdr=Munch(cdrs)
    return cdr

def extract_CDR(res_H,res_L):
    ab_info = Munch()
    ab_cdr = ""
    if res_H != None:
        # ab_info['res_H'] = res_H
        height_chain = get_CDR_simple(res_H, " heavy")
        h_cdr1, h_cdr2, h_cdr3 = height_chain.cdr1, height_chain.cdr2, height_chain.cdr3
        h_cdr1, h_cdr2, h_cdr3 = h_cdr1.replace('-', ''), h_cdr2.replace('-', ''), h_cdr3.replace('-', '')
        height_cdr = h_cdr1 + h_cdr2 + h_cdr3
        # print(h_cdr1,h_cdr2,h_cdr3)
        ab_cdr +=height_cdr

    if res_L != None:
        light_chain = get_CDR_simple(res_L, seqname="light")
        l_cdr1, l_cdr2, l_cdr3 = light_chain.cdr1, light_chain.cdr2, light_chain.cdr3
        l_cdr1, l_cdr2, l_cdr3 = l_cdr1.replace('-', ''), l_cdr2.replace('-', ''), l_cdr3.replace('-', '')
        light_cdr = l_cdr1 + l_cdr2 + l_cdr3
        ab_cdr += light_cdr

    if res_H != None:
        for i, cdr in enumerate([h_cdr1, h_cdr2, h_cdr3]):
            cdr_index = res_H.find(cdr)
            cdr_range = cdr_index + len(cdr)

            ab_info[f'H_cdr{i + 1}_range'] = [cdr_index, cdr_range]
            ab_info[f'H_cdr{i + 1}'] = cdr

        ab_info['H_cdr'] = height_cdr

    if res_L != None:
        # ab_info['res_L'] = res_L
        for i, cdr in enumerate([l_cdr1, l_cdr2, l_cdr3]):
            cdr_index = res_L.find(cdr)
            cdr_range = cdr_index + len(cdr)
            ab_info[f'L_cdr{i + 1}_range'] = [cdr_index, cdr_range]
            ab_info[f'L_cdr{i + 1}'] = cdr
        # print(f"cdr{i+1} start {cdr_index} end {cdr_range}")
        ab_info['L_cdr'] = light_cdr

    # ab_info['ab_cdr'] = ab_cdr

    return ab_info,ab_cdr

parser = argparse.ArgumentParser(description="antibody-antigen binding affinity prediction")
# setting
parser.add_argument('--data_path', default='./data', type=str, metavar='TASK',
                    help='data path')
args=parser.parse_args()
if __name__ == '__main__':
    antibody = pd.read_csv(args.data_path+'/antibody.csv')
    ab_label, ag_label = {}, {}
    ab_info_dict = {}
    if 'ab_seq' not in antibody.columns:
        antibody.insert(loc=0, column='H_cdr', value='')
        antibody.insert(loc=1, column='L_seq', value=None)
        antibody.insert(loc=2, column='ab_seq', value='')
        for index, values in antibody.iterrows():
            light, heavy= values['H'], values['L']
            ab_info,ab_cdr = extract_CDR(light, heavy)
            ab_cdr = ''
            h_cdr = ab_info['H_cdr']
            h_cdr = re.sub(r"[XUZOBJ?*-]", "", str(h_cdr))
            antibody['H_cdr'].iloc[index] = h_cdr
            ab_cdr+=h_cdr
            if 'L_cdr' in ab_info:
                l_cdr = ab_info['L_cdr']
                l_cdr = re.sub(r"[XUZOBJ?*-]", "", str(l_cdr))
                ab_cdr+=l_cdr
                antibody['L_cdr'].iloc[index] = l_cdr
            antibody['ab_seq'].iloc[index] = ab_cdr
        antibody.to_csv(args.data_path+'/antibody.csv',index=False)
