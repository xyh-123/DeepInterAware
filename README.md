# DeepInterAware: deep interaction interface-aware network for improving Antigen-antibody Interaction Prediction from sequence data

![AppVeyor](https://img.shields.io/badge/pytorch-1.12.1-red)

![AppVeyor](https://img.shields.io/badge/transformers-4.24.0-brightgreen)





## DeepInterAware

The identification of interactions between candidate antibodies and target antigens is a key step in the discovery of antibody drugs. Despite the scarcity of structural data, which poses a significant challenge to the development of antigen-antibody interaction prediction methods, the abundance of available sequence data offers a rich resource for computational modeling and analysis. In this paper, we propose **DeepInterAware** (deep interaction interface-aware network), a framework dynamically incorporating interaction interface information directly learned from sequence data, along with the inherent specificity information of the sequences. Experimental results demonstrate that DeepInterAware outperforms existing methods and exhibits promising inductive capabilities for predicting interactions involving unseen antigens or antibodies, as well as transfer capabilities for similar tasks. The interaction interface information learned by DeepInterAware enables it to capture the underlying mechanisms of antigen-antibody interactions, facilitating the precise identification of potential binding sites from antigen-antibody pairs. Additionally, DeepInterAware is adept at detecting mutations in antibody sequences and providing accurate predictions for antigen-antibody pairs with subtle sequence variations. The experimental library screening for HER2 targets further underscores DeepInterAware’s exceptional capability in identifying binding antibodies for target antigens, establishing it as a robust tool for antibody drug screening.

![Our pipeline](figs/framework.png)

## Table of contents

- [Installation](##Installation)
- [Data](##data)
- [Model inference](##model-inference)
- [Data process](##Data process)

  - [Extract the CDR loops](###Extract the CDR loops)
  - [Calculate Binding sites and Binding pairs](###Calculate Binding sites and Binding pairs)
  - [Extraction of amino acid feature](###Extraction of amino acid feature)
- [Model training](#model-training)
- [Usage](##Usage)

  - [Predict antigen-antibody binding or neutralization](###Predict antigen-antibody binding or neutralization)
  -  [Indentify Binding sites](###Indentify Binding sites)
  - [Calculate the weights of the CDR regions](###Calculate the weights of the CDR regions)
  -  [Binding affinity changes](###Binding affinity changes)
- [License](##License)
- [Cite Us](##Cite Us)

## Installation

We highly recommand that you use Anaconda for Installation
```
conda create -n DeepInterAware
conda activate DeepInterAware
pip install -r requirements.txt
```

## Data

The AVIDa-hIL6 data are available at [link](https://cognanous.com/datasets/avida-hil6). The SAbDab data are available at [link](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab). The HIV data are available from CATNAP at [link](https://www.hiv.lanl.gov/components/sequence/HIV/neutralization/download_db.comp). The SARS-CoV-2 data are available from CoVAbDab at [link](http://opig.stats.ox.ac.uk/webapps/covabdab/). 

The Antigen-Antibody data is in the `data` folder, the process data can be downloaded in this [link](https://drive.google.com/file/d/12uMgZLxpqhP70tPNp-K4LFksN4E0re30/view?usp=sharing).

* `data/AVIDa_hIL6/ab_ag_pair.csv` is the paired Ab-Ag data of AVIDa_hIL6 dataset.
* `data/SAbDAb/ab_ag_pair.csv` is the paired Ab-Ag data of SAbDAb dataset.
* `data/HIV/ab_ag_pair.csv` is the all paired Ab-Ag data of HIV dataset.
* `data/CoVAbDab/ab_ag_pair.csv` is the all paired Ab-Ag data of CoVAbDab  dataset.

## Data process

### Extract the CDR loops

To extract the CDR loops ,please run,

```
python cdr_extract.py --data_path ./data/SAbDab/
```

### Calculate Binding sites and Binding pairs

To calculate the binding sites and binding pairs,please run

```python
from binding_site import get_binding_site
full_seq,full_label,seq_dict,ab_info,label_dict = get_binding_site('3vrl','H','L','C') #PDB ID,heavy,light,antigen
```

```python
>>full_dict #Full length sequences and site label
{'res_H': 'DVKLVESGGGLVKPGGSLKLSCAASGFTFSSYTMSWVRQTPEKRLEWVAIISSGGSYTYYSDSVKGRFTISRDNAKNTLYLQMSSLKSEDTAMYYCTRDEGNGNYVEAMDYWGQGTSVTVSSAKTTPPSVYPLAPGSAAQTNSMVTLGCLVKGYFPEPVTVTWNSGSLSSGVHTFPAVLQSDLYTLSSSVTVPSSTWPSETVTCNVAHPASSTKVDKKIVPR', 'res_L': 'DIQMTQSPASLSASVGETVTITCRASGNIHNYLAWYQQKQGKSPQLLVYNAKTLADGVPSRFSGSGSGTQYSLKINSLQPEDFGSYYCQHFWSTPRTFGGGTKLEIKRADAAPTVSIFPPSSEQLTSGGASVVCFLNNFYPKDINVKWKIDGSERQNGVLNSWTDQDSKDSTYSMSSTLTLTKDEYERHNSYTCEATHKTSTSPIVKSFNRNEC', 'res_AG': 'SILDIKQGPKESFRDYVDRFFKTLRAEQCTQDVKNWMTDTLLVQNANPDCKTILRALGPGATLEEMMTACQGV'}
```

```python
>>full_label
{'label_H': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'label_L': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'label_AG': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1], 'label_AGH': [[8, 10, 46, 48, 48, 48, 48, 48, 48, 49, 51, 51, 51, 52, 52, 52, 52, 53, 54, 54, 54, 54, 54, 54, 54, 55, 55, 56, 57, 58, 64, 67, 68, 69, 70, 71, 72], [56, 56, 56, 56, 51, 53, 55, 52, 54, 56, 32, 31, 50, 32, 49, 57, 58, 104, 103, 105, 102, 99, 100, 101, 106, 105, 98, 58, 104, 104, 58, 58, 58, 56, 56, 56, 55]], 'label_AGL': [[55, 56, 56, 56, 57, 57, 58, 58, 59, 60, 64], [95, 93, 91, 92, 91, 90, 90, 31, 91, 91, 93]]}
```

```python
>>ab_info #antibody information
Munch({'H_cdr1_range': [27, 37], 'H_cdr1': 'TFSSYTMSWV', 'H_cdr2_range': [44, 61], 'H_cdr2': 'LEWVAIISSGGSYTYYS', 'H_cdr3_range': [94, 112], 'H_cdr3': 'YCTRDEGNGNYVEAMDYW', 'H_cdr': 'TFSSYTMSWVLEWVAIISSGGSYTYYSYCTRDEGNGNYVEAMDYW', 'L_cdr1_range': [27, 38], 'L_cdr1': 'NIHNYLAWYQQ', 'L_cdr2_range': [43, 57], 'L_cdr2': 'PQLLVYNAKTLADG', 'L_cdr3_range': [86, 98], 'L_cdr3': 'YCQHFWSTPRTF', 'L_cdr': 'NIHNYLAWYQQPQLLVYNAKTLADGYCQHFWSTPRTF'})
```

```python
>>seq_dict
{'H_cdr': 'DPNSDHMSWVLEWIAIIYASGTTYYAFCATYPNYPTDNLW', 'ag_seq': 'DSFVCFEHKGFDISQCPKIGGHGSKKCTGDAAFCSAYECTAQYANAYCSHA', 'ab_cdr': 'DPNSDHMSWVLEWIAIIYASGTTYYAFCATYPNYPTDNLWSVYNYLLSWYQQPKRLIYSASTLASGYCLGSYDGNSADCLAF', 'L_cdr': 'SVYNYLLSWYQQPKRLIYSASTLASGYCLGSYDGNSADCLAF'}
```

```python
>>label_dict #The label after the CDR region is extracted
{'epitope': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0], 'paratope': [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0], 'paratope-epitope': [(0, 41), (2, 38), (2, 39), (2, 40), (3, 25), (3, 41), (4, 25), (5, 16), (5, 37), (17, 17), (17, 19), (17, 20), (17, 21), (18, 37), (19, 18), (21, 18), (23, 18), (30, 16), (30, 22), (32, 21), (32, 23), (32, 24), (33, 16), (33, 25), (34, 23), (34, 44), (35, 25), (35, 42), (42, 17), (42, 20), (42, 21), (43, 13), (44, 14), (45, 21), (45, 22), (45, 23), (58, 23), (70, 21), (71, 21), (72, 21), (73, 19), (78, 21), (79, 21)]}
```

### Extraction of amino acid feature

Download the ESM2 [pretrain  model](https://huggingface.co/facebook/esm2_t12_35M_UR50D) put into the /networks/pretrained-ESM2/ . To extract the amino acid feature, please run,

```python
python feature_encodr.py --data_path ./data/SAbDab
```

## Model training

To train DeepInterAware on antigen-antibody tasks, please run
```
python main.py --config=configs/SAbDAb.yml --dataset SAbDAb --kfold
python main.py --config=configs/HIV.yml --dataset HIV --unseen_task unseen
python main.py --config=configs/AVIDa_hIL6.yml --dataset AVIDa_hIL6 --unseen_task ag_unseen
python transfer.py  --config=configs/HIV.yml --unseen_task transfer
```

## Usage

Download the checkpoint of DeepInterAware and modify the paths in the code.

| Content                  | Link                                                    |
| ------------------------ | ------------------------------------------------------- |
| Checkpoint on SAbDab     | [link](https://figshare.com/ndownloader/files/44970310) |
| Checkpoint on AVIDa-hIL6 | [link](https://figshare.com/ndownloader/files/44970310) |
| Checkpoint on HIV Unseen | [link](https://figshare.com/ndownloader/files/45053224) |
| Checkpoint on CoVAbDab   | [link](https://figshare.com/ndownloader/files/45053224) |

### Predict antigen-antibody binding or neutralization

For example, to test DeepInterAware on SAbDab test data, please run

```python
from models import DeepInterAware,load_model
form feature_encoder import getAAfeature

ag_list = [
        'DSFVCFEHKGFDISQCPKIGGHGSKKCTGDAAFCSAYECTAQYANAYCSHA',
        'SILDIKQGPKESFRDYVDRFFKTLRAEQCTQDVKNWMTDTLLVQNANPDCKTILRALGPGATLEEMMTACQGV'
] #The antigen sequence length is less than 800
ab_list = [
    ('DPNSDHMSWVLEWIAIIYASGTTYYAFCATYPNYPTDNLW','SVYNYLLSWYQQPKRLIYSASTLASGYCLGSYDGNSADCLAF'),
    ('TFSSYTMSWVLEWVAIISSGGSYTYYSYCTRDEGNGNYVEAMDYW','NIHNYLAWYQQPQLLVYNAKTLADGYCQHFWSTPRTF')
]#The CDR sequence length is less than 110
ag_token_ft,ab_token_ft = getAAfeature(ag_list, ab_list, gpu=0)
ag_mask,ab_mask = get_mask(ag_list, ab_list,gpu=0)
model= load_model(f'./configs/SAbDab.yml',model_path=f'./save_models/SAbDab.pth',gpu=0)
model.eval()
with torch.no_grad():
    output = model.inference(ag_token_ft,ab_token_ft,ag_mask,ab_mask)
print(output.score)
```

The output is:

```python
tensor([0.9996, 1.0000])
```

### Indentify Binding sites

```python
from utils.binding_site import draw_site_map, get_binding_site
from utils.feature_encoder import get_mask,getAAfeature
from models import load_model
import torch

full_seq, full_label, seq_dict, ab_info, label_dict = get_binding_site('6i9i','H','L','C')
ag_list = [seq_dict['ag_seq']]
ab_list = [(seq_dict['H_cdr'],seq_dict['L_cdr'])]
ag_token_ft,ab_token_ft = getAAfeature(ag_list, ab_list, gpu=1)
ag_mask,ab_mask,ag_len,ab_len = get_mask(ag_list, ab_list,gpu=1)

model= load_model(model_name='DeepInterAware',f'./configs/SAbDab.yml',model_path=f'./save_models/SAbDab.pth',gpu=1)
model.eval()
with torch.no_grad():
    output = model.inference(ag_token_ft,ab_token_ft,ag_mask,ab_mask)
ag_recall,ab_recall,pair_recall = draw_site_map('6i9i',output,ag_len,ab_len,label_dict,threshold=0.5)
```

Antigen attention map

![Our pipeline](figs/6i9i_ag.png)

Antibody attention map

![Our pipeline](figs/6i9i_ab.png)

### Calculate the weights of the CDR regions

The following code is executed to calculate the weights of the CDR regions.

```python
from utils.binding_site import attribution_cdr
weight_list = attribution_cdr(output, ab_info, ab_len)
>>weight_list #[HCDR1,HCDR2,HCDR3,LCDR1,LCDR2,LCDR3]
[[0.12554966, 0.21922821, 0.31082207, 0.11571003, 0.11434505, 0.11434505] ]
```

### Binding affinity changes

![affinity_changes](figs/affinity_changes.png)

For batch prediction of binding affinity changes, execute the following command

```python
python affinity_maturation.py --wt ./data/example/wt.csv --mu ./data/example/mu.csv --gpu 0
```

## License

DeepInterAware content and derivates are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). 

## Cite Us

Feel free to cite this work if you find it useful to you!

```
@article{DeepInterAware,
    title={DeepInterAware: deep interaction interface-aware network for improving Antigen-antibody Interaction Prediction from sequence data},
    author={Yuhang Xia, Zhiwei Wang, Yongkang Wang, Minyao Qiu, Wen Zhang},
    year={2024},
}
```

