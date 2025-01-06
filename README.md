# DeepInterAware: Deep interaction interface-aware network for improving antigen-antibody Interaction Prediction from sequence data

## DeepInterAware

 In this paper, we propose **DeepInterAware** (deep interaction interface-aware network), a framework dynamically incorporating interaction interface information directly learned from sequence data, along with the inherent specificity information of the sequences. Relying solely on sequence data, it allows for a more profound insight into the underlying mechanisms of antigen-antibody interactions, offering the capability to predict binding or neutralization,identify potential binding sites and predict binding free energy changes upon mutations. 

![Our pipeline](figs/framework.png)

## Table of contents

- [Dependencies](#Dependencies)
  - [From Conda](#From-Conda)
  - [From Docker](#From-Docker)

- [Data and Data process](#Data-and-Data-process)
  - [Ag-Ab binding datasets](#Ag-Ab-binding-datasets)
  - [Ag-Ab neutralization datasets](#Ag-Ab-neutralization-datasets)
  - [Binding free energy change dataset](#Binding-free-energy-change-dataset)
  - [Data process](#Data-process)
    - [Extraction of amino acid features](#extraction-of-amino-acid-features)

- [Model Train](#Model-Train)
  - [Ag-Ag Binding Prediction](#Ag-Ag-Binding-Prediction)
  - [Ag-Ab Neutralization Prediction](#Ag-Ab-Neutralization-Prediction)
  - [Binding Site Identifcation](#Binding-Site-Identifcation)
  - [Binding Free Energy Change Prediction](#Binding-Free-Energy-Change-Prediction)
  
- [Model Test](#Moldel-Test)
  - [Predict Binding or Neutralization](#Predict-Binding-or-Neutralization)
  - [Identify Potential Binding Sites](#Identify-Potential-Binding-Sites)
  - [ Predict Binding Free Energy Changes](#Predict-Binding-Free-Energy-Changes)
- [License](#License)
- [Conflict of Interest](#Conflict-of-Interest)
- [Cite Us](#cite-us)

## Dependencies

<font style="color:rgb(31, 35, 40);">Our model is tested in Linux with the following packages:</font>

+ <font style="color:rgb(31, 35, 40);">CUDA >= 11.3</font>
+ <font style="color:rgb(31, 35, 40);">PyTorch == 1.12.1 </font>
+ <font style="color:rgb(31, 35, 40);">anarchi == 1.3</font>
+ <font style="color:rgb(31, 35, 40);">ablang</font>
+ <font style="color:rgb(31, 35, 40);">antiberty</font>
+ transformers==4.24.0

### From Conda

We highly <font style="color:black;">recommend </font> that you use Anaconda for Installation:

```shell
conda create -n DeepInterAware
conda activate DeepInterAware
pip install -r requirements.txt
```

### From Docker

Our Docker can be downloaded from [Link](https://drive.google.com/file/d/12uMgZLxpqhP70tPNp-K4LFksN4E0re30/view?usp=drive_link)  and stored in the data directory.

```shell
#import the docker
docker import aai.tar deepinteraware
#run the docker
docker run  --name run_images --gpus all -idt deepinteraware
#activate the experimental environment
conda activate aai
```

## Data and Data process

### Ag-Ab binding datasets

[AVIDa-hIL6](https://cognanous.com/datasets/avida-hil6) is a comprehensive Ag-Ab binding sequence dataset for predicting AAIs in the variable domain of heavy chain antibodies (VHHs), featuring the wild-type IL-6 protein and its 30 mutants as antigens. To refine the dataset for our study, we employed ANARCI to extract the CDR loops from the antibody sequences and finally obtained 10,178 binding pairs and 315,708 non-binding pairs.

[SAbDab](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/) database  is a comprehensive compilation of all accessible Ag-Ab complexes, meticulously curated from the Protein Data Bank (PDB).  To adapt to the binding prediction task in our paper, we collected the complexes with antigen sequences containing more than 50 residues, and filtered out duplicates with the antibody CDR loops, arriving at a refined set of 1,513 binding pairs as the SAbDab dataset in our paper.

### Ag-Ab neutralization datasets

[HIV](https://www.hiv.lanl.gov/components/sequence/HIV/neutralization/) sequence database comprises neutralization antibodies related to the Human Immunodeficiency Virus (HIV). We meticulously filtered out Ag-Ab pairs that exhibited sequence homology levels exceeding 90% for both the antigens and the antibodies, and curated the HIV sequence dataset encompasses 24,907 neutralization pairs with sequences and 26,480 non-neutralization pairs.

[CoV-AbDab](https://opig.stats.ox.ac.uk/webapps/covabdab/) database offers detailed information regarding conventional antibodies and nanobodies capable of binding to various coronaviruses. We collected the Ag–Ab neutralization and non-neutralization pairs and antibody sequences from the CoV-AbDab database, and finally obtained the CoV-AbDab dataset consisting of 5,486 neutralization pairs and 9,110 non-neutralization pairs with sequences. 

### Binding free energy change dataset

[AB-Bind](https://github.com/sarahsirin/AB-Bind-Database) database  includes 1,101 mutants with experimentally determined changes in binding free energies (<font style="color:black;">△△</font>G) across 32 Ag-Ab complexes. We screened the Ag-Ab mutants annotated with light and heavy chains, consisting of 654 unique mutants. 

[SKEMPI2](https://life.bsc.es/pid/skempi2/) database  provides data on changes in protein-protein binding energy, kinetics, and thermodynamics upon mutations. As we did on the AB-Bind, we screened Ag-Ab mutants annotated with both light and heavy chains, resulting in 1,021 mutants.

### Data process

All the processed dataset can be downloaded from [Link](https://drive.google.com/file/d/12uMgZLxpqhP70tPNp-K4LFksN4E0re30/view?usp=drive_link)  and stored in the data directory.

To preprocess all datasets, please run,

```sh
bash scripts/data_process.sh
```

#### Extraction of amino acid features

Download the ESM2 [pretrain  model](https://huggingface.co/facebook/esm2_t12_35M_UR50D) put into the /networks/pretrained-ESM2/ . For encoded antibody features using AbLang, first, run pip install ablang to install the corresponding library. To extract the amino acid features of antigens and antibodies, please run,

```sh
python feature_encodr.py --data_path ./data/HIV --gpu 0
```

## Model Train

### Ag-Ag Binding Prediction

+ On the AVIDa-hIL6 dataset, we conducted  five independent experiments to evaluate the DeepInterAware , please run:

```sh
python main.py --cfg ./configs/AVIDa_hIL6.yml --dataset AVIDa_hIL6 --gpu 0 --batch_size 512
```

+ On the SAbDab dataset, we conducted  five independent experiments to evaluate the DeepInterAware , please run:

```sh
python main.py --cfg ./configs/SAbDab.yml --dataset SAbDab --gpu 0 --batch_size 128
```

+ For baselines in our paper, we also evaluated their performance on these datasets, please run:

```sh
bash scripts/train_baseline.sh
```

+ The performances of our method and these baselines on the  AVIDa-hIL6 and SAbDab datasets are demonstrated in Table 1 in our paper.

### Ag-Ab Neutralization Prediction

- On the HIV dataset, we conducted  the five independent experiments to evaluate the DeepInterAware under the Ab Unseen, Ag Unseen, and Ag&Ab Unseen scenarios , please run:

```sh
python main.py --cfg ./configs/HIV.yml --dataset HIV --unseen_task unseen --batch_size 256
python main.py --cfg ./configs/HIV.yml --dataset HIV --unseen_task ab_unseen --batch_size 256
python main.py --cfg ./configs/HIV.yml --dataset HIV --unseen_task ag_unseen --batch_size 256
```

- On the CoV-AbDab dataset, we conducted  five independent experiments to evaluate the transferability of DeepInterAware, please run:

```sh
python transfer.py  --config=configs/HIV.yml  --batch_size 32
```

- The performances of our method and these baselines on the HIV dataset and  CoV-AbDab dataset are demonstrated in Table 2 in our paper and Table 1 in Supplementary , respectively.

### Binding Site Identifcation

- On the SAbDab dataset, we conducted five-fold cross-validation to evaluate the performance of DeepInterAware in binding site identification , please run:

```sh
python site_train.py --lr 1e-3 --batch_size 32 --end_epoch 150 --epoch 300
```

### Binding Free Energy Change Prediction

On the AB-Bind and SKEMPI2 datasets, we conducted  ten-fold cross-validation to evaluate the performance of DeepInterAware in binding free energy change prediction, please run:

```sh
python ddG_train.py --dataset AB-Bind --batch_size 32 --lr 5e-4
```

## Model Test

### Predict Binding or Neutralization

```shell
python usage.py --task binding --pair_file ./data/example/binding_pair.csv --gpu 0 --model_path ./save_models/

python usage.py --task neutralization --pair_file ./data/example/hiv_neutralization_pair.csv --gpu 0 --model_path ./save_models/
```

### Identify Potential Binding Sites

+ In addition to  its primary benefits in AAI prediction, our model can learn some structural information for the sequences, and we try to validate this point by testing the capability of our method in identifying potential binding sites:

```shell
python usage.py --task binding_site --pair_file ./data/example/binding_site_pair.csv --gpu 0 --model_path ./save_models/
```

### Predict Binding Free Energy Changes

![affinity_changes](figs/affinity_changes.png)

+ The residue mutations of antigen and antibody sequences can significantly alter the AAI. To thoroughly evaluate the performance of DeepInterAware in identifying the efficacy of Ag-Ab mutants, we adopt Ag-Ab complexes and mutants as well as experimentally determined changes in binding free energies:

```sh
python usage.py --task ddG --wt ./data/example/wt.csv --mu ./data/example/mu.csv --gpu 0 --model_path ./save_models/
```

## License

<font style="color:rgb(51, 51, 51);">DeepInterAware content and derivates are licensed under </font>[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)<font style="color:rgb(51, 51, 51);">. If you have any requirements beyond the agreement, you can contact us.</font>

## Conflict of Interest

W.Z., and Y.X. are inventors on patent applications related to this work filed by Wuhan Huamei Biotech Co.,Ltd. (Chinese  patent application nos. 2023.12.21 202311783760.X). The authors declare no other competing interests.

## Cite Us

Feel free to cite this work if you find it useful to you!

```sh
@article{DeepInterAware,
    title={DeepInterAware: deep interaction interface-aware network for improving antigen-antibody interaction prediction from sequence data},
    author={Yuhang Xia, Zhiwei Wang,Feng Huang,Zhankun Xiong,Yongkang Wang, Minyao Qiu, Wen Zhang},
    year={2025},
}
```





