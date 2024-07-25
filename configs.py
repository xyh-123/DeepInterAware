from yacs.config import CfgNode as CN
import torch
import yaml

Config = CN()
Config.dump()
Config.protein = CN()

Config.protein.padding = True
Config.protein.max_antibody_len = 110
Config.protein.max_antigen_len = 800
Config.protein.use_bn = False
Config.protein.dropout = 0.1
Config.protein.channel = 192

# bcn setting
Config.bcn = CN()
Config.bcn.heads = 2

# MLP predict
Config.predict = CN()
Config.predict.name = "MLP"
# Config.predict.IN_DIM = 256 过拟合
Config.predict.in_dim = 256
Config.predict.hidden_dim = 256

Config.predict.out_dim = 256
Config.predict.binary = 2
Config.predict.simple=False

# solver,超参数设置
Config.solver = CN()
Config.solver.train_epoch = 100
Config.solver.batch_size = 128
Config.solver.num_workers = 0
Config.solver.lr = 1e-3
Config.solver.seed = 0
Config.solver.weight_decay=0

# result
Config.result = CN()
Config.result.output_dir = "./result"
Config.result.save_model = True

# # Comet configs, ignore it If not installed.
# Config.comet = CN()
# # Please change to your own workspace name on comet.
# Config.comet.workspace = "xyh-123"
# Config.comet.project_name = "ag-ab-prediction"
# Config.comet.use = False
# Config.comet.tag = None

#train setting
Config.set = CN()
Config.set.dataset='HIV'
Config.set.model_name='DeepInterAware'
Config.set.device=0
Config.set.resume_path=None
Config.set.pretrain=False
Config.set.ab_model = 'ablang' #antiberty or ablang

Config.set.project=True
# Config.set.task='cls'
Config.set.unseen_task = 'unseen'
Config.set.phy=False
Config.set.alpha = 0.4

def get_cfg_defaults():
    return Config.clone()
Config.dump(stream=open('./configs/SAbDab.yml', 'w', encoding='utf-8'))
# print(Config)
# # with open('./configs/SAbDab.yml', 'w', encoding='utf-8') as f:
# #    yaml.dump(data=Config, stream=f, allow_unicode=True)