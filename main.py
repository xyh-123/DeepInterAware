import argparse
from torch import randperm
from torch.utils.data import DataLoader,Subset
from yacs.config import load_cfg
from networks.models import DeepInterAware
from configs import get_cfg_defaults
from dataloader import return_dataset, _make_balanced_sampler, k_fold_dataset
import torch
import warnings, os

from utools.comm_utils import set_seed,mkdir
from munch import Munch
from train import Trainer
project_dir = os.path.dirname(os.path.abspath(__file__))

def return_sampler(dataset):
    if dataset != None:
        sampler = _make_balanced_sampler(dataset.label)
        return sampler
    else:
        return None

parser = argparse.ArgumentParser(description="antibody-antigen binding affinity prediction")
# setting
parser.add_argument('--data', default=f'{project_dir}/data', type=str, metavar='TASK',
                    help='data path')
parser.add_argument('--model_path', default=None, type=str, metavar='TASK',help='model train path')
parser.add_argument('--cfg', default=f'{project_dir}/configs/SAbDab.yml',required=True, help="path to config file", type=str)
# parser.add_argument('--seed', default=0, type=int, metavar='S',
#                     help='seed for HIV dataset or others')
parser.add_argument('--dataset', default='HIV', type=str, metavar='S',
                    help='training dataset')
parser.add_argument('--site_test', action='store_true',help='test the binding site')
parser.add_argument('--mode', default='debug', type=str, metavar='S',
                    help='Current mode traing,test,debug')
parser.add_argument('--gpu', default=0, type=int, metavar='S', help='run GPU number')
parser.add_argument('--weight_decay', default=0, type=float, metavar='S', help='dataset')
parser.add_argument('--train_epoch', default=100, type=int, metavar='S', help='max train epoch')
parser.add_argument('--simple', action='store_true', help='For MLP Decoder type')
parser.add_argument('--batch_size', default=32, type=int, metavar='S',help='dataset')
parser.add_argument('--lr', default=1e-3, type=float, metavar='S',help='learning rate')
parser.add_argument('--model_name', default='DeepInterAware', type=str, metavar='S'
                    # ,choices=['BindingEncoderv1','PCLEncoder','DrugBANv2','InteractionBAN']
                    ,help='pretrain model')
parser.add_argument('--ab_model', default='ablang', type=str, metavar='S'
                    # ,choices=['BindingEncoderv1','PCLEncoder','DrugBANv2','InteractionBAN']
                    , help='pretrain model')
parser.add_argument('--sampler', action='store_true',help='use sampler')  # default false
parser.add_argument('--alpha', default=0.4, type=float, metavar='S',help='loss hyperparameter')
parser.add_argument('--unseen_task', default='ab_unseen' , type=str, metavar='S',choices=['ab_unseen','ag_unseen','unseen','transfer'],
                    help='Only for HIV dataset cls or reg')
parser.add_argument('--phy', action='store_true')
parser.add_argument('--save_best', action='store_true')
# parser.add_argument('--kfold', action='store_true')
parser.add_argument('--use_bn', action='store_false')
parser.add_argument('--h_dim', default=512, type=int, metavar='S',help='dataset')
parser.add_argument('--metric_type', default='roc_auc', type=str, metavar='S',help='dataset')
parser.add_argument('--dropout', default=0.5, type=float, metavar='S',help='dataset')
parser.add_argument('--channel', default=128, type=int, metavar='S',help='dataset')
# parser.add_argument('--start_epoch', default=0, type=int, metavar='S',help='dataset')
parser.add_argument('--end_epoch', default=30, type=int, metavar='S',help='dataset')
args=parser.parse_args()
# cfg = get_cfg_defaults()
with open(args.cfg, "r") as f:
    cfg = load_cfg(f)
cfg.set.ab_model = args.ab_model
cfg.set.unseen_task = args.unseen_task
# cfg.solver.train_epoch = args.train_epoch
# cfg.solver.end_epoch = args.end_epoch
cfg.result.output_dir = f'{os.path.dirname(args.data)}/result'
cfg.set.device = args.gpu

print(cfg)
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
current_work_dir = os.path.dirname(os.path.abspath(__file__))  #

def count_model_params(model):
    print(model)
    param_size = {}
    cnt = 0
    for name, p in model.named_parameters():
        k = name.split('.')[0]
        if k not in param_size:
            param_size[k] = 0
        p_cnt = 1
        for j in p.size():
            p_cnt *= j
        param_size[k] += p_cnt
        cnt += p_cnt
    for k, v in param_size.items():
        print(f"Number of parameters for {k} = {round(v / 1024, 2)} k")
    print(f"Total parameters of model = {round(cnt / 1024, 2)} k")

if __name__ == '__main__':
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    # suffix = str(int(time() * 1000))[6:]
    mkdir(cfg.result.output_dir)
    for seed in range(0,5):
        set_seed(seed)
        cfg.solver.seed = seed
        params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': cfg.solver.num_workers,
                  'drop_last': True}

        dataFolder = f'{args.data}/{cfg.set.dataset}/'
        print(dataFolder)
        model = DeepInterAware(cfg)

        weight_p, bias_p = [], []

        count_model_params(model)
        model = model.to(device)
        for name, p in model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        # optimizer_inner = torch.optim.Adam(
        #     [{'params': weight_p, 'weight_decay': 1e-4}, {'params': bias_p, 'weight_decay': 0}], lr=args.lr)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=cfg.solver.weight_decay)
        # opt = Lookahead(optimizer_inner, k=5, alpha=0.5)


        if cfg.set.dataset != 'SAbDab':
        # if not args.kfold:
            train_dataset, val_dataset, unseen_dataset= return_dataset(cfg,dataFolder)
            # scheduler = CosineAnnealingLR(opt, T_max=50, eta_min=1e-5)
        else:
            train_dataset, val_dataset, train_pair_id, test_pair_id = k_fold_dataset(cfg, dataFolder, seed)
            unseen_dataset = None
            # scheduler = CosineAnnealingLR(opt, T_max=75, eta_min=1e-3)

        if args.sampler:
            train_sampler = return_sampler(train_dataset)
            params['shuffle'] = False
        else:
            train_sampler = None
            params['shuffle'] = True

        if args.mode=='train_test':
            train_indices = randperm(len(train_dataset)).tolist()[:1000]
            val_indices = randperm(len(val_dataset)).tolist()[:1000]
            # test_indices = randperm(len(test_dataset)).tolist()[:1000]
            train_dataset,val_dataset=Subset(train_dataset,train_indices),Subset(val_dataset,val_indices)
            if unseen_dataset != None:
                unseen_indices = randperm(len(unseen_dataset)).tolist()[:1000]
                unseen_dataset = Subset(unseen_dataset,unseen_indices)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, **params)
        params['shuffle'] = True
        params['drop_last'] = False
        val_dataloader = DataLoader(val_dataset, **params)
        if unseen_dataset != None:
            unseen_dataloader = DataLoader(unseen_dataset, sampler=None, **params)
        else:
            unseen_dataloader = None

        trainer_parameter = Munch(
            device=device,
            current_epoch=0,
            save_best=args.save_best,
            metric_type=args.metric_type,
            sampler=args.sampler,
            batch_size=args.batch_size,
            # start_epoch=args.start_epoch,
            end_epoch=cfg.solver.end_epoch,
            model = model,
            opt=opt,
            # scheduler=scheduler,
            # optimizer_inner = optimizer_inner,
            cfg=cfg,
            site_test=args.site_test
            # cfg=cfg
        )
        trainer_parameter.train_dataloader = train_dataloader
        trainer_parameter.val_dataloader = val_dataloader
        trainer_parameter.unseen_dataloader = unseen_dataloader
        # trainer = SMAICFTrainer(trainer_parameter)
        trainer = Trainer(trainer_parameter)
        print()
        print(f"Directory for saving result: {trainer.save_file_path}")
        result = trainer.train()

        with open(os.path.join(trainer.save_file_path, "model_architecture.txt"), "w") as wf:
            wf.write(str(model))
