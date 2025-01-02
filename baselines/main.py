from trainer import Trainer
from dataloader import return_dataset, k_fold_dataset
from munch import Munch
from utools.comm_utils import set_seed, mkdir
from torch.utils.data import DataLoader
import torch
import argparse
import warnings, os
from models import return_model
from yacs.config import load_cfg
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parser = argparse.ArgumentParser(description="antibody-antigen binding affinity prediction")
# setting
parser.add_argument('--data', default='./data', type=str, metavar='TASK',
                    help='data path')
parser.add_argument('--model_path', default=None, type=str, metavar='TASK',help='model train path')
parser.add_argument('--cfg', default='./baselines/baseline.yml', help="path to config file", type=str)
parser.add_argument('--dataset', default='HIV', type=str, metavar='S',
                    help='training dataset')
parser.add_argument('--CKSAAP', action='store_true')
parser.add_argument('--gpu', default=0, type=int, metavar='S', help='run GPU number')
parser.add_argument('--train_epoch', default=100, type=int, metavar='S', help='max train epoch')
parser.add_argument('--batch_size', default=32, type=int, metavar='S',help='dataset')
parser.add_argument('--lr', default=1e-3, type=float, metavar='S',help='learning rate')
parser.add_argument('--model_name', default='DeeppAAI', type=str, metavar='S',help='baseline model')
parser.add_argument('--unseen_task', default='ab_unseen' , type=str, metavar='S',choices=['ab_unseen','ag_unseen','unseen','transfer'],
                    help='Only for HIV dataset cls or reg')
parser.add_argument('--h_dim', default=512, type=int, metavar='S',help='dataset')
parser.add_argument('--kfold', action='store_true')
parser.add_argument('--metric_type', default='roc_auc', type=str, metavar='S',help='dataset')
parser.add_argument('--save_best', action='store_false')
args=parser.parse_args()
# cfg = get_cfg_defaults()
with open(args.cfg, "r") as f:
    cfg = load_cfg(f)

cfg.set.dataset = args.dataset
cfg.set.unseen_task = args.unseen_task
cfg.set.model_name = args.model_name
cfg.set.CKSAAP = args.CKSAAP
output_dir = f'{project_dir}/result'
print(cfg)
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')


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
    mkdir(output_dir)
    for seed in range(0,5):
        set_seed(seed)
        params = {'batch_size': args.batch_size, 'shuffle': True,
                  'drop_last': True}
        cfg.set.seed = seed
        dataFolder = f'{args.data}/{cfg.set.dataset}/'

        if not args.kfold:
            train_dataset, val_dataset, unseen_dataset= return_dataset(cfg,dataFolder)
            # scheduler = CosineAnnealingLR(opt, T_max=50, eta_min=1e-5)
        else:
            train_dataset, val_dataset, train_pair_id, test_pair_id = k_fold_dataset(cfg, dataFolder, seed)
            unseen_dataset = None
            # scheduler = CosineAnnealingLR(opt, T_max=75, eta_min=1e-3)
        model = return_model(cfg)
        count_model_params(model)
        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        # opt = Lookahead(optimizer_inner, k=5, alpha=0.5)
        train_dataloader = DataLoader(train_dataset, sampler=None, **params)
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
            batch_size=args.batch_size,
            metric_type=args.metric_type,
            model = model,
            seed = seed,
            output_dir = output_dir,
            train_epoch =args.train_epoch,
            opt=opt,
            cfg=cfg,
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
