import  torch
import  numpy as np
import  argparse

import os
import shutil
import yaml
import argparse
from otrans.model import Transformer
from otrans.optim import *
from otrans.train import Trainer
from otrans.data import AudioDataset

from meta import Meta
from    learner import LearnerTransformer

def main(args):

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda')

    net = LearnerTransformer(args.config, args)

    # device = torch.device('cuda')
    # maml = Meta(args, config).to(device)
    #
    # tmp = filter(lambda x: x.requires_grad, maml.parameters())
    # num = sum(map(lambda x: np.prod(x.shape), tmp))
    # print(maml)
    # print('Total trainable tensors:', num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None)
    parser.add_argument('-n', '--ngpu', type=int, default=1)
    parser.add_argument('-s', '--seed', type=int, default=1222)
    parser.add_argument('-p', '--parallel_mode', type=str, default='dp')
    parser.add_argument('-r', '--local_rank', type=int, default=0)

    parser.add_argument('--epoch', type=int, help='epoch number', default=40000)
    parser.add_argument('--n_way', type=int, help='n way', default=5)
    parser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    parser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    parser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    parser.add_argument('--imgc', type=int, help='imgc', default=1)
    parser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    parser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    parser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    parser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    parser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)


    cmd_args = parser.parse_args()

    if cmd_args.parallel_mode == 'ddp':
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ["OMP_NUM_THREADS"] = '1'
        torch.cuda.set_device(cmd_args.local_rank)

    main(cmd_args)