import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict

from models.classic_network import SDicLMIL,AttnMIL,init_dct

from utils import *
from wsi_dataloader import C16DatasetV3
from scipy import linalg
from timm.optim.adamp import AdamP
from radam import RAdam
from lookhead import Lookahead


def main():
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    parser.add_argument('--dataroot', default="datasets/tcga-lung/feats/ImageNet", type=str, help='dataroot for the CAMELYON16 dataset')
    parser.add_argument('--backgrd_thres', default=30, type=int, help='background threshold')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers used in dataloader [4]')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=1e-4, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--dropout_patch', default=0.0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--seed', default='0', type=int, help='random seed')
    parser.add_argument('--model', default='SDicL', type=str, help='MIL model [dsmil]')
    parser.add_argument('--save_dir', default='./trained_models/CAMELYON16', type=str, help='the directory used to save all the output')
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 1e-5)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100],gamma=0.2)
    trainset = C16DatasetV3(args, 'train')
    testset = C16DatasetV3(args, 'test')
    val = C16DatasetV3(args, 'val')

    trainloader = DataLoader(trainset, 1, shuffle=True, num_workers=args.num_workers, drop_last=False, pin_memory=True)

    testloader = DataLoader(testset, 1, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)

    valloader = DataLoader(val, 1, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)

    totall = 0
    avage = 0
    count = 0
    train_totall = 0
    test_totall = 0
    val_totall = 0
    for batch_id, (feats, label) in enumerate(trainloader):
        #print(feats.shape)
        count += 1
        train_totall += 1
        _,n,_ = feats.shape
        totall +=n
    print('totall training set:',train_totall)

    for batch_id, (feats, label) in enumerate(testloader):
        #print(feats.shape)
        count += 1
        test_totall += 1
        _,n,_ = feats.shape
        totall +=n
    print('totall testing set:', test_totall)

    for batch_id, (feats, label) in enumerate(valloader):
        #print(feats.shape)
        count += 1
        val_totall += 1
        _,n,_ = feats.shape
        totall +=n
    print('totall validation set:', val_totall)

    print(totall)
    print(totall/count)

if __name__ == '__main__':
    main()