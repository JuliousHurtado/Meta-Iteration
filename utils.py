import torch
import os
import argparse
from collections import defaultdict

import learn2learn as l2l

from model.models import MiniImagenetCNN, TaskManager
from method.maml import MAML
from method.regularizer import FilterReg, LinearReg, FilterSparseReg

#--------------------------Load Model------------------------------#
def getModel(args, cls_per_task, device):
    return TaskManager(cls_per_task, args.ways, args.hidden_size, args.num_layers, args.task_normalization, device).to(device)
    #return MiniImagenetCNN(args.ways, args.hidden_size, args.num_layers, args.task_normalization).to(device)

def getMetaAlgorithm(model, fast_lr, first_order):
    return MAML(model, lr=fast_lr, first_order=first_order)
    
def getRegularizer(convFilter, c_theta, linearReg, c_omega, sparseFilter):
    regularizator = []

    if convFilter:
        regularizator.append(FilterReg(c_theta))
    if linearReg:
        regularizator.append(LinearReg(c_omega))
    if sparseFilter:
        regularizator.append(FilterSparseReg(c_theta))

    return regularizator

#--------------------------Args-----------------------------------#
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def getArguments():
    parser = argparse.ArgumentParser(description='Meta-iteration')

    #---------------------Meta-Learning----------------------------#
    parser.add_argument('--ways', type=int, default=5)
    parser.add_argument('--shots', type=int, default=5)
    parser.add_argument('--num-iterations', type=int, default=100)
    parser.add_argument('--meta-batch-size', type=int, default=16)
    parser.add_argument('--adaptation-steps', type=int, default=5)
    parser.add_argument('--fast-lr', type=float, default=0.5)
    parser.add_argument('--first-order', type=str2bool, default=False)
    parser.add_argument('--meta-label', type=str, default='random', choices=['rotnet', 'random', 'supervised'])

    #---------------------Datasets---------------------------------#
    parser.add_argument('--dataset', type=str, default='tiny-imagenet', choices=['tiny-imagenet', 'random', 'cifar10'])
    parser.add_argument('--train-dataset', type=str, default='./')
    parser.add_argument('--val-dataset', type=str, default='./')
    parser.add_argument('--amount-split', type=int, default=10)

    #---------------------Model------------------------------------#
    parser.add_argument('--hidden-size', type=int, default=32)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--task-normalization', type=str2bool, default=True)

    #---------------------Regularization---------------------------#
    
    #---------------------Extras-----------------------------------#
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--num-data', type=int, default=-1)
    parser.add_argument('--save-model', type=str2bool, default=True)
    parser.add_argument('--meta-learn', type=str2bool, default=True)



    return parser

def saveValues(name_file, results, model, args):
    torch.save({
            'results': results,
            'args': args,
            'checkpoint': model.state_dict()
            }, name_file)