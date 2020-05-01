import torch
import os
import argparse
from collections import defaultdict

import learn2learn as l2l

from model.models import MiniImagenetCNN, TaskManager
from method.maml import MAML
from method.regularizer import FilterReg, LinearReg, FilterSparseReg, GroupMask
# from method.ewc import EWC
# from method.MAS import MAS


#--------------------------Load Model------------------------------#
def getModel(args, cls_per_task, device):
    return TaskManager(cls_per_task, args.ways, args.hidden_size, args.num_layers, args.task_normalization, device).to(device)
    #return MiniImagenetCNN(args.ways, args.hidden_size, args.num_layers, args.task_normalization).to(device)

def getMetaAlgorithm(model, fast_lr, first_order):
    return MAML(model, lr=fast_lr, first_order=first_order)
    
def getMetaRegularizer(convFilter, c_theta, linearReg, c_omega, sparseFilter):
    use_meta_reg = {'linear': False, 'filter': False, 'sparse': False}
    regs = []

    if convFilter:
        regs.append(FilterReg(c_theta))
        use_meta_reg['filter'] = True
    if linearReg:
        regs.append(LinearReg(c_omega))
        use_meta_reg['linear'] = True
    if sparseFilter:
        regs.append(FilterSparseReg(c_theta))
        use_meta_reg['sparse'] = True

    return {'reg': regs, 'use': use_meta_reg}

def getTaskRegularizer(task_reg, ewc_importance, c_theta, c_lambda, c_sparse):
    reg_used = {'ewc': False, 'gs_mask': False, 'mas': False, 'si': False}
    reg_used[task_reg] = True

    reg = None
    if task_reg == 'ewc':
        pass

    if task_reg == 'gs_mask':
        reg = GroupMask(c_theta)

    if task_reg == 'mas':
        pass

    if task_reg == 'si':
        pass

    return { 'reg': reg, 'use': reg_used}

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
    parser.add_argument('--fast-lr', type=float, default=0.05)
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
    parser.add_argument('--meta-reg-linear', type=str2bool, default=False)
    parser.add_argument('--cost-omega', type=float, default=0.01)
    parser.add_argument('--meta-reg-filter', type=str2bool, default=False)
    parser.add_argument('--meta-reg-sparse', type=str2bool, default=False)
    parser.add_argument('--cost-theta', type=float, default=0.01)

    parser.add_argument('--task-reg', type=str, default='', choices=['','ewc', 'mas', 'si', 'gs_mask'])
    parser.add_argument('--ewc-importance', type=float, default=100)
    parser.add_argument('--reg-theta', type=float, default=0.05)
    parser.add_argument('--reg-lambda', type=float, default=0.1)
    parser.add_argument('--reg-sparse', type=float, default=0.001)

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