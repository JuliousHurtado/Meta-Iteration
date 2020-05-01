import torch
import random
import time
import numpy as np
from torch import nn
from torch import optim

from model.models import TaskNormalization
from utils import getArguments, getModel, getMetaAlgorithm, saveValues, getMetaRegularizer
from dataset.getDataloaders import getTinyImageNet, getRandomDataset, getDividedCifar10
from train.meta_training import trainingProcessMeta
from train.task_training import trainingProcessTask, addResults


def checkParamsSum(model):
    params = {}
    for n,p in model.named_parameters():
        params[n] = p.sum()

def stringRegUsed(regs, t='meta'):
    s = t

    for k in regs:
        s = s + '_' + k + '_' + str(regs[k])[0]

    return s

def adjustModelTask(model, task, lr, linear=True, norm=True):
    if linear:
        model.setLinearLayer(task)
    if norm: 
        model.setTaskNormalizationLayer(task)
    
    return optim.Adam(model.parameters(), lr)

def main(args, data_generators, model, device, meta_reg, task_reg):
    lr = args.lr
    loss = nn.CrossEntropyLoss(reduction='mean')

    results = {}
    for i in range(len(data_generators)):
        results[i] = {
            'meta_loss': [],
            'meta_acc': [],
            'train_acc': [],
            'train_loss': [],
            'test_acc': [],
            'final_acc': [],
        }

        task_dataloader = data_generators[i]
        opti = adjustModelTask(model, i, lr)

        for e in range(args.epochs):

            if args.meta_learn and e % 5 == 0:
                opti_meta = adjustModelTask(model, 'meta', lr, norm=False)            
                loss_meta, acc_meta = trainingProcessMeta(args, model, opti_meta, loss, task_dataloader['meta'], meta_reg['reg'], device)
                results[i]['meta_loss'].append(loss_meta)
                results[i]['meta_acc'].append(acc_meta)
                print('Meta: Task {4} Epoch [{0}/{1}] \t Train Loss: {2:1.4f} \t Train Acc {3:3.2f} %'.format(e, args.epochs, loss_meta, acc_meta*100, i+1))

                #opti = adjustModelTask(model, i, lr)  
            
            loss_task, acc_task = trainingProcessTask(task_dataloader['train'], model, loss, opti, task_reg, device) 
            results[i]['train_loss'].append(loss_task)
            results[i]['train_acc'].append(acc_task)            
            print('Task: Task {4} Epoch [{0}/{1}] \t Train Loss: {2:1.4f} \t Train Acc {3:3.2f} %'.format(e, args.epochs, loss_task, acc_task*100, i+1), flush=True)
            
            addResults(model, data_generators, results, device, i, False)

        addResults(model, data_generators, results, device, i, False, True)

        if args.save_model:
            name_file = '{}/{}_{}_{}_{}_{}_{}'.format('results', args.dataset, i, args.meta_learn, args.task_normalization, args.meta_label, stringRegUsed(meta_reg['use']))
            saveValues(name_file, results, model.module, args)

    if args.save_model:
        name_file = '{}/{}_{}_{}_{}_{}_{}_{}'.format('results', 'final', args.dataset, args.meta_learn, args.task_normalization, args.meta_label, stringRegUsed(meta_reg['use']), str(time.time()))
        saveValues(name_file, results, model.module, args)

if __name__ == '__main__':
    parser = getArguments()
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if use_cuda else "cpu")

    if args.dataset == 'tiny-imagenet':
        data_generators, cls_per_task = getTinyImageNet(args)
    elif args.dataset == 'random':
        data_generators, cls_per_task = getRandomDataset(args)
    elif args.dataset == 'cifar10':
        data_generators, cls_per_task = getDividedCifar10(args)

    model = getModel(args, cls_per_task, device)
    meta_model = getMetaAlgorithm(model, args.fast_lr, args.first_order)

    meta_regs = getMetaRegularizer( 
                    args.meta_reg_filter, args.cost_theta,
                    args.meta_reg_linear, args.cost_omega,
                    args.meta_reg_sparse)

    main(args, data_generators, meta_model, device, meta_regs, [])