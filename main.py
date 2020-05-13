import torch
import random
import time
import copy
import numpy as np
from torch import nn
from torch import optim

from model.models import TaskNormalization
from utils import getArguments, getModel, getMetaAlgorithm, saveValues, getMetaRegularizer, getTaskRegularizer, getEWC
#from dataset.getDataloaders import getTinyImageNet, getRandomDataset, getDividedCifar10
from dataset.cifar10 import DatasetGen as cifar10
from dataset.multidataset import DatasetGen as multi_cls

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
    
    return optim.SGD(model.parameters(), lr, momentum=0.0, weight_decay=0.0)

def main(args, data_generators, model, device, meta_reg, task_reg):
    lr = args.lr
    masks = {}

    results = {}
    for i in range(data_generators.num_task):
        results[i] = {
            'meta_loss': [],
            'meta_acc': [],
            'train_acc': [],
            'train_loss': [],
            'valid_acc': [],
            'test_acc': [],
            'final_acc': [],
        }

        task_dataloader = data_generators.get(i)
        opti = adjustModelTask(model, i, lr)

        for e in range(args.epochs):

            if args.meta_learn and e % 5 == 0 and e < args.final_meta and e > 0:
                if args.meta_label == 'ewc':
                    loss_meta, acc_meta = 0, 0
                    if i > 0:
                        opti_meta = adjustModelTask(model, 'meta', args.meta_lr, norm=True) 
                        reg = getEWC(i, data_generators.train_set, model, args.ewc_importance, args.sample_size)
                        loss_meta, acc_meta = trainingProcessTask(task_dataloader[i]['train'], model, opti_meta, reg, device) 
                    
                else:
                    opti_meta = adjustModelTask(model, 'meta', args.meta_lr, norm=True)            
                    loss_meta, acc_meta = trainingProcessMeta(args, model, opti_meta, task_dataloader[i]['meta'], meta_reg['reg'], device)
                
                results[i]['meta_loss'].append(loss_meta)
                results[i]['meta_acc'].append(acc_meta)
                print('Meta: Task {4} Epoch [{0}/{1}] \t Train Loss: {2:1.4f} \t Train Acc {3:3.2f} %'.format(e, args.epochs, loss_meta, acc_meta*100, i+1))

                adjustModelTask(model, i, lr)  
            
            loss_task, acc_task = trainingProcessTask(task_dataloader[i]['train'], model, opti, task_reg, device) 
            results[i]['train_loss'].append(loss_task)
            results[i]['train_acc'].append(acc_task)            
            print('Task: Task {4} Epoch [{0}/{1}] \t Train Loss: {2:1.4f} \t Train Acc {3:3.2f} %'.format(e, args.epochs, loss_task, acc_task*100, i+1), flush=True)
            
            addResults(model, task_dataloader, results, device, i, opti, False, False)

        if task_reg['use']['gs_mask']:
            task_reg['reg'].setMasks(model)
            masks[i] = copy.deepcopy(task_reg['reg'].masks)

        addResults(model, task_dataloader, results, device, i, opti, False, True, masks, args.re_train)
        
        for j in range(i+1):
            print(results[j]['final_acc'])

        if args.save_model:
            name_file = '{}/{}_{}_{}_{}_{}_{}_{}'.format('results', args.dataset, i, args.meta_learn, args.task_normalization, args.meta_label, stringRegUsed(meta_reg['use']), args.task_reg)
            saveValues(name_file, results, model.module, args)

    if args.save_model:
        name_file = '{}/{}_{}_{}_{}_{}_{}_{}_{}'.format('results', 'final', args.dataset, args.meta_learn, args.task_normalization, args.meta_label, stringRegUsed(meta_reg['use']), args.task_reg, str(time.time()))
        saveValues(name_file, results, model.module, args)

    for i in range(data_generators.num_task):
        print(results[i]['final_acc'])

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

    if args.dataset == 'cifar10':
        data_generators = cifar10(args)
    elif args.dataset == 'multi':
        data_generators = multi_cls(args)
        args.dataset_order = data_generators.datasets_names

    cls_per_task = data_generators.taskcla

    model = getModel(args, cls_per_task, device)
    meta_model = getMetaAlgorithm(model, args.fast_lr, args.first_order)

    meta_regs = getMetaRegularizer( 
                    args.meta_reg_filter, args.cost_theta,
                    args.meta_reg_linear, args.cost_omega,
                    args.meta_reg_sparse)

    task_regs = getTaskRegularizer(
                    args.task_reg, args.ewc_importance,
                    args.reg_theta, args.reg_lambda,
                    args.reg_sparse)

    main(args, data_generators, meta_model, device, meta_regs, task_regs)