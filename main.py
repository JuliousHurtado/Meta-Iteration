import torch
import random
import time
import numpy as np
from torch import nn
from torch import optim

from model.models import TaskNormalization
from utils import getArguments, getTinyImageNet, getRandomDataset, getModel, getMetaAlgorithm, saveValues
from train.meta_training import trainingProcessMeta
from train.task_training import trainingProcessTask, addResults

def parametersMAML(network, all_layers):
    for layer in network.children():
        # if type(layer) == torch.nn.Sequential: # if sequential layer, apply recursively to layers in sequential layer
        #     remove_sequential(layer, all_layers)
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.Linear):
            for p in layer.parameters():
                all_layers.append(p)
        if list(layer.children()) != [] and not isinstance(layer, TaskNormalization): # if leaf node, add it to list
            parametersMAML(layer, all_layers)

def parametersTask(network, all_layers, task=False):
    for layer in network.children():
        if isinstance(layer, TaskNormalization):
            parametersTask(layer, all_layers, True)
        if list(layer.children()) == [] and task: # if leaf node, add it to list
            for p in layer.parameters():
                all_layers.append(p)
        if list(layer.children()) != []:
            parametersTask(layer, all_layers)
        if isinstance(layer, torch.nn.Linear):
            for p in layer.parameters():
                all_layers.append(p)

def checkParamsSum(model):
    params = {}
    for n,p in model.named_parameters():
        params[n] = p.sum()

    # print("Meta")
    # for n,p in model.named_parameters():
    #     if params[n] != p.sum():
    #         print(n)


def adjustModelMeta(model, task_linear, task_norm, lr):
    model.setLinearLayer(task_linear)
    model.setTaskNormalizationLayer(task_norm)
    params = []
    parametersMAML(model, params)
    return optim.Adam(params, lr)

def adjustModelTask(model, task, lr):
    model.setLinearLayer(task)
    model.setTaskNormalizationLayer(task)
    params = []
    parametersTask(model, params)
    return optim.Adam(params, lr)

def adjustComplete(model, task, lr):
    model.setLinearLayer(task)
    model.setTaskNormalizationLayer(task)

    return optim.SGD(model.parameters(), lr, weight_decay=1e-4)    

def warmup(args, model, task_dataloader, loss, device, meta_warm, task_warm, lr):
    if meta_warm:
        print("Starting WarmUp Meta parameters")
        opti_meta = adjustModelMeta(model, 0, 1, lr)  
        for i in range(args.meta_warmup):
            loss_meta, acc_meta = trainingProcessMeta(args, model, opti_meta, loss, task_dataloader['meta'], [], device)

    if task_warm:
        print("Starting WarmUp Task parameters")
        opti_task = adjustModelTask(model, 1, lr)
        for i in range(20):
            loss_task, acc_task = trainingProcessTask(task_dataloader['train'], model, loss, opti_task, [], device, None) 

    print("Finishing WarmUp")

def main(args, data_generators, model, device):
    lr = args.lr
    loss = nn.CrossEntropyLoss(reduction='mean')

    warmup(args, model, data_generators[0], loss, device, args.meta_warmup, args.task_warmup, lr)

    results = {}
    for i in range(args.amount_split):
        results[i] = {
            'meta_loss': [],
            'meta_acc': [],
            'train_acc': [],
            'train_loss': [],
            'test_acc': [],
            'final_acc': [],
        }

        task_dataloader = data_generators[i]
        
        if args.task_complete:
            opti_task = adjustComplete(model, i+1, lr)

        for e in range(args.epochs):

            if args.meta_learn:
                opti_meta = adjustModelMeta(model, 0,i+1, lr)            
                loss_meta, acc_meta = trainingProcessMeta(args, model, opti_meta, loss, task_dataloader['meta'], [], device)
                results[i]['meta_loss'].append(loss_meta)
                results[i]['meta_acc'].append(acc_meta)
                print('Meta: Task {4} Epoch [{0}/{1}] \t Train Loss: {2:1.4f} \t Train Acc {3:3.2f} %'.format(e, args.epochs, loss_meta, acc_meta*100, i+1))

            if not args.task_complete:
                opti_task = adjustModelTask(model, i+1, lr)
            
            loss_task, acc_task = trainingProcessTask(task_dataloader['train'], model, loss, opti_task, [], device, None) 
            results[i]['train_loss'].append(loss_task)
            results[i]['train_acc'].append(acc_task)            
            print('Task: Task {4} Epoch [{0}/{1}] \t Train Loss: {2:1.4f} \t Train Acc {3:3.2f} %'.format(e, args.epochs, loss_task, acc_task*100, i+1), flush=True)
            
            addResults(model, data_generators, results, device, i, False)

        addResults(model, data_generators, results, device, i, False, True)

        if args.save_model:
            name_file = '{}/{}_{}_{}_{}_{}'.format('results', 'temp', args.meta_warmup, args.meta_learn, args.task_warmup, args.task_complete)
            saveValues(name_file, results, model.module, args)

    if args.save_model:
        name_file = '{}/{}_{}_{}_{}_{}'.format('results', str(time.time()), args.meta_warmup, args.meta_learn, args.task_warmup, args.task_complete)
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

    model = getModel(args, cls_per_task, device)
    meta_model = getMetaAlgorithm(model, args.fast_lr, args.first_order)

    # all_layers = []
    # parametersTask(meta_model.module, all_layers)
    # print(all_layers)
    # regs = getRegularizer( 
    #                 args.filter_reg, args.cost_theta,
    #                 args.linear_reg, args.cost_omega,
    #                 args.sparse_reg)

    main(args, data_generators, meta_model, device)