import copy
import numpy as np

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from model.models import TaskNormalization

from method.maml import MAML

def trainingProcessTask(data_loader, learner, optimizer, regs, device):
    loss = nn.CrossEntropyLoss(reduction='mean')
    learner.train()
    running_loss = 0.0
    running_corrects = 0.0
    total_batch = 0.0
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.long().to(device)

        optimizer.zero_grad()

        out = learner(inputs)
        _, preds = torch.max(out, 1)
        l = loss(out, labels)

        if regs['reg']:
            l += regs['reg'](learner)
        
        l.backward()

        if regs['reg'] and regs['use']['gs_mask']:
            regs['reg'].setGradZero(learner)

        optimizer.step()

        if regs['reg'] and regs['use']['si']:
            regs['reg'].update_w(learner)        

        running_loss += l.item()
        running_corrects += torch.sum(preds == labels.data)
        total_batch += 1

    return running_loss / total_batch, running_corrects / len(data_loader.dataset)

def trainingForHat(data_loader, learner, optimizer, regs, device, task):
    loss = nn.CrossEntropyLoss(reduction='mean')
    learner.train()
    running_loss = 0.0
    running_corrects = 0.0
    total_batch = 0.0
    for i, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device)
        labels = labels.long().to(device)

        l, out = regs['reg'](learner, i, len(data_loader.dataset), task, loss, inputs, labels)

        _, preds = torch.max(out, 1)

        running_loss += l.item()
        running_corrects += torch.sum(preds == labels.data)
        total_batch += 1

    return running_loss / total_batch, running_corrects / len(data_loader.dataset)

def test_normal(model, data_loader, device):
    model.eval()
    correct = 0
    for input, target in data_loader:
        input, target = input.to(device), target.long().to(device)
        output = model(input)
        correct += (F.softmax(output, dim=1).max(dim=1)[1] == target).data.sum()
    return correct.item() / len(data_loader.dataset)

def test_normal_masks(model, data_loader, device, masks):
    temp_model = copy.deepcopy(model)
    temp_model.setMasks(masks)
    temp_model.eval()
    correct = 0
    for input, target in data_loader:
        input, target = input.to(device), target.long().to(device)
        output = temp_model(input)
        correct += (F.softmax(output, dim=1).max(dim=1)[1] == target).data.sum()
    return correct.item() / len(data_loader.dataset)

def test_for_hat(model, data_loader, device, task):
    model.eval()
    correct = 0
    for input, target in data_loader:
        input, target = input.to(device), target.long().to(device)
        output = model(task, input, 400)
        correct += (F.softmax(output, dim=1).max(dim=1)[1] == target).data.sum()
    return correct.item() / len(data_loader.dataset)

def adjust_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']/2

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

def addResults(model, data_generators, results, device, task, opti, all_tasks=False, final_acc=False, masks=None, re_train=False, use_hat=False):
    if all_tasks:
        for j in range(task+1):
            val_accuracy = test_normal(model, data_generators[j]['valid'], device)
            results[j]['valid_acc'].append(val_accuracy)
    else:
        if use_hat:
            val_accuracy = test_for_hat(model, data_generators[task]['valid'], device, task)
        else:
            val_accuracy = test_normal(model, data_generators[task]['valid'], device)
            results[task]['valid_acc'].append(val_accuracy)

    if final_acc:
        for j in range(task+1):
            if use_hat:
                test_accuracy = test_for_hat(model, data_generators[j]['test'], device, j)
            else:
                model.setLinearLayer(j)
                model.setTaskNormalizationLayer(j)

                if re_train:
                    m = copy.deepcopy(model.model)

                    params = []
                    parametersTask(m, params)
                    opti = optim.SGD(params, 0.0001, momentum=0.0, weight_decay=0.0)
                    for _ in range(5):
                        trainingProcessTask(data_generators[j]['sample'], m, opti, {'reg': False, 'use': {'gs_mask': False}}, device)

                    test_accuracy = test_normal_masks(m, data_generators[j]['test'], device, masks[j])
                
                else:
                    if masks:
                        test_accuracy = test_normal_masks(model, data_generators[j]['test'], device, masks[j])
                    else:
                        test_accuracy = test_normal(model, data_generators[j]['test'], device)

                results[j]['final_acc'].append(test_accuracy)