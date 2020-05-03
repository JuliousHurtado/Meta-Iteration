import copy
import numpy as np

import torch
from torch.nn import functional as F

def trainingProcessTask(data_loader, learner, loss, optimizer, regs, device):
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

        if regs['use']['gs_mask']:
            regs['reg'].setGradZero(learner)

        optimizer.step()

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
    temp_model = copy.deepcopy(model.model)

    temp_model.eval()
    correct = 0
    for input, target in data_loader:
        input, target = input.to(device), target.long().to(device)
        output = temp_model(input)
        correct += (F.softmax(output, dim=1).max(dim=1)[1] == target).data.sum()
    return correct.item() / len(data_loader.dataset)

def adjust_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']/2

def addResults(model, data_generators, results, device, task, opti, all_tasks=False, final_acc=False, masks=None):
    if all_tasks:
        for j in range(task+1):
            val_accuracy = test_normal(model, data_generators[j]['valid'], device)
            results[j]['valid_acc'].append(val_accuracy)
    else:
        val_accuracy = test_normal(model, data_generators[task]['valid'], device)
        results[task]['valid_acc'].append(val_accuracy)

        if len(results[task]['valid_acc']) > 5:
            if np.mean(results[task]['valid_acc'][-6:-1]) > results[task]['valid_acc'][-1]:
                adjust_learning_rate(opti)

    if final_acc:
        for j in range(task+1):
            model.setLinearLayer(j)
            model.setTaskNormalizationLayer(j)

            if masks:
                test_accuracy = test_normal_masks(model, data_generators[j]['test'], device, masks[j])
            else:
                test_accuracy = test_normal(model, data_generators[j]['test'], device)
            
            results[j]['final_acc'].append(test_accuracy)