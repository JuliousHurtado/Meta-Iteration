import torch
import os
import argparse
import random
from collections import defaultdict

from torchvision import transforms

import learn2learn as l2l
from learn2learn.data.transforms import KShots, LoadData, NWays

from model.models import MiniImagenetCNN, TaskManager
from dataset.splitDataset import splitDataset
from dataset.randomDataset import RandomSet
from method.maml import MAML
from method.regularizer import FilterReg, LinearReg, FilterSparseReg

#--------------------------Dataset---------------------------------#
def getTinyImageNet(args):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(78),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    classes = os.listdir(args.train_dataset)
    random.shuffle(classes)

    data_loader = []
    cls_per_task = [args.ways]
    for i in range(args.amount_split):
        selected_classes = classes[i::int(len(classes)/args.amount_split)]
        cls_per_task.append(len(selected_classes))

        train_dataset = splitDataset(args.train_dataset, selected_classes, data_transforms['train'])
        val_dataset = splitDataset(args.val_dataset, selected_classes, data_transforms['val'])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

        meta_loader = None
        if args.meta_learn:
            create_bookkeeping(train_dataset)

            meta_transforms = [
                    l2l.data.transforms.NWays(train_dataset, args.ways),
                    l2l.data.transforms.KShots(train_dataset, 2*args.shots),
                    l2l.data.transforms.LoadData(train_dataset),
                    l2l.data.transforms.RemapLabels(train_dataset),
                    l2l.data.transforms.ConsecutiveLabels(train_dataset),
                ]

            meta_loader = l2l.data.TaskDataset(l2l.data.MetaDataset(train_dataset),
                                           task_transforms=meta_transforms)

        data_loader.append({ 'train': train_loader, 'val': val_loader, 'meta': meta_loader })

    return data_loader, cls_per_task

def getRandomDataset(args):
    data_loader = []
    cls_per_task = [args.ways]
    for i in range(args.amount_split):
        cls_per_task.append(10)
        train_dataset = RandomSet()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(RandomSet(), batch_size=args.batch_size, shuffle=True)
        meta_loader = None
        if args.meta_learn:
            create_bookkeeping(train_dataset)

            meta_transforms = [
                    l2l.data.transforms.NWays(train_dataset, args.ways),
                    l2l.data.transforms.KShots(train_dataset, 2*args.shots),
                    l2l.data.transforms.LoadData(train_dataset),
                    l2l.data.transforms.RemapLabels(train_dataset),
                    l2l.data.transforms.ConsecutiveLabels(train_dataset),
                ]
            meta_loader = l2l.data.TaskDataset(l2l.data.MetaDataset(train_dataset),
                                           task_transforms=meta_transforms)

        data_loader.append({ 'train': train_loader, 'val': val_loader, 'meta': meta_loader })

    return data_loader, cls_per_task

def create_bookkeeping(dataset):
    """
    Iterates over the entire dataset and creates a map of target to indices.
    Returns: A dict with key as the label and value as list of indices.
    """

    assert hasattr(dataset, '__getitem__'), \
            'Requires iterable-style dataset.'

    labels_to_indices = defaultdict(list)
    indices_to_labels = defaultdict(int)
    for i in range(len(dataset)):
        try:
            label = dataset[i][1]
            # if label is a Tensor, then take get the scalar value
            if hasattr(label, 'item'):
                label = dataset[i][1].item()
        except ValueError as e:
            raise ValueError(
                'Requires scalar labels. \n' + str(e))

        labels_to_indices[label].append(i)
        indices_to_labels[i] = label

    dataset.labels_to_indices = labels_to_indices
    dataset.indices_to_labels = indices_to_labels
    dataset.labels = list(dataset.labels_to_indices.keys())

#--------------------------Load Model------------------------------#
def getModel(args, cls_per_task, device):
    return TaskManager(cls_per_task, args.hidden_size, args.num_layers, args.task_normalization, device).to(device)
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
    parser.add_argument('--meta-warmup', type=int, default=5000)
    #---------------------Datasets---------------------------------#
    parser.add_argument('--dataset', type=str, default='tiny-imagenet', choices=['tiny-imagenet', 'random'])
    parser.add_argument('--train-dataset', type=str)
    parser.add_argument('--val-dataset', type=str)
    parser.add_argument('--amount-split', type=int, default=10)
    #---------------------Model------------------------------------#
    parser.add_argument('--hidden-size', type=int, default=32)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--task-normalization', type=str2bool, default=False)
    #---------------------Regularization---------------------------#
    #---------------------Extras-----------------------------------#
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--meta-learn', type=str2bool, default=True)
    parser.add_argument('--save-model', type=str2bool, default=True)

    return parser

def saveValues(name_file, results, model, args):
    torch.save({
            'results': results,
            'args': args,
            'checkpoint': model.state_dict()
            }, name_file)