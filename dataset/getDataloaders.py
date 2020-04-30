import torch
import random
import numpy as np
from collections import defaultdict
from PIL import Image

from torchvision import transforms
import torchvision.transforms.functional as F

import learn2learn as l2l

from dataset.splitDataset import splitDataset
from dataset.randomDataset import RandomSet
from dataset.cifar10 import DividedCIFAR10


#--------------------------Dataset---------------------------------#
def getTinyImageNet(args):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(84),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(84),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    classes = os.listdir(args.train_dataset)
    random.shuffle(classes)

    data_loader = []
    cls_per_task = []
    for i in range(args.amount_split):
        selected_classes = classes[i::int(len(classes)/args.amount_split)]
        cls_per_task.append(len(selected_classes))

        train_dataset = splitDataset(args.train_dataset, selected_classes, data_transforms['train'])
        val_dataset = splitDataset(args.val_dataset, selected_classes, data_transforms['val'])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

        meta_loader = None
        if args.meta_learn or args.meta_warmup:
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
    cls_per_task = []
    for i in range(args.amount_split):
        cls_per_task.append(10)
        train_dataset = RandomSet()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(RandomSet(), batch_size=args.batch_size, shuffle=True)
        meta_loader = None
        if args.meta_learn or args.meta_warmup:
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

def getDividedCifar10(args):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(84),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(84),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    labels = [[0,1,2,3],[4,5,6],[7,8,9]]

    data_loader = []
    cls_per_task = []
    for task in range(3):
        train_dataset = DividedCIFAR10('data', train=True, labels = labels[task], transform = data_transforms['train'], args = args)
        val_dataset = DividedCIFAR10('data', train=False, labels = labels[task], transform = data_transforms['val'], args = args)
        

        meta_loader = None
        if args.meta_learn:
            meta_dataset = DividedCIFAR10('data', train=True, labels = labels[task], transform = data_transforms['train'], args = args)
            create_bookkeeping(meta_dataset, args.ways)

            meta_transforms = [
                    l2l.data.transforms.NWays(meta_dataset, args.ways),
                    l2l.data.transforms.KShots(meta_dataset, 2*args.shots),
                    l2l.data.transforms.LoadData(meta_dataset),
                    l2l.data.transforms.RemapLabels(meta_dataset),
                    l2l.data.transforms.ConsecutiveLabels(meta_dataset),
                ]

            meta_loader = l2l.data.TaskDataset(l2l.data.MetaDataset(meta_dataset),
                                           task_transforms=meta_transforms)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

        data_loader.append({ 'train': train_loader, 'val': val_loader, 'meta': meta_loader})
        cls_per_task.append(len(labels[task]))

    return data_loader, cls_per_task

def create_bookkeeping(dataset, ways):
    """
    Iterates over the entire dataset and creates a map of target to indices.
    Returns: A dict with key as the label and value as list of indices.
    """

    assert hasattr(dataset, '__getitem__'), \
            'Requires iterable-style dataset.'

    labels_to_indices = defaultdict(list)
    indices_to_labels = defaultdict(int)

    ############################
    #
    # No sabes si esto funciona, podemos comparar con asignar 
    # labels al azar en cada epoca
    #
    ############################
    #------- RotNet Unsupervised-------#
    # data = []
    # angles = list(range(0,360,int(360/ways)))
    # for i in range(len(dataset)):
    #     angle = random.choice(angles)
    #     img = Image.fromarray(dataset.data[i])
    #     new_img = np.array(F.rotate(img, angle, False, False, None, None))
    #     label = angles.index(angle)

    #     data.append(new_img)

    #     labels_to_indices[label].append(i)
    #     indices_to_labels[i] = label
    # dataset.data = data

    #-------Random Unsupervised-------#
    labels = list(range(ways))
    for i in range(len(dataset)):
        l = random.choice(labels)

        labels_to_indices[l].append(i)
        indices_to_labels[i] = l

    #------- Original (Supervised) ---#
    # for i in range(len(dataset)):
    #     try:
    #         label = dataset[i][1]
    #         # if label is a Tensor, then take get the scalar value
    #         if hasattr(label, 'item'):
    #             label = dataset[i][1].item()
    #     except ValueError as e:
    #         raise ValueError(
    #             'Requires scalar labels. \n' + str(e))

    #     labels_to_indices[label].append(i)
    #     indices_to_labels[i] = label

    dataset.labels_to_indices = labels_to_indices
    dataset.indices_to_labels = indices_to_labels
    dataset.labels = list(dataset.labels_to_indices.keys())

    # print(dataset.labels_to_indices)