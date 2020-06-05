import torch
import random
import time
import copy
import numpy as np
from torch import nn
from torch import optim
import torchvision

import torchvision.models as models

from model.models import TaskNormalization
from utils import getArguments, getModel

from dataset.multidataset import DatasetGen as multi_cls
from dataset.cifar100 import DatasetGen as cifar100
from dataset.pmnist import DatasetGen as pmnist

from dataset.datasets_utils import join_all_datasets

from train.meta_training import trainingProcessMeta
from train.task_training import trainingProcessTask, test_normal, addResults


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        #self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 3 * 3, 4096),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(4096, 4096),
            # nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):

        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x




def scratch_learn(args, device):
    if args.dataset == 'multi':
        data_generators = multi_cls(args)
        args.dataset_order = data_generators.datasets_names
    elif args.dataset == 'cifar100':
        data_generators = cifar100(args)
    elif args.dataset == 'pmnist':
        data_generators = pmnist(args)
        args.in_channels = 1
        
    cls_per_task = data_generators.taskcla

    model = getModel(args, cls_per_task, device)

    task_regs = { 'reg': None, 'use': {'ewc': False, 'gs_mask': False, 'mas': False, 'si': False}}

    results = {}
    for i in range(data_generators.num_task):
        net = copy.deepcopy(model)

        task_dataloader = data_generators.get(i)
        opti = adjustModelTask(net, i, args.lr)
        
        results[i] = {
            'meta_loss': [],
            'meta_acc': [],
            'train_acc': [],
            'train_loss': [],
            'valid_acc': [],
            'test_acc': [],
            'final_acc': [],
            'sparsity': [],
        }

        for e in range(args.epochs):
            loss_task, acc_task = trainingProcessTask(task_dataloader[i]['train'], net, opti, task_reg, device) 
            results[i]['train_loss'].append(loss_task)
            results[i]['train_acc'].append(acc_task)            
            print('Task: Task {4} Epoch [{0}/{1}] \t Train Loss: {2:1.4f} \t Train Acc {3:3.2f} %'.format(e, args.epochs, loss_task, acc_task*100, i+1), flush=True)
            
        addResults(net, task_dataloader, results, device, i, opti, False, True, None, False)

    for i in range(data_generators.num_task):
        print(results[i]['final_acc'])

def joint_learn(args, device):
    args.epochs = 250

    task_dataloader = {}
    if args.dataset == 'multi':
        data_generators = multi_cls(args)
        cls_per_task = data_generators.taskcla

        train = join_all_datasets(data_generators, 'train', cls_per_task)
        task_dataloader['train'] = torch.utils.data.DataLoader(train, shuffle=True, num_workers=4, batch_size=64)

        test = join_all_datasets(data_generators, 'test', cls_per_task)
        task_dataloader['test'] = torch.utils.data.DataLoader(test, shuffle=True, num_workers=4, batch_size=64)

        cls_per_task = [50]

    elif args.dataset == 'cifar100':
        # data_generators = cifar100(args)
        # cls_per_task = data_generators.taskcla

        # train = join_all_datasets(data_generators, 'train', cls_per_task)
        # task_dataloader['train'] = torch.utils.data.DataLoader(train, shuffle=True, num_workers=4, batch_size=64)

        # test = join_all_datasets(data_generators, 'test', cls_per_task)
        # task_dataloader['test'] = torch.utils.data.DataLoader(test, shuffle=True, num_workers=4, batch_size=64)

        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)
            ])

        cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        task_dataloader['train'] = torch.utils.data.DataLoader(
            cifar100_training, shuffle=True, num_workers=4, batch_size=64)

        cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        task_dataloader['test'] = torch.utils.data.DataLoader(
            cifar100_test, shuffle=True, num_workers=4, batch_size=64)

        cls_per_task = [100]

    elif args.dataset == 'pmnist':
        data_generators = pmnist(args)
        args.in_channels = 1
        cls_per_task = data_generators.taskcla

        train = join_all_datasets(data_generators, 'train', cls_per_task)
        task_dataloader['train'] = torch.utils.data.DataLoader(train, shuffle=True, num_workers=4, batch_size=64)

        test = join_all_datasets(data_generators, 'test', cls_per_task)
        task_dataloader['test'] = torch.utils.data.DataLoader(test, shuffle=True, num_workers=4, batch_size=64)

        cls_per_task = [100]

    #model = getModel(args, cls_per_task, device)
    model = AlexNet(100).to(device)
    task_reg = { 'reg': None, 'use': {'ewc': False, 'gs_mask': False, 'mas': False, 'si': False}}

    #model.setLinearLayer(0)
    #opti = optim.SGD(model.parameters(), args.lr, momentum=0.0, weight_decay=0.0)
    #opti = optim.Adam(model.parameters(), args.lr)

    opti = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(opti, milestones=[60, 120, 160], gamma=0.2)

    print(model)
    for e in range(args.epochs):
        if e > 20:
            train_scheduler.step(e)

        loss_task, acc_task = trainingProcessTask(task_dataloader['train'], model, opti, task_reg, device)          
        print('Task: Task {4} Epoch [{0}/{1}] \t Train Loss: {2:1.4f} \t Train Acc {3:3.2f} %'.format(e, args.epochs, loss_task, acc_task*100, args.dataset), flush=True)
            
    acc_test = test_normal(model, task_dataloader['test'], device)
    print("Test Accuracy in {}: {}".format(args.dataset, acc_test))

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

    # scratch_learn(args, device)
    joint_learn(args, device)