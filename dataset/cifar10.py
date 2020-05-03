import torch
import random
import copy
import numpy as np
from collections import defaultdict
from PIL import Image

from torchvision import transforms
import torchvision.transforms.functional as F

import learn2learn as l2l

from dataset.datasets_utils import create_bookkeeping


from torchvision import datasets
from PIL import Image
import os
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle



class DatasetGen(object):
    def __init__(self, args):
        super(DatasetGen, self).__init__()

        self.batch_size = args.batch_size
        self.pc_valid = 0.15
        self.root = './data'
        self.args = args

        self.num_task = 3
        self.inputsize = [3,32,32]

        self.num_workers = 4
        self.pin_memory = True

        self.taskcla = []
        self.labels = [[0,1,2,3],[4,5,6],[7,8,9]]
        for i in range(self.num_task):
            self.taskcla.append(len(self.labels[i]))
        print('taskcla =', self.taskcla)

        self.data_loader = []
        self.set_dataloader()

    def set_dataloader(self):
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        for task in range(self.num_task):
            train_dataset = DividedCIFAR10('data', train=True, labels = self.labels[task], transform = data_transforms['train'], args = self.args)
            test_dataset = DividedCIFAR10('data', train=False, labels = self.labels[task], transform = data_transforms['val'], args = self.args)
            
            split = int(np.floor(self.pc_valid * len(train_dataset)))
            train_split, valid_split = torch.utils.data.random_split(train_dataset, [len(train_dataset) - split, split])

            if self.args.meta_learn:
                meta_dataset = copy.deepcopy(train_dataset)
                meta_loader = self.get_meta_loader(meta_dataset)
            else:
                meta_loader = None

            train_loader = torch.utils.data.DataLoader(train_split, batch_size=self.batch_size, num_workers=self.num_workers,
                                                   pin_memory=self.pin_memory,shuffle=True)
            valid_loader = torch.utils.data.DataLoader(valid_split, batch_size=self.batch_size,
                                                       num_workers=self.num_workers, pin_memory=self.pin_memory,shuffle=False)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                                      pin_memory=self.pin_memory,shuffle=False)

            self.data_loader.append({ 'train': train_loader, 'valid': valid_loader, 'test': test_loader, 'meta': meta_loader})

    def get(self, task_id):
        return self.data_loader

    def get_meta_loader(self, meta_dataset):
        create_bookkeeping(meta_dataset, self.args.ways, self.args.meta_label)

        meta_transforms = [
                    l2l.data.transforms.NWays(meta_dataset, self.args.ways),
                    l2l.data.transforms.KShots(meta_dataset, 2*self.args.shots),
                    l2l.data.transforms.LoadData(meta_dataset),
                    l2l.data.transforms.RemapLabels(meta_dataset),
                    l2l.data.transforms.ConsecutiveLabels(meta_dataset),
                ]

        meta_loader = l2l.data.TaskDataset(l2l.data.MetaDataset(meta_dataset),
                                           task_transforms=meta_transforms)

        return meta_loader



class PermutedMNIST(datasets.MNIST):

    def __init__(self, root="mnist", train=True, permute_idx=None, transform = None):
        super(PermutedMNIST, self).__init__(root, train, download=True)
        assert len(permute_idx) == 28 * 28
        
        self.data = torch.stack([img.float().view(-1)[permute_idx] / 255
                                           for img in self.data])

        self.transform = transform

    def __getitem__(self, index):

        img, target = self.data[index], int(self.targets[index])

        if self.transform:
            img = self.transform(img)

        return img, target

    def get_sample(self, sample_size):
        sample_idx = random.sample(range(len(self)), sample_size)
        return [img for img in self.data[sample_idx]]

class DividedCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True,
                 transform=None,
                 target_transform=None,
                 download=True, 
                 labels = [],
                 args = None):
        super(DividedCIFAR10, self).__init__(root, train, transform, target_transform, download)
        self.transform = transform
        self.target_transform = None
        self.root = root

        self.train = train  # training set or test set

        #if download:
        #    self.download()

        #if not self._check_integrity():
        #    raise RuntimeError('Dataset not found or corrupted.' +
        #                       ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                
                if 'labels' in entry:
                    lab = entry['labels']
                else:
                    lab = entry['fine_labels']

                for i,l in enumerate(lab):
                    if l in labels and (len(self.data) < args.num_data or args.num_data == -1):
                        self.data.append(entry['data'][i])
                        self.targets.append(labels.index(l))
                        
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()        

    def get_sample(self, sample_size):
        sample_idx = random.sample(range(len(self)), sample_size)
        temp = []
        for img in self.data[sample_idx]:
            if self.transform:
                img = self.transform(Image.fromarray(img)).unsqueeze(0)
            temp.append(img)
        return temp