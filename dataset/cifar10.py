from torchvision import datasets
import torch
import random
from PIL import Image
import os
import sys
import numpy as np

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

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