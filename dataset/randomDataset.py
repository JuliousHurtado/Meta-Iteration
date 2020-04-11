#!/usr/bin/env python3
import torch
from torch.utils.data import Dataset

class RandomSet(Dataset):

    def __init__(self, root = '', transform=None, target_transform=None, download=False, to_color = False):
        self.transform = transform

        self.x = torch.rand(500, 3, 84, 84)
        self.y = torch.randint(0, 5, (500,))

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, item):
        x = self.x[item]
        y = self.y[item]

        return x, y