import torch
import random

from torch import nn
from torch.nn import functional as F

class SI(object):
    """
    @inproceedings{zenke2017continual,
        title={Continual Learning Through Synaptic Intelligence},
        author={Zenke, Friedemann and Poole, Ben and Ganguli, Surya},
        booktitle={International Conference on Machine Learning},
        year={2017},
        url={https://arxiv.org/abs/1703.04200}
    }
    """
    def __init__(self,  model: nn.Module, importance: float):
        self.damping_factor = 0.1
        self.importance = importance
        #self.initialize_reg_params(model)
        self.reg_params = None

    def initialize_reg_params(self, model):
        """initialize an omega for each parameter to zero"""
        self.reg_params = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.reg_params[n] = { 
                    'omega': p.clone().detach().zero_(), 
                    'init_val': p.data.clone(),
                    'prev_omega': p.clone().detach().zero_(),
                    'w': p.data.clone().zero_(),
                    'p': p.data.clone() }

    def update_w(self, model):
        #In each iteration
        for n, p in model.named_parameters():
            if p.requires_grad:
                if p.grad is not None:
                    self.reg_params[n]['w'].add_(-p.grad*(p.detach()-self.reg_params[n]['p']))
                self.reg_params[n]['p'] = p.detach().clone()

    def init_new_task(self, model, data_loader):
        #After completing training on a task, update the per-parameter regularization strength.

        if self.reg_params is None:
            self.initialize_reg_params(model)
            
        for n,p in model.named_parameters():
            if p.requires_grad:
                change = p.detach().clone() - self.reg_params[n]['init_val']
                self.reg_params[n]['omega'] += self.reg_params[n]['w']/(change**2 + self.damping_factor)
                self.reg_params[n]['init_val'] = p.detach().clone()

                self.reg_params[n]['w'] = p.data.clone().zero_()
                self.reg_params[n]['p'] = p.data.clone()

    def __call__(self, model):
        loss = 0

        for n,p in model.named_parameters():
            if p.requires_grad:
                _loss = self.reg_params[n]['omega']*(p-self.reg_params[n]['init_val'])**2
                loss += _loss.sum()
        return loss*self.importance
