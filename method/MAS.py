import torch
import torch.optim as optim
import copy

from torch import nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class MAS(object):
    def __init__(self, model: nn.Module, importance: float):
        #self.initialize_reg_params(model)
        self.reg_params = None
        self.importance = importance
        #self.update_omega(model, dataset)

    def initialize_reg_params(self, model):
        """initialize an omega for each parameter to zero"""
        self.reg_params = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.reg_params[n] = { 
                    'omega': p.clone().detach().zero_(), 
                    'init_val': p.data.clone(),
                    'prev_omega': p.clone().detach().zero_() }

    def initialize_store_reg_params(self, model):
        """set omega to zero but after storing its value in a temp omega in which later we can accumolate them both"""
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.reg_params[n]['prev_omega'] = copy.deepcopy(self.reg_params[n]['omega'])
                self.reg_params[n]['omega'] = p.clone().detach().zero_()
                self.reg_params[n]['init_val'] = p.data.clone()

    def init_new_task(self, model, data_loader):
        if self.reg_params is None:
            self.initialize_reg_params(model)
        else:
            self.initialize_store_reg_params(model)
        self.update_omega(model, data_loader)        

    def update_omega(self, model, data_loader):
        for i, input in enumerate(data_loader):
            # get the inputs
            input = input.to(device).unsqueeze(0)
            output = model(input).view(1,-1)
            label = output.max(1)[1].view(-1)

            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in model.named_parameters():
                if p.requires_grad:
                    self.reg_params[n]['omega'] += 1e-2*(p.grad.data.abs_() + self.reg_params[n]['prev_omega']*i)/(i+1)

    def __call__(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                _loss = 2*(p.data - self.reg_params[n]['init_val']).abs_()*self.reg_params[n]['omega']
                loss += _loss.sum()
        return loss*self.importance