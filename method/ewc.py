from copy import deepcopy
import torch

from torch import nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EWC(object):
    def __init__(self, model: nn.Module, importance: float):

        self.model = model
        self.importance = importance

        #self.init_new_task(model, dataset)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.to(device)

        self.model.eval()
        for input in self.dataset:
            self.model.zero_grad()
            input = input.to(device).unsqueeze(0)
            output = self.model(input).view(1, -1)
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def init_new_task(self, model, data_loader):
        self.model = model
        self.dataset = data_loader
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = p.to(device)

    def __call__(self, model: nn.Module):
        loss = 0.0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss*self.importance