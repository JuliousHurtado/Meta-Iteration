from copy import deepcopy
import torch

from torch import nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HAT(object):
    def __init__(self, model: nn.Module, smax: float):
        self.smax=400
        self.lamb=0.75
        self.model = model
        self.mask_pre = None
        self.clipgrad=10000
        self.thres_cosh=50

    def init_new_task(self, model, data_loader, task):
        mask=self.model.mask(task,s=self.smax)
        for i in range(len(mask)):
            mask[i]=mask[i].data.clone()
        if t==0:
            self.mask_pre=mask
        else:
            for i in range(len(self.mask_pre)):
                self.mask_pre[i]=torch.max(self.mask_pre[i],mask[i])

        self.mask_back={}
        for n, _ in self.model.named_parameters():
            vals=self.model.get_view_for(n, self.mask_pre)
            if vals is not None:
                self.mask_back[n]=1-vals

    def criterion(self,outputs,targets,masks):
        reg=0
        count=0
        if self.mask_pre is not None:
            for m,mp in zip(masks,self.mask_pre):
                aux=1-mp
                reg+=(m*aux).sum()
                count+=aux.sum()
        else:
            for m in masks:
                reg+=m.sum()
                count+=np.prod(m.size()).item()
        reg/=count
        return self.ce(outputs,targets)+self.lamb*reg,reg

    def __call__(self, model, e, len_data, t, criterion, inputs, targets):
        s=(self.smax-1/self.smax)*e/len_data+1/self.smax
        self.ce=criterion
        
        output,masks=model.forward(t, inputs, s=s)
        loss,_=self.criterion(output,targets,masks)

        optimizer.zero_grad()
        loss.backward()

        # Restrict layer gradients in backprop
        if t>0:
            for n,p in model.named_parameters():
                if n in self.mask_back:
                    p.grad.data*=self.mask_back[n]

        # Compensate embedding gradients
        for n,p in model.named_parameters():
            if n.startswith('e'):
                num=torch.cosh(torch.clamp(s*p.data,-self.thres_cosh,self.thres_cosh))+1
                den=torch.cosh(p.data)+1
                p.grad.data*=self.smax/s*num/den

        # Apply step
        torch.nn.utils.clip_grad_norm(model.parameters(),self.clipgrad)
        optimizer.step()

        # Constrain embeddings
        for n,p in self.model.named_parameters():
            if n.startswith('e'):
                p.data=torch.clamp(p.data,-self.thres_emb,self.thres_emb)

        return loss