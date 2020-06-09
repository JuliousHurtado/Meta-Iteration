import sys,time
import numpy as np
import torch

########################################################################################################################

class Appr(object):

    def __init__(self,model,nepochs=50,sbatch=64,lr=0.003,lr_min=1e-4,lr_factor=3,lr_patience=5,clipgrad=10000,
                                    lamb=0.75,smax=400,args=None,device='cpu'):
        self.model=model

        self.nepochs=nepochs
        self.sbatch=sbatch
        self.lr=lr
        self.lr_min=lr_min
        self.lr_factor=lr_factor
        self.lr_patience=lr_patience
        self.clipgrad=clipgrad

        self.ce=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()

        self.lamb=lamb          # Grid search = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2.5, 4]; chosen was 0.75
        self.smax=smax          # Grid search = [25, 50, 100, 200, 400, 800]; chosen was 400
        if len(args.parameter)>=1:
            params=args.parameter.split(',')
            print('Setting parameters to',params)
            self.lamb=float(params[0])
            self.smax=float(params[1])

        self.mask_pre=None
        self.mask_back=None

        self.device=device

        return

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        return torch.optim.SGD(self.model.parameters(),lr=lr)

    def train(self,t,train_loader,val_loader):
        best_loss=np.inf
        self.optimizer=self._get_optimizer(self.lr)

        # Loop epochs
        try:
            for e in range(self.nepochs):
                # Train
                clock0=time.time()
                self.train_epoch(t,train_loader)
                clock1=time.time()
                train_loss,train_acc=self.eval(t,train_loader)
                clock2=time.time()
                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                    1000*(clock1-clock0),1000*(clock2-clock1),train_loss,100*train_acc),end='')
                # Valid
                valid_loss,valid_acc=self.eval(t,val_loader)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
                
                print()
        except KeyboardInterrupt:
            print()

        # Activations mask
        task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=False)
        mask=self.model.mask(task,s=self.smax)
        for i in range(len(mask)):
            mask[i]=torch.autograd.Variable(mask[i].data.clone(),requires_grad=False)
        if t==0:
            self.mask_pre=mask
        else:
            for i in range(len(self.mask_pre)):
                self.mask_pre[i]=torch.max(self.mask_pre[i],mask[i])

        # Weights mask
        self.mask_back={}
        for n,_ in self.model.named_parameters():
            vals=self.model.get_view_for(n,self.mask_pre)
            if vals is not None:
                self.mask_back[n]=1-vals

        return

    def train_epoch(self,t,loader,thres_cosh=50,thres_emb=6):
        self.model.train()

        r=len(loader.dataset)
        i = 0
        for images, targets in loader:
            images = images.to(self.device)
            images.requires_grad=True
            targets = targets.long().to(self.device)
            #targets.requires_grad=True
            task=torch.LongTensor([t]).to(self.device)
            s=(self.smax-1/self.smax)*i/r+1/self.smax

            outputs,masks=self.model(task,images,s=s)
            output=outputs[t]
            loss,_=self.criterion(output,targets,masks)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Restrict layer gradients in backprop
            if t>0:
                for n,p in self.model.named_parameters():
                    if n in self.mask_back:
                        p.grad.data*=self.mask_back[n]

            # Compensate embedding gradients
            for n,p in self.model.named_parameters():
                if n.startswith('e'):
                    num=torch.cosh(torch.clamp(s*p.data,-thres_cosh,thres_cosh))+1
                    den=torch.cosh(p.data)+1
                    p.grad.data*=self.smax/s*num/den

            # Apply step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.clipgrad)
            self.optimizer.step()

            # Constrain embeddings
            for n,p in self.model.named_parameters():
                if n.startswith('e'):
                    p.data=torch.clamp(p.data,-thres_emb,thres_emb)

            i += images.size(0)

        return

    def eval(self,t,loader):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        total_reg=0

        r=len(loader.dataset)

        for images, targets in loader:
            images = images.to(self.device)
            targets = targets.long().to(self.device)
            task=torch.LongTensor([t]).to(self.device)

            # Forward
            outputs,masks=self.model(task,images,s=self.smax)
            output=outputs[t]
            loss,reg=self.criterion(output,targets,masks)
            _,pred=output.max(1)
            hits=(pred==targets).float()

            # Log
            b = images.size(0)
            total_loss+=loss.data.cpu().numpy().item()*b
            total_acc+=hits.sum().data.cpu().numpy().item()
            total_num+=b
            total_reg+=reg.data.cpu().numpy().item()*b

        print('  {:.3f}  '.format(total_reg/total_num),end='')

        return total_loss/total_num,total_acc/total_num

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

########################################################################################################################
