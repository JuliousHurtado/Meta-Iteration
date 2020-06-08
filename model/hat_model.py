import sys
import torch
import numpy as np

import utils

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class HATModel(torch.nn.Module):

    def __init__(self,taskcla, device, in_channels=3):
        super(HATModel,self).__init__()

        ncha,size,_= [in_channels,32,32]
        self.taskcla=taskcla
        self.device=device

        if in_channels == 3:
            lin_size_factor = 4
        else:
            lin_size_factor = 1

        self.c1=torch.nn.Conv2d(ncha, 32, kernel_size=3, stride=(1,1), padding=1,bias=True)
        s=compute_conv_output_size(size,3)
        s=s//2
        self.c2=torch.nn.Conv2d(32,32,kernel_size=3, stride=(1,1), padding=1,bias=True)
        s=compute_conv_output_size(s,3)
        s=s//2
        self.c3=torch.nn.Conv2d(32,32,kernel_size=3, stride=(1,1), padding=1,bias=True)
        s=compute_conv_output_size(s,3)
        s=s//2
        self.c4=torch.nn.Conv2d(32,32,kernel_size=3, stride=(1,1), padding=1,bias=True)
        s=compute_conv_output_size(s,3)
        s=s//2
        self.smid=s
        self.maxpool=torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), ceil_mode=False)
        self.relu=torch.nn.ReLU()

        self.bn1 = torch.nn.BatchNorm2d(32, affine=True, track_running_stats=False)
        self.bn2 = torch.nn.BatchNorm2d(32, affine=True, track_running_stats=False)
        self.bn3 = torch.nn.BatchNorm2d(32, affine=True, track_running_stats=False)
        self.bn4 = torch.nn.BatchNorm2d(32, affine=True, track_running_stats=False)

        self.last=torch.nn.ModuleList()
        for n in self.taskcla:
            self.last.append(torch.nn.Linear(lin_size_factor*32,n))

        self.gate=torch.nn.Sigmoid()
        # All embedding stuff should start with 'e'
        self.ec1=torch.nn.Embedding(len(self.taskcla),32)
        self.ec2=torch.nn.Embedding(len(self.taskcla),32)
        self.ec3=torch.nn.Embedding(len(self.taskcla),32)
        self.ec4=torch.nn.Embedding(len(self.taskcla),32)

        return

    def forward(self,t,x,s=1):
        # Gates
        masks=self.mask(t,s=s)
        gc1,gc2,gc3,gc4=masks
        # Gated
        h=self.maxpool(self.bn1(self.relu(self.c1(x))))
        h=h*gc1.view(1,-1,1,1).expand_as(h)
        #print(h.size())
        h=self.maxpool(self.bn2(self.relu(self.c2(h))))
        h=h*gc2.view(1,-1,1,1).expand_as(h)
        h=self.maxpool(self.bn3(self.relu(self.c3(h))))
        h=h*gc3.view(1,-1,1,1).expand_as(h)
        #print(h.size())
        h=self.maxpool(self.bn4(self.relu(self.c4(h))))
        h=h*gc4.view(1,-1,1,1).expand_as(h)
        h=h.view(x.size(0),-1)
        
        h = self.last[t](h)
        return h, masks

    def mask(self,t,s=1):
        t=torch.LongTensor([t]).to(self.device)
        gc1=self.gate(s*self.ec1(t))
        gc2=self.gate(s*self.ec2(t))
        gc3=self.gate(s*self.ec3(t))
        gc4=self.gate(s*self.ec4(t))
        return [gc1,gc2,gc3,gc4]

    def get_view_for(self,n,masks):
        gc1,gc2,gc3,gc4=masks
        if n=='c1.weight':
            return gc1.data.view(-1,1,1,1).expand_as(self.c1.weight)
        elif n=='c1.bias':
            return gc1.data.view(-1)
        elif n=='c2.weight':
            post=gc2.data.view(-1,1,1,1).expand_as(self.c2.weight)
            pre=gc1.data.view(1,-1,1,1).expand_as(self.c2.weight)
            return torch.min(post,pre)
        elif n=='c2.bias':
            return gc2.data.view(-1)
        elif n=='c3.weight':
            post=gc3.data.view(-1,1,1,1).expand_as(self.c3.weight)
            pre=gc2.data.view(1,-1,1,1).expand_as(self.c3.weight)
            return torch.min(post,pre)
        elif n=='c3.bias':
            return gc3.data.view(-1)
        elif n=='c4.weight':
            post=gc4.data.view(-1,1,1,1).expand_as(self.c4.weight)
            pre=gc3.data.view(1,-1,1,1).expand_as(self.c4.weight)
            return torch.min(post,pre)
        elif n=='c4.bias':
            return gc4.data.view(-1)
        return None




class HATOriginal(torch.nn.Module):

    def __init__(self,inputsize,taskcla):
        super(Net,self).__init__()

        ncha,size,_= [in_channels,32,32]
        self.taskcla=taskcla
        self.device=device

        self.c1=torch.nn.Conv2d(ncha,64,kernel_size=size//8)
        s=utils.compute_conv_output_size(size,size//8)
        s=s//2
        self.c2=torch.nn.Conv2d(64,128,kernel_size=size//10)
        s=utils.compute_conv_output_size(s,size//10)
        s=s//2
        self.c3=torch.nn.Conv2d(128,256,kernel_size=2)
        s=utils.compute_conv_output_size(s,2)
        s=s//2
        self.smid=s
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()

        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)
        self.fc1=torch.nn.Linear(256*self.smid*self.smid,2048)
        self.fc2=torch.nn.Linear(2048,2048)
        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(2048,n))

        self.gate=torch.nn.Sigmoid()
        # All embedding stuff should start with 'e'
        self.ec1=torch.nn.Embedding(len(self.taskcla),64)
        self.ec2=torch.nn.Embedding(len(self.taskcla),128)
        self.ec3=torch.nn.Embedding(len(self.taskcla),256)
        self.efc1=torch.nn.Embedding(len(self.taskcla),2048)
        self.efc2=torch.nn.Embedding(len(self.taskcla),2048)
        """ (e.g., used in the compression experiments)
        lo,hi=0,2
        self.ec1.weight.data.uniform_(lo,hi)
        self.ec2.weight.data.uniform_(lo,hi)
        self.ec3.weight.data.uniform_(lo,hi)
        self.efc1.weight.data.uniform_(lo,hi)
        self.efc2.weight.data.uniform_(lo,hi)
        #"""

        return

    def forward(self,t,x,s=1):
        # Gates
        masks=self.mask(t,s=s)
        gc1,gc2,gc3,gfc1,gfc2=masks
        # Gated
        h=self.maxpool(self.drop1(self.relu(self.c1(x))))
        h=h*gc1.view(1,-1,1,1).expand_as(h)
        h=self.maxpool(self.drop1(self.relu(self.c2(h))))
        h=h*gc2.view(1,-1,1,1).expand_as(h)
        h=self.maxpool(self.drop2(self.relu(self.c3(h))))
        h=h*gc3.view(1,-1,1,1).expand_as(h)
        h=h.view(x.size(0),-1)
        h=self.drop2(self.relu(self.fc1(h)))
        h=h*gfc1.expand_as(h)
        h=self.drop2(self.relu(self.fc2(h)))
        h=h*gfc2.expand_as(h)
        y=[]
        for i,_ in self.taskcla:
            y.append(self.last[i](h))
        return y,masks

    def mask(self,t,s=1):
        t=torch.LongTensor([t]).to(self.device)
        gc1=self.gate(s*self.ec1(t))
        gc2=self.gate(s*self.ec2(t))
        gc3=self.gate(s*self.ec3(t))
        gfc1=self.gate(s*self.efc1(t))
        gfc2=self.gate(s*self.efc2(t))
        return [gc1,gc2,gc3,gfc1,gfc2]

    def get_view_for(self,n,masks):
        gc1,gc2,gc3,gfc1,gfc2=masks
        if n=='fc1.weight':
            post=gfc1.data.view(-1,1).expand_as(self.fc1.weight)
            pre=gc3.data.view(-1,1,1).expand((self.ec3.weight.size(1),self.smid,self.smid)).contiguous().view(1,-1).expand_as(self.fc1.weight)
            return torch.min(post,pre)
        elif n=='fc1.bias':
            return gfc1.data.view(-1)
        elif n=='fc2.weight':
            post=gfc2.data.view(-1,1).expand_as(self.fc2.weight)
            pre=gfc1.data.view(1,-1).expand_as(self.fc2.weight)
            return torch.min(post,pre)
        elif n=='fc2.bias':
            return gfc2.data.view(-1)
        elif n=='c1.weight':
            return gc1.data.view(-1,1,1,1).expand_as(self.c1.weight)
        elif n=='c1.bias':
            return gc1.data.view(-1)
        elif n=='c2.weight':
            post=gc2.data.view(-1,1,1,1).expand_as(self.c2.weight)
            pre=gc1.data.view(1,-1,1,1).expand_as(self.c2.weight)
            return torch.min(post,pre)
        elif n=='c2.bias':
            return gc2.data.view(-1)
        elif n=='c3.weight':
            post=gc3.data.view(-1,1,1,1).expand_as(self.c3.weight)
            pre=gc2.data.view(1,-1,1,1).expand_as(self.c3.weight)
            return torch.min(post,pre)
        elif n=='c3.bias':
            return gc3.data.view(-1)
        return None