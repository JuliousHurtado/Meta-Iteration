import torch
from torch import nn
import copy

def maml_init_(module):
    nn.init.xavier_uniform_(module.weight.data, gain=1.0)
    nn.init.constant_(module.bias.data, 0.0)
    return module

class TaskNormalization(nn.Module):
    def __init__(self, input_dim, output_dim, r = 4):
        super(TaskNormalization, self).__init__()

        self.r = r
        self.scale = int(input_dim/self.r)

        self.fc1 = nn.Linear(input_dim, self.scale)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.scale, output_dim)
        self.sigmoid = nn.Sigmoid()

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        data_clone = x.clone()

        out = self.preProcess(x)
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))     

        return data_clone.mul_(out.unsqueeze(2).unsqueeze(3))

    def preProcess(self, x):
        size = x.size()
        return torch.mean(x.view(size[0],size[1],size[2]*size[3]), dim=2)

class ConvBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 max_pool=True,
                 max_pool_factor=1.0,
                 task_normalization=False):
        super(ConvBlock, self).__init__()
        stride = (int(2 * max_pool_factor), int(2 * max_pool_factor))
        if max_pool:
            self.max_pool = nn.MaxPool2d(kernel_size=stride,
                                         stride=stride,
                                         ceil_mode=False,
                                         )
            stride = (1, 1)
        else:
            self.max_pool = lambda x: x
        self.normalize = nn.BatchNorm2d(out_channels,
                                        affine=True,
                                        # eps=1e-3,
                                        # momentum=0.999,
                                        track_running_stats=False,
                                        )
        #nn.init.uniform_(self.normalize.weight)
        self.relu = nn.ReLU()
        if task_normalization:
            self.taskNormalization = TaskNormalization(out_channels, out_channels)
        else:
            self.taskNormalization = nn.Identity()

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=1,
                              bias=True)
        maml_init_(self.conv)

    def forward(self, x):
        x = self.conv(x)
        x = self.normalize(x)
        x = self.relu(x)
        x = self.taskNormalization(x)
        x = self.max_pool(x)

        return x

class ConvBase(nn.Sequential):
    def __init__(self,
                 output_size,
                 hidden=64,
                 channels=1,
                 max_pool=False,
                 layers=4,
                 max_pool_factor=1.0,
                 task_normalization=False):
        core = [ConvBlock(channels,
                          hidden,
                          (3, 3),
                          max_pool=max_pool,
                          max_pool_factor=max_pool_factor,
                          task_normalization=task_normalization),
                ]
        for l in range(layers - 1):
            core.append(ConvBlock(hidden,
                                  hidden,
                                  kernel_size=(3, 3),
                                  max_pool=max_pool,
                                  max_pool_factor=max_pool_factor,
                                  task_normalization=task_normalization))
        super(ConvBase, self).__init__(*core)

class MiniImagenetCNN(nn.Module):
    def __init__(self, output_size, hidden_size=32, layers=4, task_normalization=False):
        super(MiniImagenetCNN, self).__init__()
        self.base = ConvBase(output_size=hidden_size,
                             hidden=hidden_size,
                             channels=3,
                             max_pool=True,
                             layers=layers,
                             max_pool_factor=4 // layers,
                             task_normalization=task_normalization)
        self.linear = nn.Linear(25 * hidden_size, output_size, bias=True)
        maml_init_(self.linear)
        self.hidden_size = hidden_size
        self.layers = layers

    def forward(self, x):
        x = self.base(x)
        x = self.linear(x.view(x.size(0), 4 * self.hidden_size))
        return x

    def setTaskNormalizationLayer(self, layers):
        if len(layers) != self.layers:
            print("There are not the same amount of task layers and number of layers")

        for i,l in enumerate(self.base):
            l.taskNormalization = layers[i]

    def setLinearLayer(self, layer):
        self.linear = layer


class TaskManager(nn.Module):
    def __init__(self, outputs_size, ways=5, hidden_size=32, layers=4, task_normalization=False, device = 'cpu'):
        super(TaskManager, self).__init__()
        """
        outputs_size -> Is a list of the amount of classes for the different tasks, 
                the first element is the element of the ways for the meta learning process 

        Task 0 is meta learning
        """
        self.task = {}
        self.task_normalization = task_normalization
        for i,out in enumerate(outputs_size):
            norm_layers = []
            if task_normalization:
                for _ in range(layers): 
                    norm_layers.append(TaskNormalization(hidden_size, hidden_size).to(device))
            linear_layer = nn.Linear(4 * hidden_size, out).to(device)

            self.task[i] = { 'norm': norm_layers, 'linear': linear_layer }

        self.task['meta'] = {'linear': nn.Linear(4 * hidden_size, ways).to(device)}
        self.task['meta']['norm'] = [ TaskNormalization(hidden_size, hidden_size).to(device) for _ in range(layer) ]
        self.model = MiniImagenetCNN(ways, hidden_size=hidden_size, layers=layers, task_normalization=task_normalization)

        self.setLinearLayer('meta')
        self.setTaskNormalizationLayer('meta')

    def setLinearLayer(self, task):
        self.model.setLinearLayer(self.task[task]['linear'])

    def setTaskNormalizationLayer(self, task):
        if self.task_normalization:
            self.model.setTaskNormalizationLayer(self.task[task]['norm'])

    def forward(self, x):
        return self.model(x)

    def setMasks(self, masks):
        useful.append(((m - 1) == -1).type(torch.FloatTensor).to(device))
        for j, elem in enumerate(self.model.base):
            m = ((masks[j] - 1) == -1).type(torch.FloatTensor).to(masks[j].device)
            elem.conv.weight.grad.mul_(m.view(-1,1,1,1))
            elem.conv.bias.grad.mul_(m)

            elem.normalize.weight.grad.mul_(m)
            elem.normalize.bias.grad.mul_(m)
