#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.autograd import grad

from learn2learn.algorithms.base_learner import BaseLearner
from learn2learn.utils import clone_module

from model.models import TaskNormalization


def maml_update(model, lr, grads=None):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/maml.py)
    **Description**
    Performs a MAML update on model using grads and lr.
    The function re-routes the Python object, thus avoiding in-place
    operations.
    NOTE: The model itself is updated in-place (no deepcopy), but the
          parameters' tensors are not.
    **Arguments**
    * **model** (Module) - The model to update.
    * **lr** (float) - The learning rate used to update the model.
    * **grads** (list, *optional*, default=None) - A list of gradients for each parameter
        of the model. If None, will use the gradients in .grad attributes.
    **Example**
    ~~~python
    maml = l2l.algorithms.MAML(Model(), lr=0.1)
    model = maml.clone() # The next two lines essentially implement model.adapt(loss)
    grads = autograd.grad(loss, model.parameters(), create_graph=True)
    maml_update(model, lr=0.1, grads)
    ~~~
    """

    if grads is not None:
        params = list(model.parameters())
        if not len(grads) == len(list(params)):
            msg = 'WARNING:maml_update(): Parameters and gradients have different length. ('
            msg += str(len(params)) + ' vs ' + str(len(grads)) + ')'
            print(msg)
        for p, g in zip(params, grads):
            if p.grad is not None:
                p.grad = g

    # Update the params
    for param_key in model._parameters:
        p = model._parameters[param_key]
        if p is not None and p.grad is not None:
            model._parameters[param_key] = p - lr * p.grad

    # Second, handle the buffers if necessary
    for buffer_key in model._buffers:
        buff = model._buffers[buffer_key]
        if buff is not None and buff.grad is not None:
            model._buffers[buffer_key] = buff - lr * buff.grad

    # Then, recurse for each submodule
    for module_key in model._modules:
        model._modules[module_key] = maml_update(model._modules[module_key],
                                                 lr=lr,
                                                 grads=None)
    return model


class MAML(BaseLearner):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/maml.py)
    **Description**
    High-level implementation of *Model-Agnostic Meta-Learning*.
    This class wraps an arbitrary nn.Module and augments it with `clone()` and `adapt`
    methods.
    For the first-order version of MAML (i.e. FOMAML), set the `first_order` flag to `True`
    upon initialization.
    **Arguments**
    * **model** (Module) - Module to be wrapped.
    * **lr** (float) - Fast adaptation learning rate.
    * **first_order** (bool, *optional*, default=False) - Whether to use the first-order
        approximation of MAML. (FOMAML)
    * **allow_unused** (bool, *optional*, default=False) - Whether to allow differentiation
        of unused parameters.
    **References**
    1. Finn et al. 2017. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks."
    **Example**
    ~~~python
    linear = l2l.algorithms.MAML(nn.Linear(20, 10), lr=0.01)
    clone = linear.clone()
    error = loss(clone(X), y)
    clone.adapt(error)
    error = loss(clone(X), y)
    error.backward()
    ~~~
    """

    def __init__(self, model, lr,first_order=False, allow_unused=False):
        super(MAML, self).__init__()
        self.module = model
        self.lr = lr
        self.first_order = first_order
        self.allow_unused = allow_unused

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def adapt(self, loss, first_order=None, allow_unused=None):
        """
        **Description**
        Updates the clone parameters in place using the MAML update.
        **Arguments**
        * **loss** (Tensor) - Loss to minimize upon update.
        * **first_order** (bool, *optional*, default=None) - Whether to use first- or
            second-order updates. Defaults to self.first_order.
        * **allow_unused** (bool, *optional*, default=None) - Whether to allow differentiation
        of unused parameters. Defaults to self.allow_unused.
        """
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        second_order = not first_order
        params = []
        self.parametersMAML(self.module, params)
        gradients = grad(loss,
                         #self.module.parameters(),
                         #self.getParams(),
                         params,
                         retain_graph=second_order,
                         create_graph=second_order,
                         allow_unused=allow_unused)

        grads = []
        self.completeGradient(self.module, gradients, grads)
        self.module = maml_update(self.module, self.lr, grads)

    def clone(self, first_order=None, allow_unused=None):
        """
        **Description**
        Returns a `MAML`-wrapped copy of the module whose parameters and buffers
        are `torch.clone`d from the original module.
        This implies that back-propagating losses on the cloned module will
        populate the buffers of the original module.
        For more information, refer to learn2learn.clone_module().
        **Arguments**
        * **first_order** (bool, *optional*, default=None) - Whether the clone uses first-
            or second-order updates. Defaults to self.first_order.
        * **allow_unused** (bool, *optional*, default=None) - Whether to allow differentiation
        of unused parameters. Defaults to self.allow_unused.
        """
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        return MAML(clone_module(self.module),
                    lr=self.lr,
                    first_order=first_order,
                    allow_unused=allow_unused)

    def setLinearLayer(self, task):
        self.module.setLinearLayer(task)

    def setTaskNormalizationLayer(self, task):
        self.module.setTaskNormalizationLayer(task)

    def parametersMAML(self, network, all_layers):
        for layer in network.children():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.Linear):
                for p in layer.parameters():
                    all_layers.append(p)
            if list(layer.children()) != [] and not isinstance(layer, TaskNormalization): # if leaf node, add it to list
                self.parametersMAML(layer, all_layers)

    def completeGradient(self, network, gradients, grads=[], count=0):
        for layer in network.children():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.Linear):
                for _ in layer.parameters():
                    grads.append(gradients[count])
                    count += 1
            
            if isinstance(layer, TaskNormalization):
                for p in layer.parameters():
                    p.grad = torch.zeros_like(p)
                    grads.append(torch.zeros_like(p))

            if list(layer.children()) != [] and not isinstance(layer, TaskNormalization):
                self.completeGradient(layer, gradients, grads, count)

    # def getParams(self):
    #     if len(self.freeze_layer) == 0:
    #         return self.module.parameters()
    #     else:
    #         params = []
    #         for name, param in self.module.named_parameters():
    #             if name[5] == 'r' or int(name[5]) not in self.freeze_layer:
    #                 params.append(param)

    #         return params

    # def completeGradient(self, gradients):
    #     if len(self.freeze_layer) == 0:
    #         return gradients
    #     else:
    #         grads = []
    #         i = 0
    #         for name, param in self.module.named_parameters():
    #             if name[5] == 'r' or int(name[5]) not in self.freeze_layer:
    #                 grads.append(gradients[i])
    #                 i += 1
    #             else:
    #                 grads.append(torch.zeros_like(param))


    #         return grads