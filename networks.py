import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from mylib.sim import default, advection, smagorinsky
from mylib.force import maulik_force as force

from collections import OrderedDict


class BurgersParamPDE(nn.Module):
    """
    Modèle Physique
    """

    def __init__(self, mode, params):
        super().__init__()
        self.params = params
        self.dx, self.nu, self.dt, self.N, self.L = params.get(['DX', 'nu', 'DT', 'N', 'L'])
        self.mode = mode

        if self.mode == 'default':
            self.func_forward = default
            self.param_dict = {'nu': self.nu, 'dx': self.dx}
        elif self.mode == 'advection':
            self.func_forward = advection
            self.param_dict = {'dx': self.dx}
        elif self.mode == 'smagorinsky':
            self.Cs = params.get('Cs')
            self.func_forward = smagorinsky
            self.param_dict = {'nu': self.nu, 'dx': self.dx, 'dt' :self.dt, 'Cs': self.Cs}
        elif self.mode == 'none':
            self.func_forward = lambda : 0
            self.param_dict = {}
        else:
            raise ValueError(f'mode {self.mode} not supported')
        
    
    def forward(self, state):
        return self.func_forward(state, **self.param_dict)




class SimpleConv(nn.Module):
    def __init__(self, channel=[8,16,32], kernel=3, activation="relu"):
        super().__init__()
        dictionnary = OrderedDict()
        if activation == 'relu':
            activation = nn.ReLU
        elif activation == 'elu':
            activation = nn.ELU
        else:
            raise ValueError
        

        self.channel = [1] + channel + [1]
        for i in range(len(self.channel) - 1):
            dictionnary[f"conv_{i}"] = nn.Conv1d(self.channel[i], self.channel[i+1], kernel, padding=kernel//2, padding_mode='circular', dtype=torch.double)
            if i < len(self.channel) - 2:
                dictionnary[f"relu_{i}"] = activation()
        
        self.kernel = kernel
        self.channel = channel

        self.net = nn.Sequential(dictionnary)

    def forward(self, x):

        x = x.permute(1, 0) # (space, batch) -> (batch, space)
        x = x.view(x.shape[0], 1, x.shape[1]) # (batch, space) -> (batch, 1, space) (1 is for channel)

        x = self.net(x)

        x = x.view(x.shape[0], x.shape[2]) # (batch, 1, space) -> (batch, space)
        x = x.permute(1, 0) # (batch, space) -> (space, batch)

        return x

        