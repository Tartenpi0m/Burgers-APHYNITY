from .burgers import Burgers

from numpy import pi
import torch
import math
from torch.utils.data import DataLoader

from mylib.parameters import Params
    

def param_burgers(buffer_filepath, size: tuple, DNS_params, batch_size=24, method='rk4', cond_option='dissipation'):

    dataset_train = Burgers(size[0], DNS_params, buffer_filepath, method=method, cond_option=cond_option)
    dataset_test  = Burgers(size[1], DNS_params, buffer_filepath, method=method, cond_option=cond_option, shift=size[0])

    dataloader_train_params = {
        'dataset'    : dataset_train,
        'batch_size' : batch_size,
        'num_workers': 0,
        'pin_memory' : True,
        'drop_last'  : False,
        'shuffle'    : True,
    }

    dataloader_test_params = {
        'dataset'    : dataset_test,
        'batch_size' : batch_size,
        'num_workers': 0,
        'pin_memory' : True,
        'drop_last'  : False,
        'shuffle'    : False,
    }

    dataloader_train = DataLoader(**dataloader_train_params)
    dataloader_test  = DataLoader(**dataloader_test_params)

    return dataloader_train, dataloader_test

# def init_dataloaders(DNS_params, buffer_filepath=None):
#     assert buffer_filepath is not None
#     return param_burgers(buffer_filepath, DNS_params)