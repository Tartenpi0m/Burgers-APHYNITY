import numpy as np
import os
import math, shelve
from torch.utils.data import Dataset
from scipy.integrate import solve_ivp
import torch
import torchdiffeq
from collections import OrderedDict

from mylib.parameters import Params
from mylib.force import generate_force_tensor, maulik_force
from forecasters import DerivativeEstimator, Forecaster
from networks import BurgersParamPDE
from mylib.utils import filter_flow, load_set, save_set

class Burgers(Dataset):


    def __init__(self, size, params: Params, path, method='rk4', options=None, cond_option='dissipation', shift=0):
        super().__init__()
        self.params = params
        self.forecaster = Forecaster(BurgersParamPDE('default', params), None, is_augmented=False, cond_option=cond_option,  method=method, options=None)
        self.size = size
        self.cond_option = cond_option
        self.shift = shift

        self.path = path
        self.data_path = os.path.join(path, 'data')
        if not os.path.isdir(self.data_path):
            os.mkdir(self.data_path)
            
        if  not os.path.isfile(os.path.join(self.path, 'set')):
            self.set = set()
            save_set(self.set, os.path.join(self.path, 'set'))
        else:
            self.set = load_set(os.path.join(self.path, 'set'))
    
    def _get_initial_condition(self, seed):
        return maulik_force(self.params, random_seed=seed).view(-1,1)

    
    def __getitem__(self, index):
        with torch.no_grad():
        
            seed = index + self.shift

            self.set = load_set(os.path.join(self.path, 'set'))
            if self.cond_option == 'dissipation':
                forcing_tensor = torch.zeros(1) #fake tensor for compatibility only
                initial_state = self._get_initial_condition(seed) # (space, 1)
            elif self.cond_option == 'forced':
                forcing_tensor = generate_force_tensor(self.params, random_seed=seed) # (space, n)
                initial_state = self._get_initial_condition(seed) * self.params('force_initial_strength') # (space, 1)

            if seed in self.set:
                #print("Loading data for index: ", seed)
                state_history = torch.load(os.path.join(self.data_path, str(seed)))
                if self.cond_option == 'forced':
                    forcing_tensor = filter_flow(forcing_tensor.unsqueeze(0), self.params('spatial_filt'), 1).squeeze(0) # (space, n)
            else:
                print("Generating data for index: ", seed)

                state_history = self.forecaster.forward(initial_state, self.params.get('NT'), forcing_tensor.unsqueeze(0)) # (batch, space, time)
                state_history = filter_flow(state_history, self.params('spatial_filt'), self.params('temporal_filt')).squeeze(0) # (batch, space, time) -> (space, time)
                if self.cond_option == 'forced':
                    forcing_tensor = filter_flow(forcing_tensor.unsqueeze(0), self.params('spatial_filt'), 1).squeeze(0) # (space, n)
                
                torch.save(state_history, os.path.join(self.data_path, str(seed)))
                self.set.add(seed)
                save_set(self.set, os.path.join(self.path, 'set'))

            return state_history, forcing_tensor  # (space, time), (space, n) or False
    
    def __len__(self):
        return self.size