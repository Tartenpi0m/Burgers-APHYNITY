import torch.nn as nn
from torchdiffeq import odeint_adjoint, odeint
from torchdiffeq2 import odeint as odeint2

from networks import *

from mylib.force import force as force_object
from mylib.utils import filter_flow

class DerivativeEstimator(nn.Module):
    def __init__(self, model_phy, model_aug, is_augmented):
        super().__init__()
        self.model_phy = model_phy
        self.model_aug = model_aug
        self.is_augmented = is_augmented

    def forward(self, t, state):
        res_phy = self.model_phy(state)
        if self.is_augmented:
            if isinstance(self.model_aug, NuConv):
                nu = torch.ones(1, dtype=torch.double) * self.model_phy.params('nu')
                res_aug = self.model_aug((state, nu))
            else:
                res_aug = self.model_aug(state)
            return res_phy + res_aug
        else:
            return res_phy

class Forecaster(nn.Module):
    def __init__(self, model_phy, model_aug, is_augmented, method='rk4', options=None, cond_option=None):
        super().__init__()

        self.model_phy = model_phy
        self.model_aug = model_aug

        self.derivative_estimator = DerivativeEstimator(self.model_phy, self.model_aug, is_augmented=is_augmented)
        self.method = method
        self.options = options

        self.cond_option = cond_option
        if cond_option == 'dissipation':
            self.force = False
            self.int_ = odeint 
        elif cond_option == 'forced':
            self.force = force_object(self.model_phy.params)
            self.int_ = odeint2
        else:
            raise ValueError('Unknown cond_option: ', cond_option)

        
    def forward(self, y0, t, forcing_tensor=None):
        """
        param y0: (space, batch)
        param t: (time nt)
        param forcing_tensor: (batch, space, n_force)
        """
        if not self.force:
            res = self.int_(self.derivative_estimator, y0=y0, t=t, method=self.method, options=self.options)
        else:
            res = self.int_(self.derivative_estimator, y0=y0, t=t, method=self.method, options=self.options, force=(self.force, forcing_tensor))
        # res: (time, space, batch)
        dim_seq = y0.dim() + 1
        dims = [2, 1, 0] + list(range(dim_seq))[3:]
        return res.permute(*dims)   # (batch, space, time)
    
    def get_pde_params(self):
        return self.model_phy.params
    
    
    def forward_ram_saver(self, y0, params, forcing_tensor=None, split_time_period=1):

        assert(params('t') >= 1)
        #assert((split_time_period % 0.0001) == 0)
        
        nt_split = torch.tensor_split(params('nt'), params('t')//split_time_period)
        if forcing_tensor is not None:
            forcing_tensor_split = torch.tensor_split(forcing_tensor, params('t')//split_time_period, dim=2)
        else:
            forcing_tensor_split = [None] * params('t')

        flow_list = []
        for i, (nt, forcing_tensor) in enumerate(zip(nt_split, forcing_tensor_split)):
            
            if i < len(nt_split) - 1:
                nt = torch.concat([nt, nt_split[i+1][:1]])
                if forcing_tensor is not None:
                    forcing_tensor = torch.concat([forcing_tensor, forcing_tensor_split[i+1][:,:,:1]], dim=2)

            
            flow = self.forward(y0, nt, forcing_tensor=forcing_tensor)
            y0 = flow[:, :, -1].view(-1,1)
            
            if i < len(nt_split) - 1:
                flow = filter_flow(flow[:,:,:-1], params('spatial_filt'), params('temporal_filt'))
            else:
                flow = filter_flow(flow, params('spatial_filt'), params('temporal_filt'))
            flow_list.append(flow)

        return torch.cat(flow_list, dim=2)
