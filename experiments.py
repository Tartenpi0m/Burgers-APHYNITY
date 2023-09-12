import numpy as np
import torch
import torch.nn as nn
import statistics

from utils import fix_seed, make_basedir, convert_tensor
from utils import Logger, SparsityScheduler, remove_forced_step_from_learning

import warnings

import gc

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
matplotlib.rcParams["figure.dpi"] = 100
matplotlib.rcParams['interactive'] = False

_EPSILON = 1e-5

import os, sys
from tqdm import tqdm

from test_aphynity import graphics2
import collections

from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader


class BaseExperiment(object):
    def __init__(self, device, path='./exp', seed=None):
        self.device = device
        self.path = make_basedir(path)
        os.makedirs(os.path.join(self.path, 'graphics'), exist_ok=True)

            
        if seed is not None:
            fix_seed(seed)

    def training(self, mode=True):
        for m in self.modules():
            m.train(mode)

    def evaluating(self):
        self.training(mode=False)

    def zero_grad(self):
        for optimizer in self.optimizers():
            optimizer.zero_grad()        

    def to(self, device):
        for m in self.modules():
            m.to(device)
        return self

    def modules(self):
        for name, module in self.named_modules():
            yield module

    def named_modules(self):
        for name, module in self._modules.items():
            yield name, module

    def datasets(self):
        for name, dataset in self.named_datasets():
            yield dataset

    def named_datasets(self):
        for name, dataset in self._datasets.items():
            yield name, dataset

    def optimizers(self):
        for name, optimizer in self.named_optimizers():
            yield optimizer

    def named_optimizers(self):
        for name, optimizer in self._optimizers.items():
            yield name, optimizer

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            if not hasattr(self, '_modules'):
                self._modules = collections.OrderedDict()
            self._modules[name] = value
        elif isinstance(value, DataLoader):
            if not hasattr(self, '_datasets'):
                self._datasets = collections.OrderedDict()
            self._datasets[name] = value
        elif isinstance(value, Optimizer):
            if not hasattr(self, '_optimizers'):
                self._optimizers = collections.OrderedDict()
            self._optimizers[name] = value
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        if '_datasets' in self.__dict__:
            datasets = self.__dict__['_datasets']
            if name in datasets:
                return datasets[name]
        if '_optimizers' in self.__dict__:
            optimizers = self.__dict__['_optimizers']
            if name in optimizers:
                return optimizers[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __delattr__(self, name):
        if name in self._modules:
            del self._modules[name]
        elif name in self._datasets:
            del self._datasets[name]
        elif name in self._optimizers:
            del self._optimizers[name]
        else:
            object.__delattr__(self, name)

def show(img):
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

class LoopExperiment(BaseExperiment):
    def __init__(
        self, train, test=None, root=None, nepoch=10, patience=5, **kwargs):
        super().__init__(**kwargs)
        self.train = train
        self.test = test
        self.nepoch = nepoch
        self.patience = patience
        self.logger = Logger(log_path=os.path.join(self.path, 'log.txt'), tensorboard_path=os.path.join(self.path, 'tensorboard'))
        print(' '.join(sys.argv))

    def train_step(self, backward, val=False):
        self.training()
        loss, output, batch = self.step(True, backward)

        return loss, output, batch

    def val_step(self):
        self.evaluating()
        with torch.no_grad():
            loss, output, batch = self.step(False, backward=False)

        return loss, output, batch

    def step(self, **kwargs):
        raise NotImplementedError


class SpectreLoss(nn.Module):
    def __init__(self, N=128):
        super().__init__()
        self.loss = nn.MSELoss()
        self.N = N
    
    def forward(self, pred, target):
        pred_spectre = torch.abs(torch.fft.fft(pred, dim=1)[:, :self.N,:])
        target_spectre = torch.abs(torch.fft.fft(target, dim=1)[:, :self.N,:])
        return self.loss(pred_spectre, target_spectre)

class APHYNITYExperiment(LoopExperiment):
    def __init__(self, nets, optimizer, niter=1, scheduler=None, traj_loss=True, spectre_loss=False, sparse=0, **kwargs):
        super().__init__(**kwargs)

        self.niter = niter
        self.nets = nets
        self.traj_loss = nn.MSELoss()
        self.spectre_loss = SpectreLoss(self.nets[0].model_phy.params('N'))
        self.backward_traj_loss = True if traj_loss else False
        self.backward_spectre_loss = True if spectre_loss else False
        for i in range(len(self.nets)):
            self.nets[i].to(self.device)

        self.optimizer = optimizer

        self.scheduler = None
        if scheduler['name'] == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=scheduler['plateau_factor'], patience=scheduler['plateau_patience'], threshold=scheduler['plateau_min_delta'], min_lr=scheduler['plateau_min_lr'])
            #self.scheduler.step(1000)
        elif scheduler['name'] == "CosineAnnealingLR":
            self.scheduler = CosineAnnealingLR(self.optimizer, scheduler['cosine_epoch_max'], scheduler['cosine_min_lr'])

        self.sparse = sparse
        self.sparsity_scheduler = SparsityScheduler(mode='random', missing_values=sparse)


    def _forward(self, states, net, train, backward):
        params = net.model_phy.params

        if train:   key_suffix = '_train'
        else:   key_suffix = '_test'

        target, forcing_tensor = states # (batch, space, time), (batch, space, n)
        y0 = target[:, :, 0].permute(1,0) # (batch, space, time) -> (space, batch)
        t = params('NT')
        t = convert_tensor(t, self.device)

        pred = net(y0, t, forcing_tensor)
        assert(pred.shape == target.shape)
        pred_clone = pred.detach().clone()

        pred, target = remove_forced_step_from_learning(pred, target, forcing_tensor, params, net.cond_option)

        loss_dict = {}

        if self.sparse > 0:
            loss_dict['full_traj_loss'] = self.traj_loss(pred, target)
            loss_dict['full_spectre_loss'] = self.spectre_loss(pred, target)

        pred, target = self.sparsity_scheduler(pred, target)

        traj_loss = self.traj_loss(pred, target)
        spectre_loss = self.spectre_loss(pred, target)

        loss_dict['traj_loss' + key_suffix] = traj_loss
        loss_dict['spectre_loss' + key_suffix] = spectre_loss

        if self.backward_spectre_loss and self.backward_traj_loss:
            loss = traj_loss + spectre_loss
        elif self.backward_spectre_loss:
            loss = spectre_loss
        elif self.backward_traj_loss:
            loss = traj_loss
        else:
            loss = None

        loss_dict['loss' + key_suffix] = loss

        if backward:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return loss_dict, pred_clone

    def step(self, is_training:bool, backward=True):
    
        loss_dict = None

        if is_training : batch_iter = self.train_iter
        else: batch_iter = self.test_iter

        for train, net in zip(batch_iter, self.nets):
                
            batch = next(train)
            batch = convert_tensor(batch, self.device)

            # sub iteration
            for _ in range(self.niter):
                loss, output = self._forward(batch.copy(), net, is_training, backward)

            if loss_dict is None:
                loss_dict = {key: 0 for key in loss}

            for key in loss:
                loss_dict[key] += loss[key].item()
        
        for key in loss_dict:
            loss_dict[key] /= len(self.nets)

        return loss_dict, output, batch

    def compute_loss_without_augmentation(self):
        
        memory = self.nets[0].derivative_estimator.is_augmented
        for net in self.nets:
            net.derivative_estimator.is_augmented = False

        with torch.no_grad():

            #init losses
            loss_dict_test = None

            self.test_iter = [iter(test) for test in self.test]
            iteration = 0
            # iteration across test dataset
            while 1:
                try:
                    loss, output, batch = self.val_step()

                    if loss_dict_test is None:
                        loss_dict_test = {key: 0 for key in loss}

                    for key in loss:
                        loss_dict_test[key] += loss[key]

                    iteration += 1

                except StopIteration:
                    break
            
            # mean losses
            for key in loss_dict_test:
                loss_dict_test[key] /= iteration

        for net in self.nets:
            net.derivative_estimator.is_augmented = memory

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            batch = next(iter(self.test[0]))
            DNS_filt_flow = batch[0][0, :, :]
            LES_flow = output[0,:,:]
            os.makedirs(os.path.join(self.path, 'graphics', str(-1)), exist_ok=True)
            graphics2(DNS_filt_flow.detach().cpu().numpy(), LES_flow.detach().cpu().numpy(), self.nets[-1].model_phy.params, os.path.join(self.path, 'graphics',  str(-1)), legend=['Filtered DNS', 'Non-Augmented LES'])

        return loss_dict_test
    
    def compute_loss_with_augmentation(self):

        with torch.no_grad():

            #init losses
            loss_dict_test = None

            self.test_iter = [iter(test) for test in self.test]
            iteration = 0
            # iteration across test dataset
            while 1:
                try:
                    loss, output, batch = self.val_step()

                    if loss_dict_test is None:
                        loss_dict_test = {key: 0 for key in loss}

                    for key in loss:
                        loss_dict_test[key] += loss[key]

                    iteration += 1

                except StopIteration:
                    break
            
            # mean losses
            for key in loss_dict_test:
                loss_dict_test[key] /= iteration

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            batch = next(iter(self.test[0]))
            DNS_filt_flow = batch[0][0, :, :]
            LES_flow = output[0,:,:]
            os.makedirs(os.path.join(self.path, 'graphics', str(0)), exist_ok=True)
            graphics2(DNS_filt_flow.detach().cpu().numpy(), LES_flow.detach().cpu().numpy(), self.nets[-1].model_phy.params, os.path.join(self.path, 'graphics',  str(0)))

        return loss_dict_test

    def run(self, backward):

        def duplicate_dict_metrics(loss_dict_test):

            loss_dict_train = loss_dict_test.copy()
            for key in loss_dict_test:
                loss_dict_train[key[:-4] + 'train'] = loss_dict_train.pop(key)
            
            return {**loss_dict_train, **loss_dict_test}

        early_stop = 0

        print('SubIteration: [epoch] : train loss, test loss')

        loss_dict_test = self.compute_loss_without_augmentation()
        loss_dict = duplicate_dict_metrics(loss_dict_test)
        self.logger.set_metrics_baseline(loss_dict)

        loss_dict_test = self.compute_loss_with_augmentation()
        loss_dict = duplicate_dict_metrics(loss_dict_test)
        self.logger.log(epoch=0, lr=0, metrics=loss_dict, saved=False)

        loss_test_min = None
        for epoch in range(1, self.nepoch+1):
            
            # Training

            # loss init
            loss_dict = None
            
            self.train_iter = [iter(train) for train in self.train]
            iteration = 0

            # iteration across train dataset
            while 1:
                try:
                    loss, _, _ = self.train_step(backward) 

                    if loss_dict is None:
                        loss_dict = {key: 0 for key in loss}

                    for key in loss:
                        loss_dict[key] += loss[key]

                    iteration += 1

                except StopIteration:
                    break

            # mean losses
            for key in loss_dict:
                loss_dict[key] /= iteration

        
            # Validation
            with torch.no_grad():

                #init losses
                loss_dict_test = None

                self.test_iter = [iter(test) for test in self.test]
                iteration = 0
                # iteration across test dataset
                while 1:
                    try:
                        loss, output, batch = self.val_step()

                        if loss_dict_test is None:
                            loss_dict_test = {key: 0 for key in loss}

                        for key in loss:
                            loss_dict_test[key] += loss[key]

                        iteration += 1

                    except StopIteration:
                        break
                
                # mean losses
                for key in loss_dict_test:
                    loss_dict_test[key] /= iteration
                    

                #lr = self.scheduler._last_lr[0]
                lr = self.optimizer.param_groups[-1]['lr']

                # Learning rate update
                if type(self.scheduler) is ReduceLROnPlateau:
                    self.scheduler.step(loss_dict_test['loss_test'])
                elif type(self.scheduler )is CosineAnnealingLR:
                    self.scheduler.step()
                
                # Saving model if improvements
                saved = False
                if loss_test_min == None or loss_test_min > loss_dict_test['loss_test']:
                    loss_test_min = loss_dict_test['loss_test']
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.nets[0].state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss_test_min, 
                        'channel' : self.nets[0].model_aug.channel,
                        'kernel' : self.nets[0].model_aug.kernel
                    }, os.path.join(self.path, 'model.pt'))
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.nets[-1].state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss_test_min, 
                        'channel' : self.nets[-1].model_aug.channel,
                        'kernel' : self.nets[-1].model_aug.kernel
                    }, os.path.join(self.path, 'model2.pt'))
                    saved = True
                    early_stop = 0

                    # Compute graphics
                    # with warnings.catch_warnings():
                    #     warnings.simplefilter("ignore")
                    #     batch = next(iter(self.test[0]))
                    #     batch = convert_tensor(batch, self.device)
                    #     _, output = self._forward(batch, self.nets[0], False, False)
                    #     os.makedirs(os.path.join(self.path, 'graphics', str(epoch)), exist_ok=True)
                    #     # get data for graphics
                    #     DNS_filt_flow = batch[0][0, :, :]
                    #     LES_flow = output[0,:,:]
                    #     graphics2(DNS_filt_flow.cpu().numpy(), LES_flow.cpu().numpy(), self.nets[-1].model_phy.params, os.path.join(self.path, 'graphics',  str(epoch)))
                else:
                    early_stop += 1
                

                self.logger.log(epoch, lr, {**loss_dict, **loss_dict_test}, saved=saved)
                
                # Early stop
                if early_stop >= self.patience:
                    print("Early Stop..")
                    return