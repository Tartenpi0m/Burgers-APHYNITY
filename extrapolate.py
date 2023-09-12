"""
This file allows to compute simulation with and without APHYNITY on the same time and create graphics that compare both solution
python3 extrapolate.py target_folder source_folder

target_folder : folder containing the model.pt file (the neural net weights) and where results will be stored

source_folder : folder containing simulation parameters that you want to test and in which DNS data will be fetched or saved
example of a source folder : 
    /source_folder/
        /_one_example_of_params/  #must begin by an underscore to be read by the python code
            DNS_params.json
            LES_params.json
"""


import torch
from torch.nn.functional import mse_loss

import sys

import matplotlib.pyplot as plt
# set plt style with dark background
plt.style.use('dark_background')
# add grid to plot with opacity 0.3
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

from mylib.visu import *
from mylib.parameters import Params
from notebook import sim, sim_ram_saver
import os
import sys

from mylib.utils import filter_flow
from mylib.force import generate_force_tensor, maulik_force

torch.set_grad_enabled(False)


sim_time = 50 # simulation time
source_folder = sys.argv[2]
target_folder = sys.argv[1]
print(target_folder)
model = os.path.join(target_folder, "model.pt")
target_folder = os.path.join(target_folder, "extrapolate")
os.makedirs(target_folder, exist_ok=True)

cond_option = "forced"
network = {"channel" : [32,64,64,64], "kernel" : 3}

titles = [
    'Training forcing',
    'Harder and sparser forcing',
    'Easier and more frequent forcing',
    'Dissipation',
    'Super hard and sparse forcing',
    'Biger Nu'
]


batch = 5
cond_folder = [nom_dossier for nom_dossier in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, nom_dossier)) and nom_dossier.startswith('_')]
cond_folder.sort()

for i, f in enumerate(cond_folder):
    DNS_params = Params(os.path.join(source_folder, f, 'DNS_params.json'))
    LES_params = Params(os.path.join(source_folder, f, 'LES_params.json'))

    DNS_params.set_time(t=sim_time, dt=DNS_params('dt'))
    LES_params.set_time(t=sim_time, dt=LES_params('dt'))

    cond_option = 'dissipation'
    if DNS_params.key_exist('forced_step'):
        cond_option = 'forced'

    #os.makedirs(os.path.join(source_folder, f), exist_ok=True)
    os.makedirs(os.path.join(source_folder, f, "data"), exist_ok=True)
    batch_loss = torch.zeros((batch, LES_params('nt').size()[0]))
    batch_no_aug_loss = torch.zeros((batch, LES_params('nt').size()[0]))
    batch_smag_loss = torch.zeros((batch, LES_params('nt').size()[0]))

    for r in range(batch):

        rs = r + 1000 #randomo seed
        # Avec forcage et avec dissipation:

        if os.path.exists(os.path.join(source_folder, f, "data", str(rs))):
            print(f"Loading{os.path.join(source_folder, f, 'data', str(rs))}")
            dns = torch.load(os.path.join(source_folder, f, "data", str(rs)))
            print(f"shape : {dns.shape}")
        else:
            # Simu DNS 10s
            print(f"Generate data {rs}")
            dns = sim(DNS_params, rs, cond_option, True, None, None, None, split_time_period=0.1).squeeze(0)
            torch.save(dns, os.path.join(source_folder, f, "data", str(rs)))
        

        les = sim(LES_params, rs, cond_option, False, network, model, DNS_params).squeeze(0)
        no_aug_les = sim(LES_params, rs, cond_option, False, None, None, DNS_params).squeeze(0)
        smag_les = sim(LES_params, rs, cond_option, False, None, None, DNS_params, mode="smagorinsky").squeeze(0)

        for step in range(les.shape[1]):  
            
            # cacluler la loss à chaque step
            batch_loss[r, step] = mse_loss(dns[:,step], les[:,step])
            batch_no_aug_loss[r, step] = mse_loss(dns[:,step], no_aug_les[:,step])
            batch_smag_loss[r, step] = mse_loss(dns[:,step], smag_les[:,step])

    batch_loss = torch.mean(batch_loss, dim=0)
    batch_no_aug_loss = torch.mean(batch_no_aug_loss, dim=0)
    batch_smag_loss = torch.mean(batch_smag_loss, dim=0)

    os.makedirs(os.path.join(target_folder, f), exist_ok=True)
    loss_folder = os.path.join(target_folder, f, "loss")
    graph_folder = os.path.join(target_folder, f, "graph")
    os.makedirs(loss_folder, exist_ok=True)
    os.makedirs(graph_folder, exist_ok=True)


    plt.figure()
    plt.plot(batch_loss)
    plt.plot(batch_no_aug_loss)
    plt.plot(batch_smag_loss)
    plt.legend(['aug loss', 'no_aug loss', 'smagorinsky loss'])
    plt.title(f)
    #plt.xticks(labels=torch.arange(10,LES_params('t'), 10), ticks=torch.arange(1000, LES_params('t'), 1000))
    plt.xlabel("time (0.01 sec)")
    plt.ylabel("loss")
    plt.savefig(os.path.join(loss_folder, 'loss'))
    plt.close()


    for sec in range(LES_params('t')+1):
        fig_dns = view1D_time_state(sec, dns, LES_params)
        fig_les = view1D_time_state(sec, les, LES_params)
        fig_no_aug_les = view1D_time_state(sec, no_aug_les, LES_params)
        fig = merge_figure([fig_dns, fig_les, fig_no_aug_les], ['DNS', 'Augmented LES', 'Non-augmented LES'])
        fig.suptitle(f + f"_{sec}sec")
        fig.savefig(os.path.join(graph_folder, str(sec)))
        plt.close(fig)
        # Même chose avec un autre dynamique (changer le forcage et dissipation)

print("Fini")