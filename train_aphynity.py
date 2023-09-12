from torch import optim, set_num_threads
import os, sys, argparse, json

from experiments import APHYNITYExperiment
from networks import BurgersParamPDE
from networks import SupremeNetwork, SimpleConv, NuConv
from forecasters import *
from utils import init_weights
from datasets import param_burgers #, init_dataloaders

from mylib.parameters import Params
from mylib.force import maulik_force, generate_force_tensor
from mylib.utils import filter_flow
from mylib.visu import *



__doc__ = '''Training APHYNITY.'''


KERNEL_SIZE = 3
RANDOM_SEED = 42

def cmdline_args():
        # Make parser object
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    
    p.add_argument("-c", "--cond", type=str, default='forced',
                   help='''choose conditions: 
    'dissipation' - Initial condition only
    'forced' - Forced term during the whole evolution (default)
    ''')
    p.add_argument("-r", "--root", type=str, default='./exp',
                   help='''root path for the experiments. (default: ./exp)''')
    p.add_argument("-p", "--phy", type=str, default='default',
                   help='''choose physical model type: 
    --phy advection - Advection only, without diffusion
    --phy default - Advection-diffusion equation without closure (default)
    --phy smagorinsky - Smagorinsky closure
    --phy none - No physics
    ''')
    p.add_argument("--aug", action=argparse.BooleanOptionalAction, default=True,
                   help='''enable augmentation: 
    --aug - With NN augmentaion (default)
    --no-aug - Without NN augmentation
    ''')
    p.add_argument("-w", "--warm", type=str, default=None,
                   help='''choose model.pt to start training from''')
    p.add_argument('-d', '--device', type=str, default='cpu',
                   help='''choose device:
    'cpu' - CPU only (default)
    'cuda:X' - CUDA device.'''),
    p.add_argument("-t", "--threads", type=int, default=1,
                   help='''choose maximum number of threads used by pytorch''')
    p.add_argument("-l", "--loss", type=str, default="traj",
                   help='''choose losses
                   --loss traj - Trajectory loss only
                   --loss traj-spectre - Spectre + Trajectory loss
                   --loss spectre - Spectre loss only''')
    p.add_argument("-s", "--sparse", type=float, default=0,
                   help=''' choose percentage of space points and time step to remove from learning data (percentage of missing values)
                   --sparse 0 (default)
                   --sparse 0.4

''')
    return p.parse_args()


def train_leads(model_phy_option, model_aug_option, cond_option, path, device, loss, sparse, warm_start_model=None):


    # load config
    if os.path.exists(os.path.join(path, 'config.json')):
        with open(os.path.join(path, 'config.json'), 'r') as f:
            config = json.load(f)
    else: # default config
        print("Default config")
        config = {
                "train data": 5,
                "test data": 2,
                "data path": "data/",
                "lr": 1e-3,
                "batch size": 3,
                "niter": 6,
                "nupdate": 100,
                "nepoch": 50000
            }
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        
    # Model aug
    network = config['network']
    model_aug = None
    if network['name'] == "SupremeNetwork":
        model_aug = SupremeNetwork(network["SupremeNetwork_channel"], network["SupremeNetwork_kernel"], network["SupremeNetwork_layer"])
    elif network['name'] == "SimpleConv":
        model_aug = SimpleConv(network["SimpleConv_channel"], network["SimpleConv_kernel"], network['activation'])
    elif network['name'] == "NuConv":
        model_aug = NuConv(network["NuConv_channel"], network["NuConv_nu_correction_layer"], network["NuConv_kernel"])
    else:
        print("Network not found")
        exit()

    # Data and Model phy
    train_sets = []
    test_sets = []
    nets = []
    params = []
    cond_folder = [nom_dossier for nom_dossier in os.listdir(path) if os.path.isdir(os.path.join(path, nom_dossier)) and nom_dossier.startswith('_')]
    for f in cond_folder:
        DNS_params = Params(os.path.join(path, f, 'DNS_params.json'))
        LES_params = Params(os.path.join(path, f, 'LES_params.json'))

        cond_option = 'dissipation'
        if DNS_params.key_exist('forced_step'):
            cond_option = 'forced'

        train_, test_ = param_burgers(os.path.join(path, f), (config['train data'], config['test data']), DNS_params, batch_size=config['batch size'], method='rk4', cond_option=cond_option)
        train_sets.append(train_)
        test_sets.append(test_)
        params.append(LES_params)
    
        model_phy = BurgersParamPDE(mode=model_phy_option, params=LES_params)
        net = Forecaster(model_phy=model_phy, model_aug=model_aug, is_augmented=model_aug_option, method='rk4', cond_option=cond_option)
        nets.append(net)

    if warm_start_model is not None:
        with open(warm_start_model, 'rb') as file:
            state_dict = torch.load(file)
        nets[0].load_state_dict(state_dict['model_state_dict'])
    
    if loss == 'traj': traj_loss, spectre_loss = True, False
    elif loss == 'traj-spectre': traj_loss, spectre_loss = True, True
    elif loss == 'spectre': traj_loss, spectre_loss = False, True
    else:
        raise ValueError('Unknown --loss (-l) value')

    optimizer = optim.Adam(net.parameters(), lr=config['lr'], betas=(0.9, 0.999))
    experiment = APHYNITYExperiment(
            train=train_sets, test=test_sets, nets=nets, optimizer=optimizer, 
            niter=config['niter'], nepoch=config['nepoch'], path=path, device=device, patience=config['patience'], scheduler=config['scheduler'], sparse=sparse, traj_loss=traj_loss, spectre_loss=spectre_loss
        )
    #with torch.autograd.set_detect_anomaly(True):
    experiment.run(model_aug_option)

if __name__ == '__main__':
    
    if sys.version_info<(3,7,0):
        sys.stderr.write("You need python 3.7 or later to run this script.\n")
        sys.exit(1)
        
    args = cmdline_args()
    path = args.root
    os.makedirs(path, exist_ok=True)
    
    option_dict = {
        'advection': 'Advection only, without diffusion',
        'default': 'Advection-diffusion equation without closure',
        'smagorinsky': 'Smagorinsky closure',
        'none': 'No physics'
    }
    print('#' * 80)
    print('#', option_dict[args.phy], 'is used in F_p')
    print('#', 'F_a is', 'enabled' if args.aug else 'disabled')
    print('#' * 80)

    set_num_threads(args.threads)

    try:
        train_leads(model_phy_option=args.phy, model_aug_option=args.aug, cond_option=args.cond, path=path, device=args.device, warm_start_model=args.warm, loss=args.loss, sparse=args.sparse)
        print("Training ended")
    except KeyboardInterrupt:
        print("Exiting... [Ctrl+C]")
