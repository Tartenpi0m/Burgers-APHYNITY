from forecasters import Forecaster
from mylib.parameters import Params
from networks import SimpleConv, BurgersParamPDE
from mylib.force import generate_force_tensor, maulik_force
from mylib.utils import filter_flow
import torch

def sim(params: Params, random_seed, cond_option='forced', filter=False, network=None, model=None, DNS_params=None, split_time_period=1, mode='default'):
    """
    To perform a DNS, give DNS parameters in 'params'.
    To perform a LES, give LES parameters in 'params' and give DNS parameters in 'DNS_params'.
    """

    model_phy = BurgersParamPDE(mode=mode, params=params)

    model_aug = None
    is_augmented = False
    if network is not None:
        model_aug = SimpleConv(network['channel'], network['kernel'])
        is_augmented = True
        with open(model, 'rb') as file:
            state_dict = torch.load(file)

    net = Forecaster(model_phy=model_phy, cond_option=cond_option, model_aug=model_aug, is_augmented=is_augmented, method='rk4')

    if network is not None:
        net.load_state_dict(state_dict['model_state_dict'])

    forcing_tensor = None
    if cond_option == 'forced':
        if DNS_params is None:
            forcing_tensor = generate_force_tensor(params, random_seed=random_seed).unsqueeze(0)
        else:
            forcing_tensor = filter_flow(generate_force_tensor(DNS_params, random_seed=random_seed).unsqueeze(0), DNS_params('spatial_filt'), 1)

    y0 = None
    if DNS_params is None:
        y0 = maulik_force(params, random_seed=random_seed).view(-1,1) * params('force_initial_strength') # (space, batch)
    else:
        y0 = filter_flow(maulik_force(DNS_params, random_seed=random_seed).view(1,-1,1) * DNS_params('force_initial_strength'), DNS_params('spatial_filt'), 1).view(-1,1) # (space, batch)


    if params('t') < split_time_period or filter is False:
        flow = net(y0, params('nt'), forcing_tensor=forcing_tensor)
        if filter is True:
            flow = filter_flow(flow, params('spatial_filt'), params('temporal_filt'))
    else:
        #flow = net(y0, params('nt'), forcing_tensor=forcing_tensor)
        flow = sim_ram_saver(net, y0, params, forcing_tensor=forcing_tensor, split_time_period=split_time_period)

    if model_aug is None:
        return flow
    else:
        return flow.detach()

def sim_ram_saver(net, y0, params, forcing_tensor=None, split_time_period=1):

    """
    Helper used by sim that saved ram for long dns computation
    """

    assert(params('t') >= split_time_period)
    #assert((split_time_period % 0.0001) == 0)
    
    split_number: float = params('t') / split_time_period 
    if split_number.is_integer():
        split_number = int(split_number)
    else:
        raise ValueError('split_time_period parameter is not set to a valid value')
    
    nt_split = torch.tensor_split(params('nt'), split_number)
    if forcing_tensor is not None:
        forcing_tensor_split = torch.tensor_split(forcing_tensor, split_number, dim=2)
    else:
        forcing_tensor_split = [None] * (split_number)

    flow_list = []
    for i, (nt, forcing_tensor) in enumerate(zip(nt_split, forcing_tensor_split)):

        if i < len(nt_split) - 1:
            nt = torch.concat([nt, nt_split[i+1][:1]])
            if forcing_tensor is not None:
                forcing_tensor = torch.concat([forcing_tensor, forcing_tensor_split[i+1][:,:,:1]], dim=2)

        flow = net(y0, nt, forcing_tensor=forcing_tensor)
        y0 = flow[:, :, -1].view(-1,1)

        if i < len(nt_split) - 1:
            flow = filter_flow(flow[:,:,:-1], params('spatial_filt'), params('temporal_filt'))
        else:
            flow = filter_flow(flow, params('spatial_filt'), params('temporal_filt'))
        flow_list.append(flow)

    return torch.cat(flow_list, dim=2)

