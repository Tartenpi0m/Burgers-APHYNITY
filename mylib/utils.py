import torch

def filter_flow(flow: torch.Tensor, space=1, time=1):
    """ Filter a tensor of shape (batch x space x time) by averaging over space and sampling over time
    """
    u = flow.clone()
    if time != 1:
        u = u[:, :, ::time]
    if space != 1:
        u = u.view(u.shape[0], u.shape[1]//space, space, u.shape[2]).mean(dim=2)
    return u


def jkl_tensor(u, j=True, l=True):
    """
    Returns 3 shifted arrays of u : u[i-1], u[i], u[i+1]
    """
    uk = u

    if l:
        ul = torch.zeros_like(u)
        ul[:-1] = u[1:]
        ul[-1] = u[0]

    if j:
        uj = torch.zeros_like(u)
        uj[1:] = u[:-1]
        uj[0] = u[-1]
    
    if j and l:
        return uj, uk, ul
    elif j:
        return uj, uk
    elif l:
        return uk, ul
    
def hjklm_tensor(u, h=True, j=True, k=True, l=True, m=True):
    """
    Returns 5 shifted arrays of u : : u[i-2], u[i-1], u[i], u[i+1], u[i+2]
    """
    if h:
        uh = torch.zeros_like(u)
        uh[2:] = u[:-2]
        uh[0] = u[-2]
        uh[1] = u[-1]
    if j:
        uj = torch.zeros_like(u)
        uj[1:] = u[:-1]
        uj[0] = u[-1]
    if k:
        uk = u
    if l:
        ul = torch.zeros_like(u)
        ul[:-1] = u[1:]
        ul[-1] = u[0]
    if m:
        um = torch.zeros_like(u)
        um[:-2] = u[2:]
        um[-2] = u[0]
        um[-1] = u[1]

    return uh, uj, uk, ul, um

def save_set(a_set : set, path):
    with open(path, 'w') as f:
        f.write(str(a_set))
    
def load_set(path):
    with open(path, 'r') as f:
        content = f.read()
    return eval(content)
