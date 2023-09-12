import numpy as np
import torch

from mylib.parameters import Params

class force:
    """
    Contiens les informations et fonction utiles pour forcer une simulation
    Cette objet est utilisé par ma version modifié de la bibliothèque torchdiffeq pour forcer les simulation calculer par APHYNITY
    """
    def __init__(self, params: Params, kmax=None, k0=10):
        self.params = params
        self.change = params('force_change')
        self.step = params('forced_step')
        self.strength = params('force_strength')
        self.kmax = kmax
        self.k0 = k0

    def __call__(self, random_seed=None):
        return maulik_force(self.params, self.kmax, self.k0, random_seed)
    
    def generate_force_tensor(self, random_seed=None):
        return generate_force_tensor(self.params, random_seed)

    def init_loop(self, forcing_tensor):
        self.step_count = 1
        self.change_count = 1
        self.force_indice = 0
        self.forcing_tensor = forcing_tensor 
    
    def step_loop(self):

        if self.change_count == 1:
            forced_term = self.forcing_tensor[:, :, self.force_indice] # (batch, space, n) -> (batch, space)

        if self.change_count == self.change:
            self.force_indice += 1
            self.change_count = 0

        if self.step_count == self.step:
            forced_term = self.forcing_tensor[:, :, self.force_indice] # (batch, space, n) -> (batch, space)
            self.step_count = 0
        else:
            forced_term = 0

        self.step_count += 1
        self.change_count += 1

        if isinstance(forced_term, int):
            return forced_term
        return forced_term.permute(1,0) # (batch, space) -> (space, batch)
        
def generate_force_tensor(params: Params, random_seed=None):
        
    """ 
    A partir d'une seed, génère toutes les forces qui seront utilisé pour forcer le système. 
    Toujours générer les forces avec cette fonction permet de forcer le système de la même 
    manière sur deux simulation différentes et donc notamment de comparer une LES et DNS forcées.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n = len(params('nt')) // params('force_change')
    forcing_tensor = torch.stack([maulik_force(params) * params('force_strength') for i in range(n)]).permute(1,0) 
    forcing_tensor = forcing_tensor
    return forcing_tensor  # (space, n)

def maulik_force(params: Params, kmax=None, k0=10, random_seed=None, return_energy=False):
    """
    Returns a function that generates a force/initial_condition with the Maulik spectrum
    :param params: Parameters object
    :param kmax: Maximum k value, if None, kmax = N/L
    INFO :  nu = 5*10e-4
            L =  2*np.pi
            N = 512-32768
    """
    N = params.get('N')
    L = params.get('L')

    if random_seed is not None:
        np.random.seed(random_seed)

    if kmax is None:
        kmax = 2*np.pi*N/L
    k = np.linspace(2*np.pi/L, kmax, N)

    A = (2*k0**-5 )/ (3 * np.sqrt(np.pi))
    Ek = A * k**4 * np.exp(-(k/k0)**2)

    if return_energy:
        return Ek

    new_psi_k = np.random.rand(k.size)
    uk = N * np.sqrt(2*Ek) * np.exp(2j*np.pi * new_psi_k)
    force = np.fft.ifft(uk).real
    return torch.from_numpy(force)