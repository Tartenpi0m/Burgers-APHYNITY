import torch
import json

pi = torch.pi

class Params:
    """
    Object that contains all informations to run a DNS or LES simulation without closure.
    This object is needed by my version of APHYNITY
    """

    def __init__(self, filename=None):
        self.__param__ = {}
        self.extra_keys = []

        if filename is not None:
            self.load(filename)

    def __call__(self, key):
        if type(key) == str:
            key = key.upper()
            return self.__param__[key]
        elif type(key) == list:
            return [self.__param__[str(k).upper()] for k in key]

    
    #deprecated, for compatibility only
    def get(self, key):
        return self.__call__(key)
    
    def set_mesh(self, L=2*pi, N=512):
        self.__param__["N"] = N
        self.__param__["L"] = L
        self.__param__["DX"] = L/N
        self.__param__["NX"] = torch.linspace(0 + self.__param__["DX"]/2, L - self.__param__["DX"]/2, N)
    
    def set_time(self, t=1.0, dt=0.001):
        NT = torch.arange(dt, t+dt, dt)
        self.__param__["DT"] = dt
        self.__param__["NT"] = NT
        self.__param__["T"] = t
    
    def set_key(self, key, value):
        self.__param__[key.upper()] = value
    
    def set_nu(self, nu=0.01, Cs=None):
        self.__param__["NU"] = nu
        if Cs is not None:
            self.__param__["CS"] = Cs
    
    def set_filtering(self, spatial_filtering=None, temporal_filtering=None):
        self.__param__['SPATIAL_FILT'] = int(spatial_filtering)
        self.__param__['TEMPORAL_FILT'] = int(temporal_filtering)
        if not self('N') % spatial_filtering == 0:
            print("Warning : SPATIAL_FILT should be a divisor of N")
        if not len(self('NT')) % temporal_filtering == 0:
            print("Warning : TEMPORAL_FILT should be a divisor of duree")

    def set_force(self, step=1, change=1, strength=1, initial_strength=1):
        self.__param__['FORCED_STEP'] = step
        self.__param__['FORCE_CHANGE'] = change
        self.__param__['FORCE_STRENGTH'] = strength
        self.__param__['FORCE_INITIAL_STRENGTH'] = initial_strength
    
    def set_custom(self, key: str, value):
        self.__param__[key.upper()] = value
        self.extra_keys.append(key.upper())

    def save(self, filename):
        to_save = {}
        to_save['N'], to_save['L'], to_save['T'], to_save['DT'], to_save['NU'] = self(['N', 'L', 'T', 'DT', 'NU'])
        
        if 'CS' in self.__param__.keys():
            to_save['CS'] = self('CS')
        if 'SPATIAL_FILT' in self.__param__.keys() and 'TEMPORAL_FILT' in self.__param__.keys():
            to_save['SPATIAL_FILT'], to_save['TEMPORAL_FILT'] = self(['SPATIAL_FILT', 'TEMPORAL_FILT'])
        
        if 'FORCED_STEP' in self.__param__.keys() and 'FORCE_CHANGE' in self.__param__.keys() and 'FORCE_STRENGTH' in self.__param__.keys():
            to_save['FORCED_STEP'], to_save['FORCE_CHANGE'], to_save['FORCE_STRENGTH'], to_save['FORCE_INITIAL_STRENGTH'] = self(['FORCED_STEP', 'FORCE_CHANGE', 'FORCE_STRENGTH', 'FORCE_INITIAL_STRENGTH'])

        for key in self.extra_keys:
            to_save[key] = self(key)

        with open(filename, 'w') as f:
            json.dump(to_save, f)

    def load(self, filename):
        with open(filename, 'r') as f:
            to_load = json.load(f)
        self.set_mesh(to_load['L'], to_load['N'])
        del to_load['L'], to_load['N']
        self.set_time(to_load['T'], to_load['DT'])
        del to_load['T'], to_load['DT']

        if 'CS' in to_load.keys():
            self.set_nu(nu=to_load['NU'], Cs=to_load['CS'])
            del to_load['NU'], to_load['CS']
        else:
            self.set_nu(nu=to_load['NU'])
            del to_load['NU']


        if 'SPATIAL_FILT' in to_load.keys() and 'TEMPORAL_FILT' in to_load.keys():
            self.set_filtering(to_load['SPATIAL_FILT'], to_load['TEMPORAL_FILT'])
            del to_load['SPATIAL_FILT'], to_load['TEMPORAL_FILT']
        
        if 'FORCED_STEP' in to_load.keys() and 'FORCE_CHANGE' in to_load.keys() and 'FORCE_STRENGTH' in to_load.keys() and 'FORCE_INITIAL_STRENGTH' in to_load.keys():
            self.set_force(to_load['FORCED_STEP'], to_load['FORCE_CHANGE'], to_load['FORCE_STRENGTH'], to_load['FORCE_INITIAL_STRENGTH'])
            del to_load['FORCED_STEP'], to_load['FORCE_CHANGE'], to_load['FORCE_STRENGTH'], to_load['FORCE_INITIAL_STRENGTH']
        

        for key in to_load.keys():
            print(key)
            self.set_custom(key, to_load[key])

    def key_exist(self, key: str):

        if key.upper() in self.__param__:
            return True
        else:
            return False