from mylib.rk4 import advect_rk4_explicit, diffuse_rk4_explicit, smagorinsky_rk4_explicit
from mylib.euler import advect_euler_explicit, diffuse_euler_explicit, smagorinsky_euler_explicit
from mylib.spatialSolver import advect_1order_1, advect_o3, smagorinsky_2order, diffusion_2order

import torch
from torch import nn
from torchdiffeq import odeint_adjoint, odeint

from mylib.parameters import Params

def advection(state, dx):
        return - advect_1order_1(state, dx)
    
def default(state, nu, dx):
    return diffusion_2order(state, dx, nu) - advect_1order_1(state, dx)

def smagorinsky(state, nu, dx, dt, Cs):
    return diffusion_2order(state, dx, nu) - advect_1order_1(state, dx) + smagorinsky_2order(state, dx, dt, Cs)
