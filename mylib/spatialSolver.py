from torch import abs, max
import torch
from .utils import hjklm_tensor, jkl_tensor

def advect_1order_1(u, dx):
    """ Advect a one dimensional array u with second order scheme and first order artficial dissipation.
    """
    uj, uk, ul = jkl_tensor(u)
    return (ul**2 - uj**2)/(4*dx) - ( 0.5 * ((abs(ul + uk)/2) * (ul - uk)  -  (abs(uk + uj) / 2) * (uk - uj))) / dx

def advect_2order_3(u, dx , k4=1/12):
    """ Advect a one dimensional array u with second order scheme and third order artficial dissipation.
    """
    uh, uj, uk, ul, um = hjklm_tensor(u)
    return (ul**2 - uj**2)/(4*dx) + k4/dx * ((abs(ul + uk) / 2)*(um - 3*ul + 3* uk - uj)  - (abs(uk + uj)/2) * (ul - 3*uk + 3*uj - uh))

def advect_4order_3(u, dx, k4=1/12):
    """ Advect a one dimensional array u with fourth order scheme and third order artficial dissipation.
    """
    uh, uj, uk, ul, um = hjklm_tensor(u)
    return (ul**2 - uj**2)/(4*dx) - (1/6) * (um**2 - 2*ul**2 + 2*uj**2 + uh**2) / (2*dx) - k4 * ((abs(ul + uk) / 2)*(um - 3*ul + 3* uk - uj)  - (abs(uk + uj)/2) * (ul - 3*uk + 3*uj - uh))




def advect_o2(u, dx, k4=0.0032, k2=1):

    uh, uj, uk, ul, um = hjklm_tensor(u)
    second_order_derivative = (ul**2 - uj**2)/(4*dx)

    psi = abs(ul - 2*uk + uj)/abs(ul + 2*uk + uj)
    #psi_l2 = abs(um - 2*ul + uk)/abs(um + 2*ul + uk)
    psik, psil = jkl_tensor(psi, j=False)

    eps2 = k2 * max(psil, psik)
    #eps2_j2 = k2 * abs(uk + uj) / 2 * max(psik, psij)
    eps2j, eps2k = jkl_tensor(eps2, l=False)

    eps4 = max(k4 - eps2k, torch.zeros_like(eps2k))
    eps4j, eps4k = jkl_tensor(eps4, l=False)

    third_order_artifical_dissipation = (1 / dx) * (eps4k * (abs(ul + uk) / 2)*(um - 3*ul + 3* uk - uj)  - eps4j * (abs(uk + uj)/2) * (ul - 3*uk + 3*uj - uh))
    
    stabilisation_mode = (eps2k * (abs(ul + uk) / 2) * (ul - uk) - eps2j * (abs(uk + uj)/2) * (uk - uj)) / dx
    
    return second_order_derivative + third_order_artifical_dissipation - stabilisation_mode

def advect_o3(u, dx, k4=0.0032, k2=0.5):

    uh, uj, uk, ul, um = hjklm_tensor(u)
    fh, fj, fk, fl, fm = hjklm_tensor((u**2)/2)
    third_order_derivative = (-1/12*dx) * (fm - 8*fl + 8*fj - fh)
    psi = abs(ul - 2*uk + uj)/abs(ul + 2*uk + uj)
    #psi_l2 = abs(um - 2*ul + uk)/abs(um + 2*ul + uk)
    psik, psil = jkl_tensor(psi, j=False)

    eps2 = k2 * max(psil, psik)
    #eps2_j2 = k2 * abs(uk + uj) / 2 * max(psik, psij)
    eps2j, eps2k = jkl_tensor(eps2, l=False)

    eps4 = max(k4 - eps2k, torch.zeros_like(eps2k))
    eps4j, eps4k = jkl_tensor(eps4, l=False)

    third_order_artifical_dissipation = (1 / dx) * (eps4k * (abs(ul + uk) / 2)*(um - 3*ul + 3* uk - uj)  - eps4j * (abs(uk + uj)/2) * (ul - 3*uk + 3*uj - uh))
    
    stabilisation_mode = (eps2k * (abs(ul + uk) / 2) * (ul - uk) - eps2j * (abs(uk + uj)/2) * (uk - uj)) / dx

    return third_order_derivative + third_order_artifical_dissipation - stabilisation_mode


def diffusion_2order(u, dx, nu):
    """ Diffuse with 2 oder scheme """
    uj, uk, ul = jkl_tensor(u)
    return nu * (ul - 2*uk + uj)/(dx**2)

def smagorinsky_2order(u, dx, dt, Cs=0.1):
    """ Smagorinsky model for LES with 2 order scheme"""
    uj, uk, ul = jkl_tensor(u)
    dudx_l = ((ul - uk)/(dx))
    dudx_k = ((uk - uj)/(dx))
    return ((Cs*dt)**2 * abs(dudx_l) * dudx_l - (Cs*dt)**2 * abs(dudx_k) * dudx_k) / dx
    