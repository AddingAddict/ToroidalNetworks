import abc
from functools import partial
import warnings

import numpy as np
import torch
from torchdiffeq import odeint, odeint_event

def sim_dyn(ri,T,L,M,H,LAM,E_cond,mult_tau=False):
    MU = torch.zeros_like(H)
    F = torch.ones_like(H)
    LAS = LAM*L

    # This function computes the dynamics of the rate model
    if mult_tau:
        def ode_fn(t,R):
            torch.matmul(M,R,out=MU)
            torch.add(MU,H,out=MU)
            torch.where(E_cond,ri.tE*MU,ri.tI*MU,out=MU)
            torch.add(MU,LAS,out=MU)
            torch.where(E_cond,(-R+ri.phiE(MU))/ri.tE,(-R+ri.phiI(MU))/ri.tI,out=F)
            return F
    else:
        def ode_fn(t,R):
            torch.matmul(M,R,out=MU)
            torch.add(MU,H + LAS,out=MU)
            torch.where(E_cond,(-R+ri.phiE(MU))/ri.tE,(-R+ri.phiI(MU))/ri.tI,out=F)
            return F

    def event_fn(t,R):
        meanF = torch.mean(torch.abs(F)/torch.maximum(R,1e-1*torch.ones_like(R))) - 5e-3
        if meanF < 0: meanF = 0
        return torch.tensor(meanF)

    # return odeint(ode_fn,torch.zeros_like(H),T[[0,-1]],event_fn=event_fn)
    return odeint(ode_fn,torch.zeros_like(H),T)