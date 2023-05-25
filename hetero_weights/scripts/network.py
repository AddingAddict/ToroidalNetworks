from abc import ABC, abstractmethod

import numpy as np
import torch
from scipy.special import erf, erfi
from mpmath import fp

jtheta = np.vectorize(fp.jtheta, 'D')
        
def make_periodic(x,halfL):
    '''
    Make array x, which is assumed to be in [-L,L], take values in [-L/2,L/2] using periodic boundary conditions

    Parameters
    ----------
    x : array-like
        Input values for to make periodic, assumed to be in [-L,L]
    L : float
        Size of periodic dimension

    Returns
    -------
    array-like
        Input values redefined to be in [-L/2,L/2] using periodic boundary conditions
    '''
    out = np.copy(x);
    out[out >  halfL] =  2*halfL-out[out >  halfL];
    out[out < -halfL] = -2*halfL-out[out < -halfL];
    return out

def apply_kernel(x,S,L,dx=None,kernel='gaussian'):
    '''
    Apply a kernel parameterized by width S to array x, which is assumed to be in [-L/2,L/2]

    Parameters
    ----------
    x : array-like
        Input values for kernel, assumed to be in [-L/2,L/2] 
    S : float
        Width parameter for kernel
    L : float
        Size of periodic dimension
    dx : float, optional
        Distance between locations on discretized dimension, which will rescale output to act as an integration measure
    kernel : str, optional
        Kernel type to apply

    Returns
    -------
    array-like
        Output values where kernel was applied to input values
    '''
    if kernel == 'gaussian':
        out = np.exp(-x**2/(2*S**2))/np.sqrt(2*np.pi)/S
    elif kernel == 'nonnormgaussian':
        out = np.exp(-x**2/(2*S**2))
    elif kernel == 'exponential':
        out = np.exp(-np.abs(x)/S)/S
    elif kernel == 'vonmisses':
        x_rad=x/(L/2)*np.pi
        S_rad=S/(L/2)*np.pi
        d_rad=np.pi/(L/2)
        out = np.exp(np.cos(x_rad)/S_rad)/(2*np.pi*i0(1/S_rad))*d_rad
    elif kernel == 'jtheta':
        x_rad=x/(L/2)*np.pi
        S_rad=S/(L/2)*np.pi
        d_rad=np.pi/(L/2)
        out = np.real(jtheta(3,x_rad/2,np.exp(-S_rad**2/2)))/(2*np.pi)*d_rad
    elif kernel == 'nonnormjtheta':
        x_rad=x/(L/2)*np.pi
        S_rad=S/(L/2)*np.pi
        out = np.real(jtheta(3,dori_rad/2,np.exp(-S_rad**2/2)))/np.real(jtheta(3,0,np.exp(-S_rad**2/2)))
    else:
        raise Exception('kernel not implemented')
    if dx is None:
        return out
    else:
        return out*dx

class BaseNetwork(ABC):
    def __init__(self, seed=0, n=None, NC=None, Nloc=1, profile='gaussian', normalize_by_mean=False):
        self.rng = np.random.default_rng(seed=seed)

        if NC is None:
            if n is None:
                self.NC = np.ones(1,dtype=int)
                self.n = 1
            else:
                self.NC = np.ones(n,dtype=int)
                self.n = int(n)
        elif np.isscalar(NC):
            if n is None:
                self.NC = np.ones(1)
                self.n = 1
            else:
                self.NC = NC*np.ones(n,dtype=int)
                self.n = int(n)
        else:
            if n is None:
                self.NC = np.array(NC,dtype=int)
                self.n = len(self.NC)
            else:
                self.NC = np.array(NC,dtype=int)
                self.n = int(n)
                assert self.n == len(self.NC)

        self.NT = np.sum(self.NC)
        self.Nloc = Nloc

        self.C_idxs = []
        self.C_all = []
        prev_NC = 0
        for cidx in range(self.n):
            prev_NC += 0 if cidx == 0 else self.NC[cidx-1]
            this_NC = self.NC[cidx]
            this_C_idxs = [slice(int(loc*self.NT+prev_NC),int(loc*self.NT+prev_NC+this_NC)) for loc in range(self.Nloc)]
            self.C_idxs.append(this_C_idxs)

            this_C_all = np.zeros(0,np.int8)
            for loc in range(self.Nloc):
                this_C_loc_idxs = np.arange(this_C_idxs[loc].start,this_C_idxs[loc].stop,dtype=int)
                this_C_all = np.append(this_C_all,this_C_loc_idxs)
            self.C_all.append(this_C_all)
        
        # Total number of neurons
        self.N = self.Nloc*self.NT

        self.profile = profile
        self.normalize_by_mean = normalize_by_mean

    def set_seed(self,seed_con):
        self.rng = np.random.default_rng(seed=seed)

        