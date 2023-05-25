import time

import numpy as np
import torch

import ricciardi as ric
import ring_network as network
import sim_util as su

ri = ric.Ricciardi()
ri.set_up_nonlinearity('./phi_int')
ri.set_up_nonlinearity_tensor('./phi_int_tensor')

NtE = 100

T = np.linspace(0,NtE*ri.tE,round(NtE*ri.tE/(ri.tI/3))+1)
mask_time = T>(NtE/2*ri.tE)

T_torch = torch.linspace(0,NtE*ri.tE,round(NtE*ri.tE/(ri.tI/3))+1)
mask_time_torch = T_torch>(NtE/2*ri.tE)

W = np.array(
    [[ 0.006, -0.0055],
     [ 0.002, -0.0015]])
varW = np.abs(W)*2e-5

H = np.array(
    [0.04, 0.01])
varH = H*2e-5

Lam = 1e-3
CV_Lam = 10
L = 5

sWE = 30
sWI = 30
sH = 20

this_sW = np.array([[sWE,sWI]]*2)
this_svarW = this_sW/np.sqrt(2)

this_sH = sH*np.ones(2)
this_svarH = this_sH/np.sqrt(2)

NEs = [2000,3000,4000,5000]

seeds = np.arange(1)

for NE_idx,NE in enumerate(NEs):
    params_dict = {}
    params_dict['Nl'] = 20
    params_dict['NE'] = NE//20
    params_dict['W'] = W
    params_dict['varW'] = varW
    params_dict['SW'] = this_sW
    params_dict['SvarW'] = this_svarW
    params_dict['H'] = H
    params_dict['varH'] = varH
    params_dict['SH'] = this_sH
    params_dict['SvarH'] = this_svarH
    params_dict['Lam'] = Lam
    params_dict['CV_Lam'] = CV_Lam
    params_dict['L'] = L

    print('Simulating Ring on CPU, N = {:d}'.format(2*NE))
    start = time.time()
    net,rates = su.sim_ring(params_dict,ri,T,mask_time,seeds,max_min=60,stat_stop=False)
    print('Simulation took {:.2f} s\n\n'.format(time.time()-start))

    print('Simulating Ring on GPU, N = {:d}'.format(2*NE))
    start = time.time()
    net,rates = su.sim_ring_tensor(params_dict,ri,T_torch,mask_time_torch,seeds)
    print('Simulation took {:.2f} s\n\n'.format(time.time()-start))

