import pickle

import numpy as np
import torch

import ricciardi as ric
import ring_network as network
import sim_util as su

ri = ric.Ricciardi()
ri.set_up_nonlinearity_tensor('./scripts/phi_int_tensor')

NtE = 100
T = torch.linspace(0,NtE*ri.tE,round(NtE*ri.tE/(ri.tI/3))+1)
mask_time = T>(NtE/2*ri.tE)

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
sWIs = np.linspace(20,45,6)
sHs = np.linspace(20,45,6)

try:
    with open('./sim_ring_vary_widths_tensor_results'+'.pkl', 'rb') as handle:
        results_dict = pickle.load(handle)
except:
    results_dict = {}

for sWI_idx,sWI in enumerate(sWIs):
    for sH_idx,sH in enumerate(sHs):
        if (sWI_idx,sH_idx) in results_dict: continue

        this_sW = np.array([[sWE,sWI]]*2)
        this_svarW = this_sW/np.sqrt(2)

        this_sH = sH*np.ones(2)
        this_svarH = this_sH/np.sqrt(2)

        params_dict = {}
        params_dict['Nl'] = 20
        params_dict['NE'] = 150
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

        seeds = np.arange(16)

        net,rates = su.sim_ring_tensor(params_dict,ri,T,mask_time,seeds)
        this_results_dict = su.get_ring_input_rate(params_dict,net,seeds,rates)

        this_results_dict['sWI'] = sWI
        this_results_dict['sH'] = sH

        results_dict[(sWI_idx,sH_idx)] = this_results_dict

        with open('./sim_ring_vary_widths_tensor_results'+'.pkl', 'wb') as handle:
            pickle.dump(results_dict,handle)