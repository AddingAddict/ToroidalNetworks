import argparse

import pickle

import numpy as np

import ricciardi as ric
import ring_network as network
import sim_util as su

parser = argparse.ArgumentParser()
parser.add_argument('--sWI_idx', '-sWI', help='version',type=int, default=0)
parser.add_argument('--sH_idx', '-sH', help='version',type=int, default=0)
args = vars(parser.parse_args())
sWI_idx = int(args['sWI_idx'])
sH_idx = int(args['sH_idx'])

ri = ric.Ricciardi()
ri.set_up_nonlinearity('./phi_int')

NtE = 100
T = np.linspace(0,NtE*ri.tE,round(NtE*ri.tE/(ri.tI/3))+1)
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
sWIs = np.linspace(15,45,7)
sHs = np.linspace(15,45,7)

sWI = sWIs[sWI_idx]
sH = sHs[sH_idx]

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

net,rates,_ = su.sim_ring(params_dict,ri,T,mask_time,seeds)
results_dict = su.get_ring_input_rate(params_dict,net,seeds,rates)

results_dict['sWI'] = sWI
results_dict['sH'] = sH

with open('./../results/sim_ring_vary_widths_array_results_sWI={:d}_sH={:d}'.format(int(sWI),int(sH))+'.pkl', 'wb') as handle:
    pickle.dump(results_dict,handle)