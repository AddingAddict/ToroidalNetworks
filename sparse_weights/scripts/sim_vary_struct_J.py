import argparse
import os
try:
    import pickle5 as pickle
except:
    import pickle
import numpy as np
import torch
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.figure import figaspect
import time

import base_network as base_net
import ring_network as network
import sim_util as su
import ricciardi as ric
import integrate as integ

parser = argparse.ArgumentParser()

parser.add_argument('--struct_idx', '-s',  help='which structured', type=int, default=0)
parser.add_argument('--J_idx', '-j',  help='which J', type=int, default=0)
args = vars(parser.parse_args())
print(parser.parse_args())
struct_idx= args['struct_idx']
J_idx= args['J_idx']

id = (133, 0)
with open('./../results/results_ring_'+str(id[0])+'.pkl', 'rb') as handle:
    res_dict = pickle.load(handle)[id[1]]
    prms = res_dict['prms']
    CVh = res_dict['best_monk_eX']
    bX = res_dict['best_monk_bX']
    aXs = res_dict['best_monk_aXs']
    K = prms['K']
    SoriE = prms['SoriE']
    SoriI = prms['SoriI']
    # SoriF = prms['SoriF']
    J = prms['J']
    # beta = prms['beta']
    # gE = prms['gE']
    # gI = prms['gI']
    # hE = prms['hE']
    # hI = prms['hI']
    L = prms['L']
    # CVL = prms['CVL']

ri = ric.Ricciardi()
ri.set_up_nonlinearity('./phi_int')
ri.set_up_nonlinearity_tensor()

NtE = 50
Nt = NtE*ri.tE
dt = ri.tI/5
T = torch.linspace(0,8*Nt,round(8*Nt/dt)+1)
mask_time = T>(4*Nt)
T_mask = T.cpu().numpy()[mask_time]

N = 10000
Nori = 20
NE = 4*(N//Nori)//5
NI = 1*(N//Nori)//5

prms['Nori'] = Nori
prms['NE'] = NE
prms['NI'] = NI

seeds = np.arange(50)

structs = np.arange(0,8+1)/8
Js = J*3**(np.arange(0,8+1)/8)

print('simulating struct # '+str(struct_idx+1))
print('')
struct = structs[struct_idx]

print('simulating J # '+str(J_idx+1))
print('')
J = Js[J_idx]

aX = struct*aXs[-1]
bX = bX + (1-struct)*aXs[-1]

this_prms = prms.copy()

this_prms['J'] = J

base_rates = np.zeros((len(seeds),N))
opto_rates = np.zeros((len(seeds),N))
diff_rates = np.zeros((len(seeds),N))
timeouts = np.zeros((len(seeds))).astype(bool)

for seed_idx,seed in enumerate(seeds):
    print('simulating seed # '+str(seed_idx+1))
    print('')
    
    start = time.process_time()
    
    net,M,H,B,LAS,EPS = su.gen_ring_disorder_tensor(seed,this_prms,CVh)

    print("Generating disorder took ",time.process_time() - start," s")
    print('')
    
    start = time.process_time()

    base_sol,base_timeout = integ.sim_dyn_tensor(ri,T,0.0,M,(bX*B+aX*H)*EPS,LAS,net.C_conds[0],
                                                    mult_tau=True,max_min=60)

    print("Integrating base network took ",time.process_time() - start," s")
    print('')
    
    start = time.process_time()
    
    opto_sol,opto_timeout = integ.sim_dyn_tensor(ri,T,1.0,M,(bX*B+aX*H)*EPS,LAS,net.C_conds[0],
                                                    mult_tau=True,max_min=60)

    print("Integrating opto network took ",time.process_time() - start," s")
    print('')

    diff_sol = opto_sol - base_sol

    base_rates[seed_idx] = np.mean(base_sol[:,mask_time].cpu().numpy(),axis=1)
    opto_rates[seed_idx] = np.mean(opto_sol[:,mask_time].cpu().numpy(),axis=1)
    diff_rates[seed_idx] = np.mean(diff_sol[:,mask_time].cpu().numpy(),axis=1)
    timeouts[seed_idx] = base_timeout or opto_timeout

seed_mask = np.logical_not(timeouts)
vsm_mask = net.get_oriented_neurons()[0]

all_base_means = np.mean(base_rates[seed_mask,:])
all_base_stds = np.std(base_rates[seed_mask,:])
all_opto_means = np.mean(opto_rates[seed_mask,:])
all_opto_stds = np.std(opto_rates[seed_mask,:])
all_diff_means = np.mean(diff_rates[seed_mask,:])
all_diff_stds = np.std(diff_rates[seed_mask,:])
all_norm_covs = np.cov(base_rates[seed_mask,:].flatten(),
    diff_rates[seed_mask,:].flatten())[0,1] / all_diff_stds**2

vsm_base_means = np.mean(base_rates[seed_mask,:][:,vsm_mask])
vsm_base_stds = np.std(base_rates[seed_mask,:][:,vsm_mask])
vsm_opto_means = np.mean(opto_rates[seed_mask,:][:,vsm_mask])
vsm_opto_stds = np.std(opto_rates[seed_mask,:][:,vsm_mask])
vsm_diff_means = np.mean(diff_rates[seed_mask,:][:,vsm_mask])
vsm_diff_stds = np.std(diff_rates[seed_mask,:][:,vsm_mask])
vsm_norm_covs = np.cov(base_rates[seed_mask,:][:,vsm_mask].flatten(),
    diff_rates[seed_mask,:][:,vsm_mask].flatten())[0,1] / vsm_diff_stds**2

res_dict = {}
res_dict['prms'] = this_prms
res_dict['all_base_means'] = all_base_means
res_dict['all_base_stds'] = all_base_stds
res_dict['all_opto_means'] = all_opto_means
res_dict['all_opto_stds'] = all_opto_stds
res_dict['all_diff_means'] = all_diff_means
res_dict['all_diff_stds'] = all_diff_stds
res_dict['all_norm_covs'] = all_norm_covs
res_dict['vsm_base_means'] = vsm_base_means
res_dict['vsm_base_stds'] = vsm_base_stds
res_dict['vsm_opto_means'] = vsm_opto_means
res_dict['vsm_opto_stds'] = vsm_opto_stds
res_dict['vsm_diff_means'] = vsm_diff_means
res_dict['vsm_diff_stds'] = vsm_diff_stds
res_dict['vsm_norm_covs'] = vsm_norm_covs
res_dict['timeouts'] = timeouts

with open('./../results/vary_struct_{:d}_J_{:d}'.format(struct_idx,J_idx)+'.pkl', 'wb') as handle:
    pickle.dump(res_dict,handle)