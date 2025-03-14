import argparse
import os
try:
    import pickle5 as pickle
except:
    import pickle
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares
import time

import ring_network as network
import sim_util as su
import ricciardi as ric
import integrate as integ

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser(description=('This python script takes results from sampled spatial model parameters, '
    'trains a net to interpolate the results, and finds parameters that best fit the experimental results'))

parser.add_argument('--njob', '-n',  help='which number job', type=int, default=0)
args = vars(parser.parse_args())
print(parser.parse_args())
njob= args['njob']

with open('./../notebooks/candidate_prms.txt', 'r') as f:
    id_list = f.read().split('\n')
    id = id_list[njob].replace('(','').replace(')','').replace('\'','').split(', ')
file = id[0]
idx = int(id[1])

print('reading file {:s}'.format(file))
print('reading idx {:d} prm in file'.format(idx))

CVh_mult = 1
CVL_mult = 1

with open('./../results/refit_candidate_prms_{:d}.pkl'.format(njob), 'rb') as handle:
    res_dict = pickle.load(handle)
prms = res_dict['prms']
CVh = res_dict['best_monk_eX']
bX = res_dict['best_monk_bX']
aXs = res_dict['best_monk_aXs']

ri = ric.Ricciardi()
ri.set_up_nonlinearity('./phi_int')
ri.set_up_nonlinearity_tensor()

NtE = 50
Nt = NtE*ri.tE
dt = ri.tI/5
T = torch.linspace(0,5*Nt,round(5*Nt/dt)+1)
mask_time = T>(4*Nt)
T_mask = T.cpu().numpy()[mask_time]

N = 10000
Nori = 20
NE = 4*(N//Nori)//5
NI = 1*(N//Nori)//5

prms = prms.copy()
prms['Nori'] = Nori
prms['NE'] = NE
prms['NI'] = NI

seeds = np.arange(10)

monk_base_means =       np.array([20.38, 43.32, 54.76, 64.54, 70.97, 72.69])
monk_base_stds =        np.array([17.06, 32.41, 38.93, 42.76, 45.17, 48.61])
monk_opto_means =       np.array([30.82, 44.93, 53.36, 60.46, 64.09, 68.87])
monk_opto_stds =        np.array([36.36, 42.87, 45.13, 49.31, 47.53, 52.24])
monk_diff_means =       np.array([10.44,  1.61, -1.41, -4.08, -6.88, -3.82])
monk_diff_stds =        np.array([37.77, 42.48, 42.24, 45.43, 41.78, 41.71])
monk_norm_covs =        np.array([-0.1456, -0.2999, -0.3792, -0.3831, -0.4664, -0.4226])

monk_base_means_err =   np.array([ 2.39,  4.49,  5.38,  5.90,  6.22,  6.69])
monk_base_stds_err =    np.array([ 2.29,  3.67,  4.48,  5.10,  5.61,  5.32])
monk_opto_means_err =   np.array([ 5.03,  5.86,  6.16,  6.74,  6.50,  7.15])
monk_opto_stds_err =    np.array([ 7.73,  6.47,  5.90,  6.20,  4.93,  4.74])
monk_diff_means_err =   np.array([ 5.28,  5.90,  5.84,  6.28,  5.75,  5.76])
monk_diff_stds_err =    np.array([ 8.36,  8.74,  8.01, 10.04,  8.51,  8.94])
monk_norm_covs_err =    np.array([ 0.1075,  0.1354,  0.1579,  0.1496,  0.1717,  0.1665])

monk_nc = len(monk_base_means)

big_start = time.process_time()

all_base_means = np.zeros((len(aXs),len(seeds)))
all_base_stds = np.zeros((len(aXs),len(seeds)))
all_opto_means = np.zeros((len(aXs),len(seeds)))
all_opto_stds = np.zeros((len(aXs),len(seeds)))
all_diff_means = np.zeros((len(aXs),len(seeds)))
all_diff_stds = np.zeros((len(aXs),len(seeds)))
all_norm_covs = np.zeros((len(aXs),len(seeds)))

vsm_base_means = np.zeros((len(aXs),len(seeds)))
vsm_base_stds = np.zeros((len(aXs),len(seeds)))
vsm_opto_means = np.zeros((len(aXs),len(seeds)))
vsm_opto_stds = np.zeros((len(aXs),len(seeds)))
vsm_diff_means = np.zeros((len(aXs),len(seeds)))
vsm_diff_stds = np.zeros((len(aXs),len(seeds)))
vsm_norm_covs = np.zeros((len(aXs),len(seeds)))

osm_base_means = np.zeros((len(aXs),len(seeds)))
osm_base_stds = np.zeros((len(aXs),len(seeds)))
osm_opto_means = np.zeros((len(aXs),len(seeds)))
osm_opto_stds = np.zeros((len(aXs),len(seeds)))
osm_diff_means = np.zeros((len(aXs),len(seeds)))
osm_diff_stds = np.zeros((len(aXs),len(seeds)))
osm_norm_covs = np.zeros((len(aXs),len(seeds)))

timeouts = np.zeros((len(aXs),len(seeds))).astype(bool)

for aX_idx,aX in enumerate(aXs):
    if aX_idx > 0 and np.all(timeouts[aX_idx-1,:]):
        all_base_means[aX_idx,:] = all_base_means[aX_idx-1,:] + aX
        all_base_stds[aX_idx,:] = all_base_stds[aX_idx-1,:] + aX
        all_opto_means[aX_idx,:] = all_opto_means[aX_idx-1,:] + aX
        all_opto_stds[aX_idx,:] = all_opto_stds[aX_idx-1,:] + aX
        all_diff_means[aX_idx,:] = all_diff_means[aX_idx-1,:] + aX
        all_diff_stds[aX_idx,:] = all_diff_stds[aX_idx-1,:] + aX
        all_norm_covs[aX_idx,:] = all_norm_covs[aX_idx-1,:] + aX
        vsm_base_means[aX_idx,:] = vsm_base_means[aX_idx-1,:] + aX
        vsm_base_stds[aX_idx,:] = vsm_base_stds[aX_idx-1,:] + aX
        vsm_opto_means[aX_idx,:] = vsm_opto_means[aX_idx-1,:] + aX
        vsm_opto_stds[aX_idx,:] = vsm_opto_stds[aX_idx-1,:] + aX
        vsm_diff_means[aX_idx,:] = vsm_diff_means[aX_idx-1,:] + aX
        vsm_diff_stds[aX_idx,:] = vsm_diff_stds[aX_idx-1,:] + aX
        vsm_norm_covs[aX_idx,:] = vsm_norm_covs[aX_idx-1,:] + aX
        osm_base_means[aX_idx,:] = osm_base_means[aX_idx-1,:] + aX
        osm_base_stds[aX_idx,:] = osm_base_stds[aX_idx-1,:] + aX
        osm_opto_means[aX_idx,:] = osm_opto_means[aX_idx-1,:] + aX
        osm_opto_stds[aX_idx,:] = osm_opto_stds[aX_idx-1,:] + aX
        osm_diff_means[aX_idx,:] = osm_diff_means[aX_idx-1,:] + aX
        osm_diff_stds[aX_idx,:] = osm_diff_stds[aX_idx-1,:] + aX
        osm_norm_covs[aX_idx,:] = osm_norm_covs[aX_idx-1,:] + aX
        timeouts[aX_idx,:] = True
        continue
            
    print('simulating aX # '+str(aX_idx+1))
    print('')
    
    rX = bX
    cA = aX/bX

    for seed_idx,seed in enumerate(seeds):
        if seed_idx > 0 and timeouts[aX_idx,seed_idx-1]:
            all_base_means[aX_idx,seed_idx] = all_base_means[aX_idx,seed_idx-1] + 100*seed
            all_base_stds[aX_idx,seed_idx] = all_base_stds[aX_idx,seed_idx-1] + 100*seed
            all_opto_means[aX_idx,seed_idx] = all_opto_means[aX_idx,seed_idx-1] + 100*seed
            all_opto_stds[aX_idx,seed_idx] = all_opto_stds[aX_idx,seed_idx-1] + 100*seed
            all_diff_means[aX_idx,seed_idx] = all_diff_means[aX_idx,seed_idx-1] + 100*seed
            all_diff_stds[aX_idx,seed_idx] = all_diff_stds[aX_idx,seed_idx-1] + 100*seed
            all_norm_covs[aX_idx,seed_idx] = all_norm_covs[aX_idx,seed_idx-1] + 100*seed
            vsm_base_means[aX_idx,seed_idx] = vsm_base_means[aX_idx,seed_idx-1] + 100*seed
            vsm_base_stds[aX_idx,seed_idx] = vsm_base_stds[aX_idx,seed_idx-1] + 100*seed
            vsm_opto_means[aX_idx,seed_idx] = vsm_opto_means[aX_idx,seed_idx-1] + 100*seed
            vsm_opto_stds[aX_idx,seed_idx] = vsm_opto_stds[aX_idx,seed_idx-1] + 100*seed
            vsm_diff_means[aX_idx,seed_idx] = vsm_diff_means[aX_idx,seed_idx-1] + 100*seed
            vsm_diff_stds[aX_idx,seed_idx] = vsm_diff_stds[aX_idx,seed_idx-1] + 100*seed
            vsm_norm_covs[aX_idx,seed_idx] = vsm_norm_covs[aX_idx,seed_idx-1] + 100*seed
            timeouts[aX_idx,seed_idx] = True
            continue
            
        print('simulating seed # '+str(seed_idx+1))
        print('')
        
        start = time.process_time()

        net,this_M,this_H,this_B,this_LAS,this_EPS = su.gen_ring_disorder_tensor(seed,prms,CVh)
        M = this_M.cpu().numpy()
        H = (rX*(this_B+cA*this_H)*this_EPS).cpu().numpy()
        LAS = this_LAS.cpu().numpy()

        print("Generating disorder took ",time.process_time() - start," s")
        print('')
        
        start = time.process_time()

        base_sol,base_timeout = integ.sim_dyn_tensor(ri,T,0.0,this_M,rX*(this_B+cA*this_H)*this_EPS,
                                                    this_LAS,net.C_conds[0],mult_tau=True,max_min=15)
        opto_sol,opto_timeout = integ.sim_dyn_tensor(ri,T,1.0,this_M,rX*(this_B+cA*this_H)*this_EPS,
                                                    this_LAS,net.C_conds[0],mult_tau=True,max_min=15)
        diff_sol = opto_sol - base_sol
        timeout = base_timeout or opto_timeout

        print("Integrating networks took ",time.process_time() - start," s")
        print('')

        base_rates = np.mean(base_sol[:,mask_time].cpu().numpy(),axis=1)
        opto_rates = np.mean(opto_sol[:,mask_time].cpu().numpy(),axis=1)
        diff_rates = np.mean(diff_sol[:,mask_time].cpu().numpy(),axis=1)

        all_base_means[aX_idx,seed_idx] = np.mean(base_rates)
        all_base_stds[aX_idx,seed_idx] = np.std(base_rates)
        all_opto_means[aX_idx,seed_idx] = np.mean(opto_rates)
        all_opto_stds[aX_idx,seed_idx] = np.std(opto_rates)
        all_diff_means[aX_idx,seed_idx] = np.mean(diff_rates)
        all_diff_stds[aX_idx,seed_idx] = np.std(diff_rates)
        all_norm_covs[aX_idx,seed_idx] = np.cov(base_rates,
            diff_rates)[0,1] / all_diff_stds[aX_idx,seed_idx]**2

        vsm_base_means[aX_idx,seed_idx] = np.mean(base_rates[net.get_oriented_neurons(delta_ori=4.5)])
        vsm_base_stds[aX_idx,seed_idx] = np.std(base_rates[net.get_oriented_neurons(delta_ori=4.5)])
        vsm_opto_means[aX_idx,seed_idx] = np.mean(opto_rates[net.get_oriented_neurons(delta_ori=4.5)])
        vsm_opto_stds[aX_idx,seed_idx] = np.std(opto_rates[net.get_oriented_neurons(delta_ori=4.5)])
        vsm_diff_means[aX_idx,seed_idx] = np.mean(diff_rates[net.get_oriented_neurons(delta_ori=4.5)])
        vsm_diff_stds[aX_idx,seed_idx] = np.std(diff_rates[net.get_oriented_neurons(delta_ori=4.5)])
        vsm_norm_covs[aX_idx,seed_idx] = np.cov(base_rates[net.get_oriented_neurons(delta_ori=4.5)],
            diff_rates[net.get_oriented_neurons(delta_ori=4.5)])[0,1] / vsm_diff_stds[aX_idx,seed_idx]**2

        osm_base_means[aX_idx,seed_idx] = np.mean(base_rates[net.get_oriented_neurons(delta_ori=4.5,vis_ori=90)])
        osm_base_stds[aX_idx,seed_idx] = np.std(base_rates[net.get_oriented_neurons(delta_ori=4.5,vis_ori=90)])
        osm_opto_means[aX_idx,seed_idx] = np.mean(opto_rates[net.get_oriented_neurons(delta_ori=4.5,vis_ori=90)])
        osm_opto_stds[aX_idx,seed_idx] = np.std(opto_rates[net.get_oriented_neurons(delta_ori=4.5,vis_ori=90)])
        osm_diff_means[aX_idx,seed_idx] = np.mean(diff_rates[net.get_oriented_neurons(delta_ori=4.5,vis_ori=90)])
        osm_diff_stds[aX_idx,seed_idx] = np.std(diff_rates[net.get_oriented_neurons(delta_ori=4.5,vis_ori=90)])
        osm_norm_covs[aX_idx,seed_idx] = np.cov(base_rates[net.get_oriented_neurons(delta_ori=4.5,vis_ori=90)],
            diff_rates[net.get_oriented_neurons(delta_ori=4.5,vis_ori=90)])[0,1] / osm_diff_stds[aX_idx,seed_idx]**2
        
        if timeout:
            all_norm_covs[aX_idx,seed_idx] = 1000
            vsm_norm_covs[aX_idx,seed_idx] = 1000
            osm_norm_covs[aX_idx,seed_idx] = 1000

        timeout = timeout or vsm_base_means[aX_idx,seed_idx] > 100
        
        timeouts[aX_idx,seed_idx] = timeout

print("Simulating inputs took ",time.process_time() - big_start," s")
print('')

all_norm_covs = np.mean(all_norm_covs*all_diff_stds**2,-1)
all_base_means = np.mean(all_base_means,-1)
all_base_stds = np.sqrt(np.mean(all_base_stds**2,-1))
all_opto_means = np.mean(all_opto_means,-1)
all_opto_stds = np.sqrt(np.mean(all_opto_stds**2,-1))
all_diff_means = np.mean(all_diff_means,-1)
all_diff_stds = np.sqrt(np.mean(all_diff_stds**2,-1))
all_norm_covs = all_norm_covs / all_diff_stds**2
vsm_norm_covs = np.mean(vsm_norm_covs*vsm_diff_stds**2,-1)
vsm_base_means = np.mean(vsm_base_means,-1)
vsm_base_stds = np.sqrt(np.mean(vsm_base_stds**2,-1))
vsm_opto_means = np.mean(vsm_opto_means,-1)
vsm_opto_stds = np.sqrt(np.mean(vsm_opto_stds**2,-1))
vsm_diff_means = np.mean(vsm_diff_means,-1)
vsm_diff_stds = np.sqrt(np.mean(vsm_diff_stds**2,-1))
vsm_norm_covs = vsm_norm_covs / vsm_diff_stds**2
osm_norm_covs = np.mean(osm_norm_covs*osm_diff_stds**2,-1)
osm_base_means = np.mean(osm_base_means,-1)
osm_base_stds = np.sqrt(np.mean(osm_base_stds**2,-1))
osm_opto_means = np.mean(osm_opto_means,-1)
osm_opto_stds = np.sqrt(np.mean(osm_opto_stds**2,-1))
osm_diff_means = np.mean(osm_diff_means,-1)
osm_diff_stds = np.sqrt(np.mean(osm_diff_stds**2,-1))
osm_norm_covs = osm_norm_covs / osm_diff_stds**2

start = time.process_time()

res_dict = {}
res_dict['prms'] = prms
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
res_dict['osm_base_means'] = osm_base_means
res_dict['osm_base_stds'] = osm_base_stds
res_dict['osm_opto_means'] = osm_opto_means
res_dict['osm_opto_stds'] = osm_opto_stds
res_dict['osm_diff_means'] = osm_diff_means
res_dict['osm_diff_stds'] = osm_diff_stds
res_dict['osm_norm_covs'] = osm_norm_covs
res_dict['timeouts'] = timeouts

res_dict['file'] = file
res_dict['prm_idx'] = idx
res_dict['idx'] = (file,idx)

with open('./../results/candidate_prms_baseline_supp_{:d}'.format(njob)+'.pkl',
          'wb') as handle:
    pickle.dump(res_dict,handle)
