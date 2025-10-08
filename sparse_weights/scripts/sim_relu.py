import argparse
import ast
import os
try:
    import pickle5 as pickle
except:
    import pickle
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares
import torch
import time

import base_network as base_net
import ring_network as network
import sim_util as su
from ssn import SSN
import integrate as integ

parser = argparse.ArgumentParser()

parser.add_argument('--c_idx', '-c',  help='which contrast', type=int, default=0)
parser.add_argument('--b_idx', '-b',  help='which g1line', type=int, default=0)
parser.add_argument('--K', '-K',  help='number of connections', type=int, default=250)
args = vars(parser.parse_args())
print(parser.parse_args())
c_idx= args['c_idx']
b_idx= args['b_idx']
K= args['K']

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

NE = 320
NI = 80
Nori = 24
Sori = np.array([30,30])
Sstim = 30

with open('./../results/relu_params.pkl','wb') as handle:
    relu_prms = pickle.load(handle)

prms = {
    'K': K,
    'SoriE': Sori[0],
    'SoriI': Sori[1],
    'SoriF': Sstim,
    'J': relu_prms['J'],
    'beta': relu_prms['beta'],
    'gE': relu_prms['gE'],
    'gI': relu_prms['gI'],
    'hE': 100/relu_prms['J']/K,
    'hI': 100/relu_prms['J']/relu_prms['beta']/K,
    'L': 0,
    'CVL': 1,
    'Nori': Nori,
    'NE': NE,
    'NI': NI,
}

ri = SSN(n=1,k=1)

NtE = 50
Nt = NtE*ri.tE
dt = ri.tI/5
T = torch.linspace(0,5*Nt,round(5*Nt/dt)+1)
mask_time = T>(4*Nt)
T_mask = T.cpu().numpy()[mask_time]

seeds = np.arange(10)#200)

print('simulating baseline # '+str(b_idx+1)+' contrast # '+str(c_idx+1))
print('')
base_E = np.array([40,45,40])[b_idx]
base_I = np.array([40,40,45])[b_idx]
con = np.array([0,20,50,100])[c_idx]

cA = con / base_E
rX = base_E / 100

def simulate_networks(prms,rX,cA):
    N = prms.get('Nori',180) * (prms.get('NE',4) + prms.get('NI',1))
    rs = np.zeros((len(seeds),2,N))
    mus = np.zeros((len(seeds),2,N))
    muXs = np.zeros((len(seeds),2,N))
    muEs = np.zeros((len(seeds),2,N))
    muIs = np.zeros((len(seeds),2,N))
    Ls = np.zeros((len(seeds),2))
    TOs = np.zeros((len(seeds),2))

    for seed_idx,seed in enumerate(seeds):
        print('simulating seed # '+str(seed_idx+1))
        print('')
        
        start = time.process_time()

        net,this_M,this_H,this_B,this_LAS,_ = su.gen_ring_disorder_tensor(seed,prms,0)
        this_B[net.C_all[1]] *= base_I / base_E
        this_H2 = torch.roll(this_H,N//2).to(device)
        M = this_M.cpu().numpy()
        H = (rX*(this_B+cA*this_H)).cpu().numpy()
        LAS = this_LAS.cpu().numpy()

        print("Generating disorder took ",time.process_time() - start," s")
        print('')

        start = time.process_time()

        g1_sol,g1_timeout = integ.sim_dyn_tensor(ri,T,0.0,this_M,rX*(this_B+cA*this_H),
                                                     this_LAS,net.C_conds[0],mult_tau=False,max_min=30)
        Ls[seed_idx,0] = np.max(integ.calc_lyapunov_exp_tensor(ri,T[T>=4*Nt],0.0,this_M,
                                                               rX*(this_B+cA*this_H),this_LAS,
                                                               net.C_conds[0],g1_sol[:,T>=4*Nt].to(device),
                                                               10,2*Nt,2*ri.tE,
                                                               mult_tau=False).cpu().numpy())
        rs[seed_idx,0] = np.mean(g1_sol[:,mask_time].cpu().numpy(),-1)
        TOs[seed_idx,0] = g1_timeout

        muXs[seed_idx,0] = H
        muEs[seed_idx,0] = M[:,net.C_all[0]]@rs[seed_idx,0,net.C_all[0]] + H
        muIs[seed_idx,0] = M[:,net.C_all[1]]@rs[seed_idx,0,net.C_all[1]]

        print("Integrating one grating network took ",time.process_time() - start," s")
        print('')
        
        H = (rX*(this_B+cA*(this_H+this_H2))).cpu().numpy()

        start = time.process_time()

        g2_sol,g2_timeout = integ.sim_dyn_tensor(ri,T,0.0,this_M,rX*(this_B+cA*(this_H+this_H2)),
                                                     this_LAS,net.C_conds[0],mult_tau=False,max_min=30)
        Ls[seed_idx,1] = np.max(integ.calc_lyapunov_exp_tensor(ri,T[T>=4*Nt],0.0,this_M,
                                                               rX*(this_B+cA*(this_H+this_H2)),this_LAS,
                                                               net.C_conds[0],g2_sol[:,T>=4*Nt].to(device),
                                                               10,2*Nt,2*ri.tE,
                                                               mult_tau=False).cpu().numpy())
        rs[seed_idx,1] = np.mean(g2_sol[:,mask_time].cpu().numpy(),-1)
        TOs[seed_idx,1] = g2_timeout

        muXs[seed_idx,1] = H
        muEs[seed_idx,1] = M[:,net.C_all[0]]@rs[seed_idx,1,net.C_all[0]] + H
        muIs[seed_idx,1] = M[:,net.C_all[1]]@rs[seed_idx,1,net.C_all[1]]

        print("Integrating two grating network took ",time.process_time() - start," s")
        print('')
        
        mus[seed_idx] = muEs[seed_idx] + muIs[seed_idx]

    return net,rs,mus,muXs,muEs,muIs,Ls,TOs

# Simulate network where structure is removed by increasing g1line fraction
print('simulating g1line fraction network')
print('')
        
μrEs = np.zeros((len(seeds),2,Nori))
μrIs = np.zeros((len(seeds),2,Nori))
ΣrEs = np.zeros((len(seeds),2,Nori))
ΣrIs = np.zeros((len(seeds),2,Nori))
μhEs = np.zeros((len(seeds),2,Nori))
μhIs = np.zeros((len(seeds),2,Nori))
ΣhEs = np.zeros((len(seeds),2,Nori))
ΣhIs = np.zeros((len(seeds),2,Nori))
balEs = np.zeros((len(seeds),2,Nori))
balIs = np.zeros((len(seeds),2,Nori))
Lexps = np.zeros((len(seeds),2))
timeouts = np.zeros((len(seeds),2)).astype(bool)

net,rs,mus,muXs,muEs,muIs,Ls,TOs = simulate_networks(prms,rX,cA)

start = time.process_time()

for nloc in range(Nori):
    μrEs[:,:,nloc] = np.mean(rs[:,:,net.C_idxs[0][nloc]],axis=-1)
    μrIs[:,:,nloc] = np.mean(rs[:,:,net.C_idxs[1][nloc]],axis=-1)
    ΣrEs[:,:,nloc] = np.var(rs[:,:,net.C_idxs[0][nloc]],axis=-1)
    ΣrIs[:,:,nloc] = np.var(rs[:,:,net.C_idxs[1][nloc]],axis=-1)
    μhEs[:,:,nloc] = np.mean(mus[:,:,net.C_idxs[0][nloc]],axis=-1)
    μhIs[:,:,nloc] = np.mean(mus[:,:,net.C_idxs[1][nloc]],axis=-1)
    ΣhEs[:,:,nloc] = np.var(mus[:,:,net.C_idxs[0][nloc]],axis=-1)
    ΣhIs[:,:,nloc] = np.var(mus[:,:,net.C_idxs[1][nloc]],axis=-1)
    balEs[:,:,nloc] = np.mean(np.abs(mus[:,:,net.C_idxs[0][nloc]])/muEs[:,:,net.C_idxs[0][nloc]],axis=-1)
    balIs[:,:,nloc] = np.mean(np.abs(mus[:,:,net.C_idxs[1][nloc]])/muEs[:,:,net.C_idxs[1][nloc]],axis=-1)
Lexps[:,:] = Ls
timeouts[:,:] = TOs

seed_mask = np.logical_not(np.any(timeouts,axis=-1))
vsm_mask = net.get_oriented_neurons(delta_ori=4.5)[0]
osm_mask = net.get_oriented_neurons(delta_ori=4.5,vis_ori=90)[0]

g1_rates = rs[:,0,:]
g2_rates = rs[:,1,:]

all_g1_means = np.mean(g1_rates[seed_mask,:])
all_g1_stds = np.std(g1_rates[seed_mask,:])
all_g2_bals = np.mean(np.abs(mus[seed_mask,1,:])/muEs[seed_mask,1,:])
all_g2_means = np.mean(g2_rates[seed_mask,:])
all_g2_stds = np.std(g2_rates[seed_mask,:])
all_g1_bals = np.mean(np.abs(mus[seed_mask,0,:])/muEs[seed_mask,0,:])

vsm_g1_means = np.mean(g1_rates[seed_mask,:][:,vsm_mask])
vsm_g1_stds = np.std(g1_rates[seed_mask,:][:,vsm_mask])
vsm_g1_bals = np.mean(np.abs(mus[seed_mask,0,:][:,vsm_mask])/muEs[seed_mask,0,:][:,vsm_mask])
vsm_g2_means = np.mean(g2_rates[seed_mask,:][:,vsm_mask])
vsm_g2_stds = np.std(g2_rates[seed_mask,:][:,vsm_mask])
vsm_g2_bals = np.mean(np.abs(mus[seed_mask,1,:][:,vsm_mask])/muEs[seed_mask,1,:][:,vsm_mask])

osm_g1_means = np.mean(g1_rates[seed_mask,:][:,osm_mask])
osm_g1_stds = np.std(g1_rates[seed_mask,:][:,osm_mask])
osm_g1_bals = np.mean(np.abs(mus[seed_mask,0,:][:,osm_mask])/muEs[seed_mask,0,:][:,osm_mask])
osm_g2_means = np.mean(g2_rates[seed_mask,:][:,osm_mask])
osm_g2_stds = np.std(g2_rates[seed_mask,:][:,osm_mask])
osm_g2_bals = np.mean(np.abs(mus[seed_mask,1,:][:,osm_mask])/muEs[seed_mask,1,:][:,osm_mask])

print("Saving statistics took ",time.process_time() - start," s")
print('')

res_dict = {}
    
res_dict['prms'] = prms
res_dict['μrEs'] = μrEs
res_dict['μrIs'] = μrIs
res_dict['ΣrEs'] = ΣrEs
res_dict['ΣrIs'] = ΣrIs
res_dict['μhEs'] = μhEs
res_dict['μhIs'] = μhIs
res_dict['ΣhEs'] = ΣhEs
res_dict['ΣhIs'] = ΣhIs
res_dict['balEs'] = balEs
res_dict['balIs'] = balIs
res_dict['Lexps'] = Lexps
res_dict['all_g1_means'] = all_g1_means
res_dict['all_g1_stds'] = all_g1_stds
res_dict['all_g1_bals'] = all_g1_bals
res_dict['all_g2_means'] = all_g2_means
res_dict['all_g2_stds'] = all_g2_stds
res_dict['all_g2_bals'] = all_g2_bals
res_dict['vsm_g1_means'] = vsm_g1_means
res_dict['vsm_g1_stds'] = vsm_g1_stds
res_dict['vsm_g1_bals'] = vsm_g1_bals
res_dict['vsm_g2_means'] = vsm_g2_means
res_dict['vsm_g2_stds'] = vsm_g2_stds
res_dict['vsm_g2_bals'] = vsm_g2_bals
res_dict['osm_g1_means'] = osm_g1_means
res_dict['osm_g1_stds'] = osm_g1_stds
res_dict['osm_g1_bals'] = osm_g1_bals
res_dict['osm_g2_means'] = osm_g2_means
res_dict['osm_g2_stds'] = osm_g2_stds
res_dict['osm_g2_bals'] = osm_g2_bals
res_dict['timeouts'] = timeouts

res_file = './../results/sim_relu'.format(str(id))
res_file = res_file + '_c_{:d}_b_{:d}_K_{:d}'.format(c_idx,b_idx,K)

with open(res_file+'.pkl', 'wb') as handle:
    pickle.dump(res_dict,handle)
