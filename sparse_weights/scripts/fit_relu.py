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

parser.add_argument('--job_id', '-i', help='completely arbitrary job id label',type=int, default=0)
parser.add_argument('--bayes_iter', '-bi', help='bayessian inference interation (0 = use prior, 1 = use first posterior)',type=int, default=0)
parser.add_argument('--num_samp', '-ns', help='number of samples',type=int, default=50)
parser.add_argument('--K', '-K',  help='number of connections', type=int, default=300)
args = vars(parser.parse_args())
print(parser.parse_args())
job_id = int(args['job_id'])
bayes_iter = int(args['bayes_iter'])
num_samp = int(args['num_samp'])
K= args['K']

# device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

NE = 160
NI = 40
Nori = 24
Sori = np.array([30,30])
Sstim = 30

ri = SSN(n=1,k=1)

NtE = 50
Nt = NtE*ri.tE
dt = ri.tI/5
T = torch.linspace(0,5*Nt,round(5*Nt/dt)+1)
mask_time = T>(4*Nt)
T_mask = T.cpu().numpy()[mask_time]

seeds = np.arange(5)

def simulate_networks(prms,rX,cA):
    N = prms.get('Nori',180) * (prms.get('NE',4) + prms.get('NI',1))
    rs = np.zeros((len(seeds),N))
    mus = np.zeros((len(seeds),N))
    muXs = np.zeros((len(seeds),N))
    muEs = np.zeros((len(seeds),N))
    muIs = np.zeros((len(seeds),N))
    Ls = np.zeros((len(seeds),))
    TOs = np.zeros((len(seeds),))

    for seed_idx,seed in enumerate(seeds):
        print('simulating seed # '+str(seed_idx+1))
        print('')
        
        start = time.process_time()

        net,this_M,this_H,this_B,this_LAS,_ = su.gen_ring_disorder_tensor(seed,prms,0,device=device)
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
        Ls[seed_idx] = np.max(integ.calc_lyapunov_exp_tensor(ri,T[T>=4*Nt],0.0,this_M,
                                                               rX*(this_B+cA*this_H),this_LAS,
                                                               net.C_conds[0],g1_sol[:,T>=4*Nt].to(device),
                                                               10,2*Nt,2*ri.tE,
                                                               mult_tau=False).cpu().numpy())
        rs[seed_idx] = np.mean(g1_sol[:,mask_time].cpu().numpy(),-1)
        TOs[seed_idx] = g1_timeout

        muXs[seed_idx] = H
        muEs[seed_idx] = M[:,net.C_all[0]]@rs[seed_idx,net.C_all[0]] + H
        muIs[seed_idx] = M[:,net.C_all[1]]@rs[seed_idx,net.C_all[1]]
        
        mus[seed_idx] = muEs[seed_idx] + muIs[seed_idx]

    return net,rs,mus,muXs,muEs,muIs,Ls,TOs

rng = np.random.default_rng(job_id)

if bayes_iter==0:
    Js = 10**rng.uniform(-4,-3,size=num_samp)
    betas = 10**rng.uniform(-1,0.7,size=num_samp)
    gEs = rng.uniform(3,7,size=num_samp)
    gIs = rng.uniform(2,gEs-0.5,size=num_samp)
else:
    with open(f'./../results/relu_posterior_{bayes_iter:d}.pkl','rb') as handle:
        posterior = pickle.load(handle)
    samples = posterior.sample((num_samp,)).cpu().numpy()
    Js = 10**samples[:,0]
    betas = 10**samples[:,1]
    gEs = samples[:,2]
    gIs = 2 + (gEs-2.5)*samples[:,3]

vsm_g1_means = np.zeros((num_samp,3))
vsm_g1_stds = np.zeros((num_samp,3))
vsm_g1_bals = np.zeros((num_samp,3))

osm_g1_means = np.zeros((num_samp,3))
osm_g1_stds = np.zeros((num_samp,3))
osm_g1_bals = np.zeros((num_samp,3))

g1_Lexps = np.zeros((num_samp,3))
g1_timeouts = np.zeros((num_samp,3)).astype(bool)

for idx,(J,beta,gE,gI) in enumerate(zip(Js,betas,gEs,gIs)):
    print('simulating parameter set # '+str(idx+1)+' / '+str(num_samp))
    
    start = time.process_time()
    
    prms = {
        'K': K,
        'SoriE': Sori[0],
        'SoriI': Sori[1],
        'SoriF': Sstim,
        'J': J,
        'beta': beta,
        'gE': gE,
        'gI': gI,
        'hE': 100/J/K,
        'hI': 100/J/beta/K,
        'L': 0,
        'CVL': 1,
        'Nori': Nori,
        'NE': NE,
        'NI': NI,
    }
    
    for b_idx in range(3):
        base_E = np.array([40,45,40])[b_idx]
        base_I = np.array([40,40,45])[b_idx]
        con = 100

        cA = con / base_E
        rX = base_E / 100
                
        μrEs = np.zeros((len(seeds),Nori))
        μrIs = np.zeros((len(seeds),Nori))
        ΣrEs = np.zeros((len(seeds),Nori))
        ΣrIs = np.zeros((len(seeds),Nori))
        μhEs = np.zeros((len(seeds),Nori))
        μhIs = np.zeros((len(seeds),Nori))
        ΣhEs = np.zeros((len(seeds),Nori))
        ΣhIs = np.zeros((len(seeds),Nori))
        balEs = np.zeros((len(seeds),Nori))
        balIs = np.zeros((len(seeds),Nori))
        Lexps = np.zeros((len(seeds),))
        timeouts = np.zeros((len(seeds),)).astype(bool)

        net,rs,mus,muXs,muEs,muIs,Ls,TOs = simulate_networks(prms,rX,cA)

        for nloc in range(Nori):
            μrEs[:,nloc] = np.mean(rs[:,net.C_idxs[0][nloc]],axis=-1)
            μrIs[:,nloc] = np.mean(rs[:,net.C_idxs[1][nloc]],axis=-1)
            ΣrEs[:,nloc] = np.var(rs[:,net.C_idxs[0][nloc]],axis=-1)
            ΣrIs[:,nloc] = np.var(rs[:,net.C_idxs[1][nloc]],axis=-1)
            μhEs[:,nloc] = np.mean(mus[:,net.C_idxs[0][nloc]],axis=-1)
            μhIs[:,nloc] = np.mean(mus[:,net.C_idxs[1][nloc]],axis=-1)
            ΣhEs[:,nloc] = np.var(mus[:,net.C_idxs[0][nloc]],axis=-1)
            ΣhIs[:,nloc] = np.var(mus[:,net.C_idxs[1][nloc]],axis=-1)
            balEs[:,nloc] = np.mean(np.abs(mus[:,net.C_idxs[0][nloc]])/muEs[:,net.C_idxs[0][nloc]],axis=-1)
            balIs[:,nloc] = np.mean(np.abs(mus[:,net.C_idxs[1][nloc]])/muEs[:,net.C_idxs[1][nloc]],axis=-1)
        Lexps[:] = Ls
        timeouts[:] = TOs

        seed_mask = np.logical_not(timeouts)
        vsm_mask = net.get_oriented_neurons(delta_ori=4.5)[0]
        osm_mask = net.get_oriented_neurons(delta_ori=4.5,vis_ori=90)[0]

        g1_rates = rs

        vsm_g1_means[idx,b_idx] = np.mean(g1_rates[seed_mask,:][:,vsm_mask])
        vsm_g1_stds[idx,b_idx] = np.std(g1_rates[seed_mask,:][:,vsm_mask])
        vsm_g1_bals[idx,b_idx] = np.mean(np.abs(mus[seed_mask,:][:,vsm_mask])/muEs[seed_mask,:][:,vsm_mask])

        osm_g1_means[idx,b_idx] = np.mean(g1_rates[seed_mask,:][:,osm_mask])
        osm_g1_stds[idx,b_idx] = np.std(g1_rates[seed_mask,:][:,osm_mask])
        osm_g1_bals[idx,b_idx] = np.mean(np.abs(mus[seed_mask,:][:,osm_mask])/muEs[seed_mask,:][:,osm_mask])
        
        g1_Lexps[idx,b_idx] = np.mean(Lexps[seed_mask])
        g1_timeouts[idx,b_idx] = np.mean(timeouts)

    print("Saving statistics took ",time.process_time() - start," s")
    print('')

res_dict = {}

res_dict['Js'] = Js
res_dict['betas'] = betas
res_dict['gEs'] = gEs
res_dict['gIs'] = gIs
res_dict['vsm_g1_means'] = vsm_g1_means
res_dict['vsm_g1_stds'] = vsm_g1_stds
res_dict['vsm_g1_bals'] = vsm_g1_bals
res_dict['osm_g1_means'] = osm_g1_means
res_dict['osm_g1_stds'] = osm_g1_stds
res_dict['osm_g1_bals'] = osm_g1_bals
res_dict['Lexps'] = g1_Lexps
res_dict['timeouts'] = g1_timeouts

res_file = './../results/fit_relu'
res_file = res_file + '_K_{:d}_j_{:d}_b_{:d}'.format(K,job_id,bayes_iter)

with open(res_file+'.pkl', 'wb') as handle:
    pickle.dump(res_dict,handle)
