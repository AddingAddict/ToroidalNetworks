import argparse
import os
try:
    import pickle5 as pickle
except:
    import pickle
import numpy as np
import torch
import time

import base_network as base_net
import ring_network as network
import sim_util as su
import ricciardi as ric
import integrate as integ

parser = argparse.ArgumentParser()

parser.add_argument('--c1_idx', '-c1',  help='which contrast for peak 1', type=int, default=0)
parser.add_argument('--c2_idx', '-c2',  help='which contrast for peak 2', type=int, default=0)
args = vars(parser.parse_args())
print(parser.parse_args())
c1_idx= args['c1_idx']
c2_idx= args['c2_idx']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

id = None
if id is None:
    with open('./../results/best_fit.pkl', 'rb') as handle:
        res_dict = pickle.load(handle)
elif len(id)==2:
    with open('./../results/results_ring_{:d}.pkl'.format(
            id[0]), 'rb') as handle:
        res_dict = pickle.load(handle)[id[-1]]
else:
    with open('./../results/results_ring_perturb_njob-{:d}_nrep-{:d}_ntry-{:d}.pkl'.format(
            id[0],id[1],id[2]), 'rb') as handle:
        res_dict = pickle.load(handle)[id[-1]]
prms = res_dict['prms']
CVh = res_dict['best_monk_eX']
bX = res_dict['best_monk_bX']
aXs = res_dict['best_monk_aXs']
K = prms['K']
SoriE = prms['SoriE']
SoriI = prms['SoriI']
# SoriF = prms['SoriF']
# J = prms['J']
# beta = prms['beta']
# gE = prms['gE']
# gI = prms['gI']
# hE = prms['hE']
# hI = prms['hI']
L = prms['L']
CVL = prms['CVL']

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

prms['Nori'] = Nori
prms['NE'] = NE
prms['NI'] = NI

seeds = np.arange(50)

cons = np.arange(0,4+1)/4

print('simulating contrast # '+str(c1_idx+1)+' for peak 1, contrast # '+str(c2_idx+1)+' for peak 2')
print('')
con1 = cons[c1_idx]
con2 = cons[c2_idx]

cA1 = con1*aXs[-1]/bX
cA2 = con2*aXs[-1]/bX
rX = bX

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
Lexps = np.zeros((len(seeds)))
timeouts = np.zeros((len(seeds))).astype(bool)

def simulate_networks(prms,rX,cA1,cA2,CVh):
    N = prms.get('Nori',180) * (prms.get('NE',4) + prms.get('NI',1))
    rs = np.zeros((len(seeds),N))
    mus = np.zeros((len(seeds),N))
    muEs = np.zeros((len(seeds),N))
    muIs = np.zeros((len(seeds),N))
    Ls = np.zeros((len(seeds)))
    TOs = np.zeros((len(seeds)))

    for seed_idx,seed in enumerate(seeds):
        print('simulating seed # '+str(seed_idx+1))
        print('')
        
        start = time.process_time()

        net,this_M,this_H1,this_B,this_LAS,this_EPS = su.gen_ring_disorder_tensor(seed,prms,CVh)
        this_H2 = torch.roll(this_H1,N//2).to(device)
        M = this_M.cpu().numpy()
        H = (rX*(this_B+cA1*this_H1+cA2*this_H2)*this_EPS).cpu().numpy()
        LAS = this_LAS.cpu().numpy()

        print("Generating disorder took ",time.process_time() - start," s")
        print('')

        start = time.process_time()

        base_sol,base_timeout = integ.sim_dyn_tensor(ri,T,0.0,this_M,rX*(this_B+cA1*this_H1+cA2*this_H2)*this_EPS,
                                                     this_LAS,net.C_conds[0],mult_tau=True,max_min=30)
        Ls[seed_idx] = np.max(integ.calc_lyapunov_exp_tensor(ri,T[T>=3*Nt],0.0,this_M,
                                                               rX*(this_B+cA1*this_H1+cA2*this_H2)*this_EPS,this_LAS,
                                                               net.C_conds[0],base_sol[:,T>=3*Nt],10,1*Nt,2*ri.tE,
                                                               mult_tau=True).cpu().numpy())
        rs[seed_idx] = np.mean(base_sol[:,mask_time].cpu().numpy(),-1)
        TOs[seed_idx] = base_timeout

        print("Integrating base network took ",time.process_time() - start," s")
        print('')

        start = time.process_time()

        muEs[seed_idx] = M[:,net.C_all[0]]@rs[seed_idx,net.C_all[0]] + H
        muIs[seed_idx] = M[:,net.C_all[1]]@rs[seed_idx,net.C_all[1]]
        muEs[seed_idx,net.C_all[0]] *= ri.tE
        muEs[seed_idx,net.C_all[1]] *= ri.tI
        muIs[seed_idx,net.C_all[0]] *= ri.tE
        muIs[seed_idx,net.C_all[1]] *= ri.tI
        mus[seed_idx] = muEs[seed_idx] + muIs[seed_idx]

        print("Calculating statistics took ",time.process_time() - start," s")
        print('')

    return net,rs,mus,muEs,muIs,Ls,TOs

# Simulate network where structure is removed by increasing baseline fraction
print('simulating baseline fraction network')
print('')
this_prms = prms.copy()

net,rs,mus,muEs,muIs,Ls,TOs = simulate_networks(this_prms,rX,cA1,cA2,CVh)

start = time.process_time()

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
vsm1_mask = net.get_oriented_neurons()[0]
vsm2_mask = net.get_oriented_neurons(vis_ori=90)[0]

base_rates = rs

all_base_means = np.mean(base_rates[seed_mask,:])
all_base_stds = np.std(base_rates[seed_mask,:])

vsm1_base_means = np.mean(base_rates[seed_mask,:][:,vsm1_mask])
vsm1_base_stds = np.std(base_rates[seed_mask,:][:,vsm1_mask])

vsm2_base_means = np.mean(base_rates[seed_mask,:][:,vsm2_mask])
vsm2_base_stds = np.std(base_rates[seed_mask,:][:,vsm2_mask])

print("Saving statistics took ",time.process_time() - start," s")
print('')

res_dict = {}
res_dict['prms'] = this_prms
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
res_dict['all_base_means'] = all_base_means
res_dict['all_base_stds'] = all_base_stds
res_dict['vsm1_base_means'] = vsm1_base_means
res_dict['vsm1_base_stds'] = vsm1_base_stds
res_dict['vsm2_base_means'] = vsm2_base_means
res_dict['vsm2_base_stds'] = vsm2_base_stds
res_dict['timeouts'] = timeouts

with open('./../results/vary_id_{:s}_c1_{:d}_c2_{:d}'.format(str(id),c1_idx,c2_idx)+'.pkl', 'wb') as handle:
    pickle.dump(res_dict,handle)