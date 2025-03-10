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
import ricciardi as ric
import integrate as integ

parser = argparse.ArgumentParser()

parser.add_argument('--c_idx', '-c',  help='which contrast', type=int, default=0)
parser.add_argument('--SoriE_mult', '-SoriEm',  help='multiplier for SoriE', type=float, default=1.0)
parser.add_argument('--SoriI_mult', '-SoriIm',  help='multiplier for SoriI', type=float, default=1.0)
parser.add_argument('--SoriF_mult', '-SoriFm',  help='multiplier for SoriF', type=float, default=1.0)
parser.add_argument('--CVh_mult', '-CVhm',  help='multiplier for CVh', type=float, default=1.0)
parser.add_argument('--J_mult', '-Jm',  help='multiplier for J', type=float, default=1.0)
parser.add_argument('--beta_mult', '-betam',  help='multiplier for beta', type=float, default=1.0)
parser.add_argument('--gE_mult', '-gEm',  help='multiplier for gE', type=float, default=1.0)
parser.add_argument('--gI_mult', '-gIm',  help='multiplier for gI', type=float, default=1.0)
parser.add_argument('--hE_mult', '-hEm',  help='multiplier for hE', type=float, default=1.0)
parser.add_argument('--hI_mult', '-hIm',  help='multiplier for hI', type=float, default=1.0)
parser.add_argument('--L_mult', '-Lm',  help='multiplier for L', type=float, default=1.0)
parser.add_argument('--CVL_mult', '-CVLm',  help='multiplier for CVL', type=float, default=1.0)
args = vars(parser.parse_args())
print(parser.parse_args())
c_idx= args['c_idx']
SoriE_mult= args['SoriE_mult']
SoriI_mult= args['SoriI_mult']
SoriF_mult= args['SoriF_mult']
CVh_mult= args['CVh_mult']
J_mult= args['J_mult']
beta_mult= args['beta_mult']
gE_mult= args['gE_mult']
gI_mult= args['gI_mult']
hE_mult= args['hE_mult']
hI_mult= args['hI_mult']
L_mult= args['L_mult']
CVL_mult= args['CVL_mult']

id = None
# id = (133,0,79,3)
if id is None:
    with open('./../results/best_fit.pkl', 'rb') as handle:
        res_dict = pickle.load(handle)
elif len(id)==1:
    with open('./../results/refit_candidate_prms_{:d}.pkl'.format(
            id[0]), 'rb') as handle:
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
# SoriE = prms['SoriE']
# SoriI = prms['SoriI']
# SoriF = prms['SoriF']
# J = prms['J']
# beta = prms['beta']
# gE = prms['gE']
# gI = prms['gI']
# hE = prms['hE']
# hI = prms['hI']
# L = prms['L']
# CVL = prms['CVL']

prms['rho'] = 0.2

CVh = CVh*CVh_mult
prms['SoriE'] = prms['SoriE']*SoriE_mult
prms['SoriI'] = prms['SoriI']*SoriI_mult
prms['SoriF'] = prms['SoriF']*SoriF_mult
prms['J'] = prms['J']*J_mult
prms['beta'] = prms['beta']*beta_mult
prms['gE'] = prms['gE']*gE_mult
prms['gI'] = prms['gI']*gI_mult
prms['hE'] = prms['hE']*hE_mult
prms['hI'] = prms['hI']*hI_mult
prms['L'] = prms['L']*L_mult
prms['CVL'] = prms['CVL']*CVL_mult

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
Nori = 2000
NE = 4*(N//Nori)//5
NI = 1*(N//Nori)//5

prms['Nori'] = Nori
prms['NE'] = NE
prms['NI'] = NI

seeds = np.arange(50)

print('simulating contrast # '+str(c_idx+1))
print('')
aXs = np.concatenate([aXs,aXs[-1]*np.arange(1.0+0.2,2.0+0.2,0.2)])
aX = aXs[c_idx]

cA = aX/bX
rX = bX

def simulate_networks(prms,rX,cA,CVh):
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

        net,this_M,this_H,this_B,this_LAS,this_EPS = su.gen_ring_disorder_tensor(seed,prms,CVh)
        M = this_M.cpu().numpy()
        H = (rX*(this_B+cA*this_H)*this_EPS).cpu().numpy()
        LAS = this_LAS.cpu().numpy()

        print("Generating disorder took ",time.process_time() - start," s")
        print('')

        start = time.process_time()

        base_sol,base_timeout = integ.sim_dyn_tensor(ri,T,0.0,this_M,rX*(this_B+cA*this_H)*this_EPS,
                                                     this_LAS,net.C_conds[0],mult_tau=True,max_min=30)
        Ls[seed_idx,0] = np.max(integ.calc_lyapunov_exp_tensor(ri,T[T>=4*Nt],0.0,this_M,
                                                               rX*(this_B+cA*this_H)*this_EPS,this_LAS,
                                                               net.C_conds[0],base_sol[:,T>=4*Nt],10,2*Nt,2*ri.tE,
                                                               mult_tau=True).cpu().numpy())
        rs[seed_idx,0] = np.mean(base_sol[:,mask_time].cpu().numpy(),-1)
        TOs[seed_idx,0] = base_timeout

        print("Integrating base network took ",time.process_time() - start," s")
        print('')

        start = time.process_time()
        
        opto_sol,opto_timeout = integ.sim_dyn_tensor(ri,T,1.0,this_M,rX*(this_B+cA*this_H)*this_EPS,
                                                     this_LAS,net.C_conds[0],mult_tau=True,max_min=30)
        Ls[seed_idx,1] = np.max(integ.calc_lyapunov_exp_tensor(ri,T[T>=4*Nt],1.0,this_M,
                                                               rX*(this_B+cA*this_H)*this_EPS,this_LAS,
                                                               net.C_conds[0],opto_sol[:,T>=4*Nt],10,2*Nt,2*ri.tE,
                                                               mult_tau=True).cpu().numpy())
        rs[seed_idx,1] = np.mean(opto_sol[:,mask_time].cpu().numpy(),-1)
        TOs[seed_idx,1] = opto_timeout

        print("Integrating opto network took ",time.process_time() - start," s")
        print('')

        start = time.process_time()

        muXs[seed_idx,0] = H
        muEs[seed_idx,0] = M[:,net.C_all[0]]@rs[seed_idx,0,net.C_all[0]] + H
        muIs[seed_idx,0] = M[:,net.C_all[1]]@rs[seed_idx,0,net.C_all[1]]
        muXs[seed_idx,0,net.C_all[0]] *= ri.tE
        muXs[seed_idx,0,net.C_all[1]] *= ri.tI
        muEs[seed_idx,0,net.C_all[0]] *= ri.tE
        muEs[seed_idx,0,net.C_all[1]] *= ri.tI
        muIs[seed_idx,0,net.C_all[0]] *= ri.tE
        muIs[seed_idx,0,net.C_all[1]] *= ri.tI
        mus[seed_idx,0] = muEs[seed_idx,0] + muIs[seed_idx,0]

        muXs[seed_idx,1] = H
        muEs[seed_idx,1] = M[:,net.C_all[0]]@rs[seed_idx,1,net.C_all[0]] + H
        muIs[seed_idx,1] = M[:,net.C_all[1]]@rs[seed_idx,1,net.C_all[1]]
        muXs[seed_idx,1,net.C_all[0]] *= ri.tE
        muXs[seed_idx,1,net.C_all[1]] *= ri.tI
        muEs[seed_idx,1,net.C_all[0]] *= ri.tE
        muEs[seed_idx,1,net.C_all[1]] *= ri.tI
        muIs[seed_idx,1,net.C_all[0]] *= ri.tE
        muIs[seed_idx,1,net.C_all[1]] *= ri.tI
        muXs[seed_idx,1] = muXs[seed_idx,1] + LAS
        muEs[seed_idx,1] = muEs[seed_idx,1] + LAS
        mus[seed_idx,1] = muEs[seed_idx,1] + muIs[seed_idx,1]

        print("Calculating statistics took ",time.process_time() - start," s")
        print('')

    return net,rs,mus,muXs,muEs,muIs,Ls,TOs

# Simulate network where structure is removed by increasing baseline fraction
print('simulating baseline fraction network')
print('')

Nbigori = 20

μrEs = np.zeros((len(seeds),3,Nbigori))
μrIs = np.zeros((len(seeds),3,Nbigori))
ΣrEs = np.zeros((len(seeds),4,Nbigori))
ΣrIs = np.zeros((len(seeds),4,Nbigori))
μhEs = np.zeros((len(seeds),3,Nbigori))
μhIs = np.zeros((len(seeds),3,Nbigori))
ΣhEs = np.zeros((len(seeds),4,Nbigori))
ΣhIs = np.zeros((len(seeds),4,Nbigori))
balEs = np.zeros((len(seeds),2,Nbigori))
balIs = np.zeros((len(seeds),2,Nbigori))
Lexps = np.zeros((len(seeds),2))
timeouts = np.zeros((len(seeds),2)).astype(bool)
# μrEs = np.zeros((2,len(seeds),3,Nbigori))
# μrIs = np.zeros((2,len(seeds),3,Nbigori))
# ΣrEs = np.zeros((2,len(seeds),4,Nbigori))
# ΣrIs = np.zeros((2,len(seeds),4,Nbigori))
# μhEs = np.zeros((2,len(seeds),3,Nbigori))
# μhIs = np.zeros((2,len(seeds),3,Nbigori))
# ΣhEs = np.zeros((2,len(seeds),4,Nbigori))
# ΣhIs = np.zeros((2,len(seeds),4,Nbigori))
# balEs = np.zeros((2,len(seeds),2,Nbigori))
# balIs = np.zeros((2,len(seeds),2,Nbigori))
# Lexps = np.zeros((2,len(seeds),2))
# timeouts = np.zeros((2,len(seeds),2)).astype(bool)
# all_base_means = np.zeros(2)
# all_base_stds = np.zeros(2)
# all_opto_means = np.zeros(2)
# all_opto_stds = np.zeros(2)
# all_diff_means = np.zeros(2)
# all_diff_stds = np.zeros(2)
# all_norm_covs = np.zeros(2)
# vsm_base_means = np.zeros(2)
# vsm_base_stds = np.zeros(2)
# vsm_opto_means = np.zeros(2)
# vsm_opto_stds = np.zeros(2)
# vsm_diff_means = np.zeros(2)
# vsm_diff_stds = np.zeros(2)
# vsm_norm_covs = np.zeros(2)

net,rs,mus,muXs,muEs,muIs,Ls,TOs = simulate_networks(prms,rX,cA,CVh)

start = time.process_time()

C_big_idxs = [[None]*Nbigori,[None]*Nbigori]
Nrange = np.arange(N)

for nloc in range(Nori):
    nbigori = nloc//(Nori//Nbigori)
    for i in range(2):
        if C_big_idxs[i][nbigori] is None:
            C_big_idxs[i][nbigori] = (Nrange[net.C_idxs[i][nloc]] - (N//Nbigori//2)) % N
        else:
            C_big_idxs[i][nbigori] = np.append(C_big_idxs[i][nbigori],(Nrange[net.C_idxs[i][nloc]] - (N//Nbigori//2)) % N)

for nloc in range(Nbigori):
    μrEs[:,:2,nloc] = np.mean(rs[:,:,C_big_idxs[0][nloc]],axis=-1)
    μrIs[:,:2,nloc] = np.mean(rs[:,:,C_big_idxs[1][nloc]],axis=-1)
    ΣrEs[:,:2,nloc] = np.var(rs[:,:,C_big_idxs[0][nloc]],axis=-1)
    ΣrIs[:,:2,nloc] = np.var(rs[:,:,C_big_idxs[1][nloc]],axis=-1)
    μhEs[:,:2,nloc] = np.mean(mus[:,:,C_big_idxs[0][nloc]],axis=-1)
    μhIs[:,:2,nloc] = np.mean(mus[:,:,C_big_idxs[1][nloc]],axis=-1)
    ΣhEs[:,:2,nloc] = np.var(mus[:,:,C_big_idxs[0][nloc]],axis=-1)
    ΣhIs[:,:2,nloc] = np.var(mus[:,:,C_big_idxs[1][nloc]],axis=-1)
    balEs[:,:,nloc] = np.mean(np.abs(mus[:,:,C_big_idxs[0][nloc]])/muEs[:,:,C_big_idxs[0][nloc]],axis=-1)
    balIs[:,:,nloc] = np.mean(np.abs(mus[:,:,C_big_idxs[1][nloc]])/muEs[:,:,C_big_idxs[1][nloc]],axis=-1)

    μrEs[:,2,nloc] = np.mean(rs[:,1,C_big_idxs[0][nloc]]-rs[:,0,C_big_idxs[0][nloc]],axis=-1)
    μrIs[:,2,nloc] = np.mean(rs[:,1,C_big_idxs[1][nloc]]-rs[:,0,C_big_idxs[1][nloc]],axis=-1)
    ΣrEs[:,2,nloc] = np.var(rs[:,1,C_big_idxs[0][nloc]]-rs[:,0,C_big_idxs[0][nloc]],axis=-1)
    ΣrIs[:,2,nloc] = np.var(rs[:,1,C_big_idxs[1][nloc]]-rs[:,0,C_big_idxs[1][nloc]],axis=-1)
    μhEs[:,2,nloc] = np.mean(mus[:,1,C_big_idxs[0][nloc]]-mus[:,0,C_big_idxs[0][nloc]],axis=-1)
    μhIs[:,2,nloc] = np.mean(mus[:,1,C_big_idxs[1][nloc]]-mus[:,0,C_big_idxs[1][nloc]],axis=-1)
    ΣhEs[:,2,nloc] = np.var(mus[:,1,C_big_idxs[0][nloc]]-mus[:,0,C_big_idxs[0][nloc]],axis=-1)
    ΣhIs[:,2,nloc] = np.var(mus[:,1,C_big_idxs[1][nloc]]-mus[:,0,C_big_idxs[1][nloc]],axis=-1)

    for seed_idx in range(len(seeds)):
        ΣrEs[seed_idx,3,nloc] = np.cov(rs[seed_idx,0,C_big_idxs[0][nloc]],
                                    rs[seed_idx,1,C_big_idxs[0][nloc]]-rs[seed_idx,0,C_big_idxs[0][nloc]])[0,1]
        ΣrIs[seed_idx,3,nloc] = np.cov(rs[seed_idx,0,C_big_idxs[1][nloc]],
                                    rs[seed_idx,1,C_big_idxs[1][nloc]]-rs[seed_idx,0,C_big_idxs[1][nloc]])[0,1]
Lexps[:,:] = Ls
timeouts[:,:] = TOs

seed_mask = np.logical_not(np.any(timeouts,axis=-1))
vsm_mask = net.get_oriented_neurons(delta_ori=4.5)[0]
osm_mask = net.get_oriented_neurons(delta_ori=4.5,vis_ori=90)[0]

base_rates = rs[:,0,:]
opto_rates = rs[:,1,:]
diff_rates = opto_rates - base_rates

all_base_means = np.mean(base_rates[seed_mask,:])
all_base_stds = np.std(base_rates[seed_mask,:])
all_opto_means = np.mean(opto_rates[seed_mask,:])
all_opto_stds = np.std(opto_rates[seed_mask,:])
all_diff_means = np.mean(diff_rates[seed_mask,:])
all_diff_stds = np.std(diff_rates[seed_mask,:])
all_norm_covs = np.cov(base_rates[seed_mask,:].flatten(),
    diff_rates[seed_mask,:].flatten())[0,1] / all_diff_stds**2
all_bals = np.mean(np.abs(mus[seed_mask,0,:])/muEs[seed_mask,0,:])
all_oves = np.mean((muXs[seed_mask,1,:]-muXs[seed_mask,0,:])/muEs[seed_mask,0,:])
all_ovxs = np.mean(muXs[seed_mask,1,:]/muXs[seed_mask,0,:]-1)

vsm_base_means = np.mean(base_rates[seed_mask,:][:,vsm_mask])
vsm_base_stds = np.std(base_rates[seed_mask,:][:,vsm_mask])
vsm_opto_means = np.mean(opto_rates[seed_mask,:][:,vsm_mask])
vsm_opto_stds = np.std(opto_rates[seed_mask,:][:,vsm_mask])
vsm_diff_means = np.mean(diff_rates[seed_mask,:][:,vsm_mask])
vsm_diff_stds = np.std(diff_rates[seed_mask,:][:,vsm_mask])
vsm_norm_covs = np.cov(base_rates[seed_mask,:][:,vsm_mask].flatten(),
    diff_rates[seed_mask,:][:,vsm_mask].flatten())[0,1] / vsm_diff_stds**2
vsm_bals = np.mean(np.abs(mus[seed_mask,0,:][:,vsm_mask])/muEs[seed_mask,0,:][:,vsm_mask])
vsm_oves = np.mean((muXs[seed_mask,1,:][:,vsm_mask]-muXs[seed_mask,0,:][:,vsm_mask])/muEs[seed_mask,0,:][:,vsm_mask])
vsm_ovxs = np.mean(muXs[seed_mask,1,:][:,vsm_mask]/muXs[seed_mask,0,:][:,vsm_mask]-1)

osm_base_means = np.mean(base_rates[seed_mask,:][:,osm_mask])
osm_base_stds = np.std(base_rates[seed_mask,:][:,osm_mask])
osm_opto_means = np.mean(opto_rates[seed_mask,:][:,osm_mask])
osm_opto_stds = np.std(opto_rates[seed_mask,:][:,osm_mask])
osm_diff_means = np.mean(diff_rates[seed_mask,:][:,osm_mask])
osm_diff_stds = np.std(diff_rates[seed_mask,:][:,osm_mask])
osm_norm_covs = np.cov(base_rates[seed_mask,:][:,osm_mask].flatten(),
    diff_rates[seed_mask,:][:,osm_mask].flatten())[0,1] / osm_diff_stds**2
osm_bals = np.mean(np.abs(mus[seed_mask,0,:][:,osm_mask])/muEs[seed_mask,0,:][:,osm_mask])
osm_oves = np.mean((muXs[seed_mask,1,:][:,osm_mask]-muXs[seed_mask,0,:][:,osm_mask])/muEs[seed_mask,0,:][:,osm_mask])
osm_ovxs = np.mean(muXs[seed_mask,1,:][:,osm_mask]/muXs[seed_mask,0,:][:,osm_mask]-1)

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
res_dict['all_base_means'] = all_base_means
res_dict['all_base_stds'] = all_base_stds
res_dict['all_opto_means'] = all_opto_means
res_dict['all_opto_stds'] = all_opto_stds
res_dict['all_diff_means'] = all_diff_means
res_dict['all_diff_stds'] = all_diff_stds
res_dict['all_norm_covs'] = all_norm_covs
res_dict['all_bals'] = all_bals
res_dict['all_oves'] = all_oves
res_dict['all_ovxs'] = all_ovxs
res_dict['vsm_base_means'] = vsm_base_means
res_dict['vsm_base_stds'] = vsm_base_stds
res_dict['vsm_opto_means'] = vsm_opto_means
res_dict['vsm_opto_stds'] = vsm_opto_stds
res_dict['vsm_diff_means'] = vsm_diff_means
res_dict['vsm_diff_stds'] = vsm_diff_stds
res_dict['vsm_norm_covs'] = vsm_norm_covs
res_dict['vsm_bals'] = vsm_bals
res_dict['vsm_oves'] = vsm_oves
res_dict['vsm_ovxs'] = vsm_ovxs
res_dict['osm_base_means'] = osm_base_means
res_dict['osm_base_stds'] = osm_base_stds
res_dict['osm_opto_means'] = osm_opto_means
res_dict['osm_opto_stds'] = osm_opto_stds
res_dict['osm_diff_means'] = osm_diff_means
res_dict['osm_diff_stds'] = osm_diff_stds
res_dict['osm_norm_covs'] = osm_norm_covs
res_dict['osm_bals'] = osm_bals
res_dict['osm_oves'] = osm_oves
res_dict['osm_ovxs'] = osm_ovxs
res_dict['timeouts'] = timeouts

res_file = './../results/corr_net_id_{:s}'.format(str(id))
if not np.isclose(SoriE_mult,1.0):
    res_file = res_file + '_SoriEx{:.2f}'.format(SoriE_mult)
if not np.isclose(SoriI_mult,1.0):
    res_file = res_file + '_SoriIx{:.2f}'.format(SoriI_mult)
if not np.isclose(SoriF_mult,1.0):
    res_file = res_file + '_SoriFx{:.2f}'.format(SoriF_mult)
if not np.isclose(CVh_mult,1.0):
    res_file = res_file + '_CVhx{:.2f}'.format(CVh_mult)
if not np.isclose(J_mult,1.0):
    res_file = res_file + '_Jx{:.2f}'.format(J_mult)
if not np.isclose(beta_mult,1.0):
    res_file = res_file + '_betax{:.2f}'.format(beta_mult)
if not np.isclose(gE_mult,1.0):
    res_file = res_file + '_gEx{:.2f}'.format(gE_mult)
if not np.isclose(gI_mult,1.0):
    res_file = res_file + '_gIx{:.2f}'.format(gI_mult)
if not np.isclose(hE_mult,1.0):
    res_file = res_file + '_hEx{:.2f}'.format(hE_mult)
if not np.isclose(hI_mult,1.0):
    res_file = res_file + '_hIx{:.2f}'.format(hI_mult)
if not np.isclose(L_mult,1.0):
    res_file = res_file + '_Lx{:.2f}'.format(L_mult)
if not np.isclose(CVL_mult,1.0):
    res_file = res_file + '_CVLx{:.2f}'.format(CVL_mult)
res_file = res_file + '_c_{:d}'.format(c_idx)

with open(res_file+'.pkl', 'wb') as handle:
    pickle.dump(res_dict,handle)
