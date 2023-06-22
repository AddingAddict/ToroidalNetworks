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

parser.add_argument('--rX_idx', '-r',  help='which rX', type=int, default=0)
parser.add_argument('--K_idx', '-k',  help='which K', type=int, default=0)
args = vars(parser.parse_args())
print(parser.parse_args())
rX_idx= args['rX_idx']
K_idx= args['K_idx']

id = (133, 0)
with open('./../results/results_ring_'+str(id[0])+'.pkl', 'rb') as handle:
    res_dict = pickle.load(handle)[id[1]]
    prms = res_dict['prms']
    CVh = res_dict['best_monk_eX']
    bX = res_dict['best_monk_bX']
    aXs = res_dict['best_monk_aXs']
#     K = prms['K']
    SoriE = prms['SoriE']
    SoriI = prms['SoriI']
    # SoriF = prms['SoriF']
#     J = prms['J']
#     beta = prms['beta']
#     gE = prms['gE']
#     gI = prms['gI']
#     hE = prms['hE']
#     hI = prms['hI']
#     L = prms['L']
#     CVL = prms['CVL']

ri = ric.Ricciardi()
ri.set_up_nonlinearity('./phi_int')
ri.set_up_nonlinearity_tensor()

NtE = 50
Nt = NtE*ri.tE
dt = ri.tI/3
T = torch.linspace(0,4*Nt,round(4*Nt/dt)+1)
mask_time = T>(2*Nt)
T_mask = T.cpu().numpy()[mask_time][2::3]

N = 10000
Nori = 20
NE = 4*(N//Nori)//5
NI = 1*(N//Nori)//5

prms['Nori'] = Nori
prms['NE'] = NE
prms['NI'] = NI

seeds = np.arange(20)

cAs = aXs[[0,-1]]/bX
rXs = bX*10**np.arange(-4/3,1/3+0.1,1/3)
Ks = np.round(500*10**np.arange(-4/3,1/3+0.1,1/3)).astype(np.int32)

μrEs = np.zeros((2,2*Nori))
μrIs = np.zeros((2,2*Nori))
ΣrEs = np.zeros((2,2*Nori))
ΣrIs = np.zeros((2,2*Nori))
μhEs = np.zeros((2,2*Nori))
μhIs = np.zeros((2,2*Nori))
ΣhEs = np.zeros((2,2*Nori))
ΣhIs = np.zeros((2,2*Nori))
balEs = np.zeros((2,2*Nori))
balIs = np.zeros((2,2*Nori))
Mevs = np.zeros(2)

print('simulating rX # '+str(rX_idx+1))
print('')
rX = rXs[rX_idx]

print('simulating K # '+str(K_idx+1))
print('')
K = Ks[K_idx]

for cA_idx,cA in enumerate(cAs):
    print('simulating contrast # '+str(cA_idx+1))
    print('')
    K_prms = prms.copy()
    K_prms['K'] = K
    K_prms['J'] = prms['J'] / np.sqrt(K/500)

    rs = np.zeros((len(seeds),N,len(T_mask)))
    mus = np.zeros((len(seeds),N,len(T_mask)))
    muEs = np.zeros((len(seeds),N,len(T_mask)))
    muIs = np.zeros((len(seeds),N,len(T_mask)))

    for seed_idx,seed in enumerate(seeds):
        print('simulating seed # '+str(seed_idx+1))
        print('')
        
        start = time.process_time()

        net,this_M,this_H,this_B,this_LAS,this_EPS = su.gen_ring_disorder_tensor(seed,K_prms,CVh)
        M = this_M.cpu().numpy()
        H = (rX*(this_B+cA*this_H)*this_EPS).cpu().numpy()

        print("Generating disorder took ",time.process_time() - start," s")
        print('')

        start = time.process_time()

        sol,_ = integ.sim_dyn_tensor(ri,T,0.0,this_M,rX*(this_B+cA*this_H)*this_EPS,
                                          this_LAS,net.C_conds[0],mult_tau=True)
        rs[seed_idx] = sol[:,mask_time].cpu().numpy()[:,2::3]

        muEs[seed_idx] = M[:,net.C_all[0]]@rs[seed_idx,net.C_all[0],:] + H[:,None]
        muIs[seed_idx] = M[:,net.C_all[1]]@rs[seed_idx,net.C_all[1],:]
        muEs[seed_idx,net.C_all[0],:] *= ri.tE
        muEs[seed_idx,net.C_all[1],:] *= ri.tI
        muIs[seed_idx,net.C_all[0],:] *= ri.tE
        muIs[seed_idx,net.C_all[1],:] *= ri.tI
        mus[seed_idx] = muEs[seed_idx] + muIs[seed_idx]

        print("Integrating network took ",time.process_time() - start," s")
        print('')

    start = time.process_time()

    for nloc in range(Nori):
        μrEs[cA_idx] = np.mean(rs[:,net.C_idxs[0][nloc],:])
        μrIs[cA_idx] = np.mean(rs[:,net.C_idxs[1][nloc],:])
        ΣrEs[cA_idx] = np.var(rs[:,net.C_idxs[0][nloc],:])
        ΣrIs[cA_idx] = np.var(rs[:,net.C_idxs[1][nloc],:])
        μhEs[cA_idx] = np.mean(mus[:,net.C_idxs[0][nloc],:])
        μhIs[cA_idx] = np.mean(mus[:,net.C_idxs[1][nloc],:])
        ΣhEs[cA_idx] = np.var(mus[:,net.C_idxs[0][nloc],:])
        ΣhIs[cA_idx] = np.var(mus[:,net.C_idxs[1][nloc],:])
        balEs[cA_idx] = np.mean(np.abs(mus[:,net.C_idxs[0][nloc],:])/muEs[:,net.C_idxs[0][nloc],:])
        balIs[cA_idx] = np.mean(np.abs(mus[:,net.C_idxs[1][nloc],:])/muEs[:,net.C_idxs[1][nloc],:])

    gain2s = np.zeros(2*Nori)

    gain2s[:Nori] = np.mean(np.mean(ri.dphiE(mus[:,net.C_all[0],:])**2,axis=(0,-1))\
        .reshape((Nori,Nori*NE//Nori)),axis=-1)
    gain2s[Nori:] = np.mean(np.mean(ri.dphiI(mus[:,net.C_all[1],:])**2,axis=(0,-1))\
        .reshape((Nori,Nori*NI//Nori)),axis=-1)

    Mpop = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            Mpop[i,j] = np.max(np.abs(M[net.C_all[i],:][:,net.C_all[j]]))
            if j == 1:
                Mpop[i,j] *= -1

    varWs = np.zeros((2*Nori,2*Nori))

    ori_diff = base_net.make_periodic(np.abs(np.arange(Nori) -\
                                             np.arange(Nori)[:,None])*180/Nori,90)
    Ekern = base_net.apply_kernel(ori_diff,SoriE,180,180/Nori,kernel='gaussian')
    Ikern = base_net.apply_kernel(ori_diff,SoriI,180,180/Nori,kernel='gaussian')

    varWs[:Nori,:Nori] = ri.tE**2*K  *Mpop[0,0]**2*Ekern
    varWs[Nori:,:Nori] = ri.tI**2*K  *Mpop[1,0]**2*Ekern
    varWs[:Nori,Nori:] = ri.tE**2*K/4*Mpop[0,1]**2*Ikern
    varWs[Nori:,Nori:] = ri.tI**2*K/4*Mpop[1,1]**2*Ikern

    stabM = varWs@np.diag(gain2s)

    evals = np.linalg.eigvals(stabM)

    Mevs[cA_idx] = np.max(np.real(evals))

    print("Calculating statistics took ",time.process_time() - start," s")
    print('')

res_dict = {}

prms['CVh'] = CVh
res_dict['prms'] = prms

res_dict['cAs'] = cAs
res_dict['rX'] = rX
res_dict['K'] = K

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
res_dict['Mevs'] = Mevs

with open('./../results/vary_rX_{:d}_K_{:d}'.format(rX_idx,K_idx)+'.pkl', 'wb') as handle:
    pickle.dump(res_dict,handle)
