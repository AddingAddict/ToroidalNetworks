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

parser.add_argument('--rX_idx', '-r',  help='which rX', type=int, default=0)
parser.add_argument('--N_idx', '-n',  help='which N', type=int, default=0)
args = vars(parser.parse_args())
print(parser.parse_args())
rX_idx= args['rX_idx']
N_idx= args['N_idx']

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
    # J = prms['J']
    # beta = prms['beta']
    # gE = prms['gE']
    # gI = prms['gI']
    # hE = prms['hE']
    # hI = prms['hI']
    # L = prms['L']
    # CVL = prms['CVL']

ri = ric.Ricciardi()
ri.set_up_nonlinearity('./phi_int')
ri.set_up_nonlinearity_tensor()

NtE = 50
Nt = NtE*ri.tE
dt = ri.tI/3
T = torch.linspace(0,8*Nt,round(8*Nt/dt)+1)
mask_time = T>(4*Nt)
T_mask = T.cpu().numpy()[mask_time]

Nori = 20

prms['Nori'] = Nori

seeds = np.arange(20)

cAs = aXs[[0,-1]]/bX
rXs = bX*10**np.arange(-1.4,0.4+0.1,0.2)
Ns = 5*Nori*np.round(10000/(5*Nori)*10**np.arange(-1.4/3,0.4/3+0.01,0.2/3)).astype(np.int32)

# μrEs = np.zeros((2,Nori))
# μrIs = np.zeros((2,Nori))
# ΣrEs = np.zeros((2,Nori))
# ΣrIs = np.zeros((2,Nori))
# μhEs = np.zeros((2,Nori))
# μhIs = np.zeros((2,Nori))
# ΣhEs = np.zeros((2,Nori))
# ΣhIs = np.zeros((2,Nori))
# balEs = np.zeros((2,Nori))
# balIs = np.zeros((2,Nori))
# Mevs = np.zeros(2)
# Lexps = np.zeros(2)
μrEs = np.zeros((len(seeds),3,Nori))
μrIs = np.zeros((len(seeds),3,Nori))
ΣrEs = np.zeros((len(seeds),3,Nori))
ΣrIs = np.zeros((len(seeds),3,Nori))
μhEs = np.zeros((len(seeds),3,Nori))
μhIs = np.zeros((len(seeds),3,Nori))
ΣhEs = np.zeros((len(seeds),3,Nori))
ΣhIs = np.zeros((len(seeds),3,Nori))
balEs = np.zeros((len(seeds),3,Nori))
balIs = np.zeros((len(seeds),3,Nori))
Mevs = np.zeros((len(seeds),3))
Lexps = np.zeros((len(seeds),3))

print('simulating rX # '+str(rX_idx+1))
print('')
rX = rXs[rX_idx]

print('simulating N # '+str(N_idx+1))
print('')
N = Ns[N_idx]
NE = 4*(N//Nori)//5
NI = 1*(N//Nori)//5
prms['NE'] = NE
prms['NI'] = NI

def simulate_networks(prms,rX,cA,CVh):
    N = prms.get('Nori',180) * (prms.get('NE',4) + prms.get('NI',1))
    rs = np.zeros((len(seeds),N,len(T_mask)))
    mus = np.zeros((len(seeds),N,len(T_mask)))
    muEs = np.zeros((len(seeds),N,len(T_mask)))
    muIs = np.zeros((len(seeds),N,len(T_mask)))
    Ls = np.zeros((len(seeds)))
    Ms = np.zeros((len(seeds)))

    for seed_idx,seed in enumerate(seeds):
        print('simulating seed # '+str(seed_idx+1))
        print('')
        
        start = time.process_time()

        net,this_M,this_H,this_B,this_LAS,this_EPS = su.gen_ring_disorder_tensor(seed,prms,CVh)
        M = this_M.cpu().numpy()
        H = (rX*(this_B+cA*this_H)*this_EPS).cpu().numpy()

        print("Generating disorder took ",time.process_time() - start," s")
        print('')

        start = time.process_time()

        sol,_ = integ.sim_dyn_tensor(ri,T,0.0,this_M,rX*(this_B+cA*this_H)*this_EPS,
                                     this_LAS,net.C_conds[0],mult_tau=True)
        Ls[seed_idx] = np.max(integ.calc_lyapunov_exp_tensor(ri,T[T>=4*Nt],0.0,this_M,
                                                             rX*(this_B+cA*this_H)*this_EPS,this_LAS,
                                                             net.C_conds[0],sol[:,T>=4*Nt],10,2*Nt,2*ri.tE,
                                                             mult_tau=True).cpu().numpy())
        rs[seed_idx] = sol[:,mask_time].cpu().numpy()

        print("Integrating network took ",time.process_time() - start," s")
        print('')

        start = time.process_time()

        muEs[seed_idx] = M[:,net.C_all[0]]@rs[seed_idx,net.C_all[0],:] + H[:,None]
        muIs[seed_idx] = M[:,net.C_all[1]]@rs[seed_idx,net.C_all[1],:]
        muEs[seed_idx,net.C_all[0],:] *= ri.tE
        muEs[seed_idx,net.C_all[1],:] *= ri.tI
        muIs[seed_idx,net.C_all[0],:] *= ri.tE
        muIs[seed_idx,net.C_all[1],:] *= ri.tI
        mus[seed_idx] = muEs[seed_idx] + muIs[seed_idx]

        gain2s = np.zeros(2*Nori)

        gain2s[:Nori] = np.mean(np.mean(ri.dphiE(mus[seed_idx,net.C_all[0],:])**2,axis=-1)\
            .reshape((Nori,Nori*NE//Nori)),axis=-1)
        gain2s[Nori:] = np.mean(np.mean(ri.dphiI(mus[seed_idx,net.C_all[1],:])**2,axis=-1)\
            .reshape((Nori,Nori*NI//Nori)),axis=-1)

        Mpop = np.zeros((2,2))
        for i in range(2):
            for j in range(2):
                Mpop[i,j] = np.max(np.abs(M[net.C_all[i],:][:,net.C_all[j]]))
                if j == 1:
                    Mpop[i,j] *= -1

        varWs = np.zeros((2*Nori,2*Nori))

        if prms.get('basefrac',0)==1:
            Ekern = 1/Nori
            Ikern = 1/Nori
        else:
            ori_diff = base_net.make_periodic(np.abs(np.arange(Nori) -\
                                                     np.arange(Nori)[:,None])*180/Nori,90)
            Ekern = prms.get('basefrac',0) + (1-prms.get('basefrac',0))*\
                base_net.apply_kernel(ori_diff,SoriE,180,180/Nori,kernel='gaussian')
            Ikern = prms.get('basefrac',0) + (1-prms.get('basefrac',0))*\
                base_net.apply_kernel(ori_diff,SoriI,180,180/Nori,kernel='gaussian')

        varWs[:Nori,:Nori] = ri.tE**2*prms['K']  *Mpop[0,0]**2*Ekern
        varWs[Nori:,:Nori] = ri.tI**2*prms['K']  *Mpop[1,0]**2*Ekern
        varWs[:Nori,Nori:] = ri.tE**2*prms['K']/4*Mpop[0,1]**2*Ikern
        varWs[Nori:,Nori:] = ri.tI**2*prms['K']/4*Mpop[1,1]**2*Ikern

        stabM = varWs@np.diag(gain2s)

        evals = np.linalg.eigvals(stabM)

        Ms[seed_idx] = np.max(np.real(evals))

        print("Calculating statistics took ",time.process_time() - start," s")
        print('')

    return net,rs,mus,muEs,muIs,Ls,Ms

# Simulate zero contrast network, but with all-to-all connectivity\
print('simulating all-to-all network')
print('')
N_prms = prms.copy()
N_prms['N'] = N
N_prms['basefrac'] = 1

net,rs,mus,muEs,muIs,Ls,Ms = simulate_networks(N_prms,rX,cAs[0],CVh)
rs = np.mean(rs,-1)
mus = np.mean(mus,-1)
muEs = np.mean(muEs,-1)
muIs = np.mean(muIs,-1)

start = time.process_time()

for nloc in range(Nori):
    μrEs[:,0,nloc] = np.mean(rs[:,net.C_idxs[0][nloc]],axis=-1)
    μrIs[:,0,nloc] = np.mean(rs[:,net.C_idxs[1][nloc]],axis=-1)
    ΣrEs[:,0,nloc] = np.var(rs[:,net.C_idxs[0][nloc]],axis=-1)
    ΣrIs[:,0,nloc] = np.var(rs[:,net.C_idxs[1][nloc]],axis=-1)
    μhEs[:,0,nloc] = np.mean(mus[:,net.C_idxs[0][nloc]],axis=-1)
    μhIs[:,0,nloc] = np.mean(mus[:,net.C_idxs[1][nloc]],axis=-1)
    ΣhEs[:,0,nloc] = np.var(mus[:,net.C_idxs[0][nloc]],axis=-1)
    ΣhIs[:,0,nloc] = np.var(mus[:,net.C_idxs[1][nloc]],axis=-1)
    balEs[:,0,nloc] = np.mean(np.abs(mus[:,net.C_idxs[0][nloc]])/muEs[:,net.C_idxs[0][nloc]],axis=-1)
    balIs[:,0,nloc] = np.mean(np.abs(mus[:,net.C_idxs[1][nloc]])/muEs[:,net.C_idxs[1][nloc]],axis=-1)
Lexps[:,0] = Ls
Mevs[:,0] = Ms

print("Saving statistics took ",time.process_time() - start," s")
print('')

for cA_idx,cA in enumerate(cAs):
    print('simulating contrast # '+str(cA_idx+1))
    print('')
    N_prms = prms.copy()
    N_prms['N'] = N

    net,rs,mus,muEs,muIs,Ls,Ms = simulate_networks(N_prms,rX,cA,CVh)
    rs = np.mean(rs,-1)
    mus = np.mean(mus,-1)
    muEs = np.mean(muEs,-1)
    muIs = np.mean(muIs,-1)

    start = time.process_time()

    for nloc in range(Nori):
        μrEs[:,cA_idx+1,nloc] = np.mean(rs[:,net.C_idxs[0][nloc]],axis=-1)
        μrIs[:,cA_idx+1,nloc] = np.mean(rs[:,net.C_idxs[1][nloc]],axis=-1)
        ΣrEs[:,cA_idx+1,nloc] = np.var(rs[:,net.C_idxs[0][nloc]],axis=-1)
        ΣrIs[:,cA_idx+1,nloc] = np.var(rs[:,net.C_idxs[1][nloc]],axis=-1)
        μhEs[:,cA_idx+1,nloc] = np.mean(mus[:,net.C_idxs[0][nloc]],axis=-1)
        μhIs[:,cA_idx+1,nloc] = np.mean(mus[:,net.C_idxs[1][nloc]],axis=-1)
        ΣhEs[:,cA_idx+1,nloc] = np.var(mus[:,net.C_idxs[0][nloc]],axis=-1)
        ΣhIs[:,cA_idx+1,nloc] = np.var(mus[:,net.C_idxs[1][nloc]],axis=-1)
        balEs[:,cA_idx+1,nloc] = np.mean(np.abs(mus[:,net.C_idxs[0][nloc]])/muEs[:,net.C_idxs[0][nloc]],axis=-1)
        balIs[:,cA_idx+1,nloc] = np.mean(np.abs(mus[:,net.C_idxs[1][nloc]])/muEs[:,net.C_idxs[1][nloc]],axis=-1)
    Lexps[:,cA_idx+1] = Ls
    Mevs[:,cA_idx+1] = Ms

    print("Saving statistics took ",time.process_time() - start," s")
    print('')

res_dict = {}

prms['CVh'] = CVh
res_dict['prms'] = prms

res_dict['cAs'] = cAs
res_dict['rX'] = rX
res_dict['N'] = N

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
res_dict['Lexps'] = Lexps

with open('./../results/vary_rX_{:d}_N_{:d}'.format(rX_idx,N_idx)+'.pkl', 'wb') as handle:
    pickle.dump(res_dict,handle)
