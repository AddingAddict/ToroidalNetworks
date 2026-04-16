import argparse
import ast
import os
try:
    import pickle5 as pickle
except:
    import pickle
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
import torch
import time

import base_network as base_net
import ring_network as network
import sim_util as su
import ricciardi as ric
import integrate as integ

parser = argparse.ArgumentParser()

parser.add_argument('--j_idx', '-j',  help='which fit model to use', type=int, default=0)
args = vars(parser.parse_args())
print(parser.parse_args())
j_idx= args['j_idx']

with open('./../notebooks/best_fits.pkl', 'rb') as handle:
    theta = pickle.load(handle)[j_idx]

def get_J(theta):
    '''
    theta[:,0] = det(J)/(|Jei| * |Jie|) = 1 - (|Jee| * |Jii|) / (|Jei| * |Jie|)
    theta[:,1] = (|Jee|-|Jii|)/(|Jei| + |Jie|)
    theta[:,2] = (log10[|Jei|] + log10[|Jie|]) / 2
    theta[:,3] = (log10[|Jei|] - log10[|Jie|]) / 2
    
    returns: [Jee,Jei,Jie,Jii]
    '''
    Jei = -10**(theta[2] + theta[3])
    Jie =  10**(theta[2] - theta[3])
    Jee_m_Jii = (-Jei + Jie) * theta[1]
    Jee_p_Jii_2 = 4*((theta[0] - 1) * Jei*Jie) + Jee_m_Jii**2
    Jee = 0.5*(Jee_m_Jii + torch.sqrt(Jee_p_Jii_2))
    Jii = -(Jee - Jee_m_Jii)
    
    return Jee,Jei,Jie,Jii

Jee,Jei,Jie,Jii = get_J(theta)
Jee,Jei,Jie,Jii = Jee.item(),-Jei.item(),Jie.item(),-Jii.item()

exact_bE = theta[4].item()
exact_bI = theta[5].item()
L = theta[6].item()
CVL = 10**theta[7].item()
exact_hE = theta[8].item()
exact_hI = theta[9].item()
SoriE = theta[10].item()
SoriI = theta[11].item()
SoriF = theta[12].item()

K = 500

J = Jee
beta = Jee / Jie
gE = Jei / Jee
gI = Jii / Jie
bE = exact_bE / (J*K)
bI = exact_bI / (J*K) * beta
hE = exact_hE / (J*K)
hI = exact_hI / (J*K) * beta

prms = {
    'K': K,
    'SoriE': SoriE,
    'SoriI': SoriI,
    'SoriF': SoriF,
    'J': J,
    'beta': beta,
    'gE': gE,
    'gI': gI,
    'bE': bE,
    'bI': bI,
    'hE': hE,
    'hI': hI,
    'L': L,
    'CVL': CVL,
    'Nori': 20,
    'NE': 400,
    'NI': 100
}

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

seeds = np.arange(20)
aXs = np.linspace(0,1.5,11)

def simulate_networks(prms,aX):
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

        net,this_M,this_H,this_B,this_LAS,_ = su.gen_ring_disorder_tensor(seed,prms,0,diff_base=True)
        M = this_M.cpu().numpy()
        H = (this_B+aX*this_H).cpu().numpy()
        LAS = this_LAS.cpu().numpy()

        print("Generating disorder took ",time.process_time() - start," s")
        print('')

        start = time.process_time()

        base_sol,base_timeout = integ.sim_dyn_tensor(ri,T,0.0,this_M,this_B+aX*this_H,
                                                     this_LAS,net.C_conds[0],mult_tau=True,max_min=30)
        Ls[seed_idx,0] = np.max(integ.calc_lyapunov_exp_tensor(ri,T[T>=4*Nt],0.0,this_M,
                                                               this_B+aX*this_H,this_LAS,
                                                               net.C_conds[0],base_sol[:,T>=4*Nt],10,2*Nt,2*ri.tE,
                                                               mult_tau=True).cpu().numpy())
        rs[seed_idx,0] = np.mean(base_sol[:,mask_time].cpu().numpy(),-1)
        TOs[seed_idx,0] = base_timeout

        print("Integrating base network took ",time.process_time() - start," s")
        print('')

        start = time.process_time()
        
        opto_sol,opto_timeout = integ.sim_dyn_tensor(ri,T,1.0,this_M,this_B+aX*this_H,
                                                     this_LAS,net.C_conds[0],mult_tau=True,max_min=30)
        Ls[seed_idx,1] = np.max(integ.calc_lyapunov_exp_tensor(ri,T[T>=4*Nt],1.0,this_M,
                                                               this_B+aX*this_H,this_LAS,
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

means = np.zeros((len(aXs),3))
stds = np.zeros((len(aXs),3))

for aX_idx,aX in enumerate(aXs):
    start = time.process_time()
    
    net,rs,mus,muXs,muEs,muIs,Ls,TOs = simulate_networks(prms,aX)
    seed_mask = np.logical_not(np.any(TOs,axis=-1))
    if aX_idx == 0:
        vsm_mask = np.arange(N)
    else:
        vsm_mask = net.get_oriented_neurons(delta_ori=4.5)[0]
    means[aX_idx,0] = np.mean(rs[seed_mask,0,:][:,vsm_mask])
    stds[aX_idx,0] = np.std(rs[seed_mask,0,:][:,vsm_mask])
    means[aX_idx,1] = np.mean(rs[seed_mask,1,:][:,vsm_mask])
    stds[aX_idx,1] = np.std(rs[seed_mask,1,:][:,vsm_mask])
    means[aX_idx,2] = np.mean(rs[seed_mask,1,:][:,vsm_mask] - rs[seed_mask,0,:][:,vsm_mask])
    stds[aX_idx,2] = np.std(rs[seed_mask,1,:][:,vsm_mask] - rs[seed_mask,0,:][:,vsm_mask])
    
    print("Simulating networks took ",time.process_time() - start," s")
    
monk_nc = 5

monk_base_means =       np.array([43.32, 54.76, 64.54, 70.97, 72.69])
monk_base_stds =        np.array([32.41, 38.93, 42.76, 45.17, 48.61])
monk_opto_means =       np.array([44.93, 53.36, 60.46, 64.09, 68.87])
monk_opto_stds =        np.array([42.87, 45.13, 49.31, 47.53, 52.24])
monk_diff_means =       np.array([ 1.61, -1.41, -4.08, -6.88, -3.82])
monk_diff_stds =        np.array([42.48, 42.24, 45.43, 41.78, 41.71])

monk_base_means_err =   np.array([ 4.49,  5.38,  5.90,  6.22,  6.69])
monk_base_stds_err =    np.array([ 3.67,  4.48,  5.10,  5.61,  5.32])
monk_opto_means_err =   np.array([ 5.86,  6.16,  6.74,  6.50,  7.15])
monk_opto_stds_err =    np.array([ 6.47,  5.90,  6.20,  4.93,  4.74])
monk_diff_means_err =   np.array([ 5.90,  5.84,  6.28,  5.75,  5.76])
monk_diff_stds_err =    np.array([ 8.74,  8.01, 10.04,  8.51,  8.94])

mean_itp = PchipInterpolator(aXs, means,extrapolate=True)
std_itp = PchipInterpolator(aXs, stds,extrapolate=True)

def residuals(prop_aXs):
    pred_means = mean_itp(prop_aXs)
    pred_stds = std_itp(prop_aXs)
    res = np.array([(pred_means[:,0]-monk_base_means)/monk_base_means_err,
                    (pred_stds[:,0]-monk_base_stds)/monk_base_stds_err,
                    (pred_means[:,1]-monk_opto_means)/monk_opto_means_err,
                    (pred_stds[:,1]-monk_opto_stds)/monk_opto_stds_err,
                    (pred_means[:,2]-monk_diff_means)/monk_diff_means_err,
                    (pred_stds[:,2]-monk_diff_stds)/monk_diff_stds_err])
    return res.ravel()

x0 = np.linspace(0,1,monk_nc+1)[1:]
results = least_squares(residuals,x0)
best_aXs = results.x

best_means = np.concatenate([means[0:1,:], mean_itp(best_aXs)],axis=0)
best_stds = np.concatenate([stds[0:1,:], std_itp(best_aXs)],axis=0)
cost = results.cost

print("Saving statistics took ",time.process_time() - start," s")
print('')

res_dict = {}
    
res_dict['prms'] = prms
res_dict['best_aXs'] = best_aXs
res_dict['best_means'] = best_means
res_dict['best_stds'] = best_stds
res_dict['cost'] = cost

res_file = f'./../results/sim_test_j_{j_idx}'

with open(res_file+'.pkl', 'wb') as handle:
    pickle.dump(res_dict,handle)
