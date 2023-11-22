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

parser.add_argument('--c_idx', '-c',  help='which correlation', type=int, default=0)
args = vars(parser.parse_args())
print(parser.parse_args())
c_idx= args['c_idx']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

net = network.RingNetwork(NC=[NE,NI],Nori=Nori)
net.generate_disorder(1e-3*np.array([[0.15,-1],[0.8,-3]]),
                      np.array([[30,20],[30,20]]),
                      np.array([0.25,0.20]),
                      np.array([20,20]),500)
net.generate_tensors()
rX = 20
this_B = torch.where(net.C_conds[0],0.25,0.20)

basefracs = np.arange(8+1)/8
seeds = np.arange(5)

corrs = np.linspace(0,1,21)[1:-1]
cvs = np.sqrt(1/(1-corrs)**2 - 1)

print('simulating input correlation # '+str(c_idx+1))
print('')
corr = corrs[c_idx]
cv = cvs[c_idx]

shape = 1/cv**2
scale = 1/shape

def simulate_networks(basefrac):
    inps = np.zeros((len(seeds),N))
    rates = np.zeros((len(seeds),N))
    Ls = np.zeros((len(seeds)))
    TOs = np.zeros((len(seeds)))
    
    this_M = net.M_torch
    this_H = net.H_torch
    
    mean_inps = rX*(basefrac*this_B + (1-basefrac)*this_H)
    sol,timeout = integ.sim_dyn_tensor(ri,T,0.0,this_M,mean_inps,this_H,net.C_conds[0],mult_tau=True,max_min=30)
    mean_rates = np.mean(sol[:,mask_time].cpu().numpy(),-1)

    for seed_idx,seed in enumerate(seeds):
        print('simulating seed # '+str(seed_idx+1))
        print('')
        
        start = time.process_time()
        
        this_EPS = torch.from_numpy(np.random.default_rng(seed).gamma(
            shape,scale=scale,size=net.N).astype(np.float32)).to(device)

        # M = this_M.cpu().numpy()
        inps[seed_idx] = (mean_inps*this_EPS).cpu().numpy()

        print("Generating disorder took ",time.process_time() - start," s")
        print('')

        start = time.process_time()

        sol,timeout = integ.sim_dyn_tensor(ri,T,0.0,this_M,mean_inps*this_EPS,this_H,net.C_conds[0],
                                           mult_tau=True,max_min=30)
        Ls[seed_idx] = np.max(integ.calc_lyapunov_exp_tensor(ri,T[T>=3*Nt],0.0,this_M,
                                                               mean_inps*this_EPS,this_H,
                                                               net.C_conds[0],sol[:,T>=3*Nt],10,1*Nt,2*ri.tE,
                                                               mult_tau=True).cpu().numpy())
        rates[seed_idx] = np.mean(sol[:,mask_time].cpu().numpy(),-1)
        TOs[seed_idx] = timeout

        print("Integrating cv network took ",time.process_time() - start," s")
        print('')

        start = time.process_time()

        print("Calculating statistics took ",time.process_time() - start," s")
        print('')
        
    mean_inps = mean_inps.cpu().numpy()

    # return mean_inps,rates,mus,muXs,muEs,muIs,Ls,TOs
    return mean_inps,mean_rates,inps,rates,Ls,TOs

# Simulate network where structure is removed by increasing baseline fraction
print('simulating baseline fraction network')
print('')
        
μrEs = np.zeros((len(basefracs),len(seeds),Nori))
μrIs = np.zeros((len(basefracs),len(seeds),Nori))
ΣrEs = np.zeros((len(basefracs),len(seeds),Nori))
ΣrIs = np.zeros((len(basefracs),len(seeds),Nori))
in_corrs = np.zeros((len(basefracs),len(seeds)))
out_corrs = np.zeros((len(basefracs),len(seeds)))
Lexps = np.zeros((len(basefracs),len(seeds)))
timeouts = np.zeros((len(basefracs),len(seeds))).astype(bool)

for frac_idx,frac in enumerate(basefracs):
    print('simulating basefrac # '+str(frac_idx+1))
    print('')
    
    mean_inps,mean_rates,inps,rates,Ls,TOs = simulate_networks(frac)

    start = time.process_time()

    for nloc in range(Nori):
        μrEs[frac_idx,:,nloc] = np.mean(rates[:,net.C_idxs[0][nloc]],axis=-1)
        μrIs[frac_idx,:,nloc] = np.mean(rates[:,net.C_idxs[1][nloc]],axis=-1)
        ΣrEs[frac_idx,:,nloc] = np.var(rates[:,net.C_idxs[0][nloc]],axis=-1)
        ΣrIs[frac_idx,:,nloc] = np.var(rates[:,net.C_idxs[1][nloc]],axis=-1)
        
    in_corrs[frac_idx,:] = 1 - np.sum(mean_inps[None,:]*inps,axis=-1)/\
        (np.linalg.norm(mean_inps)*np.linalg.norm(inps,axis=-1))
    out_corrs[frac_idx,:] = 1 - np.sum(mean_rates[None,:]*rates,axis=-1)/\
        (np.linalg.norm(mean_rates)*np.linalg.norm(rates,axis=-1))
        
    Lexps[frac_idx,:] = Ls
    timeouts[frac_idx,:] = TOs

    print("Saving statistics took ",time.process_time() - start," s")
    print('')

res_dict = {}

print(in_corrs)
print(out_corrs)
    
res_dict['μrEs'] = μrEs
res_dict['μrIs'] = μrIs
res_dict['ΣrEs'] = ΣrEs
res_dict['ΣrIs'] = ΣrIs
res_dict['in_corrs'] = in_corrs
res_dict['out_corrs'] = out_corrs
res_dict['Lexps'] = Lexps
res_dict['timeouts'] = timeouts

with open('./../results/in_out_corrs_{:d}'.format(c_idx)+'.pkl', 'wb') as handle:
    pickle.dump(res_dict,handle)