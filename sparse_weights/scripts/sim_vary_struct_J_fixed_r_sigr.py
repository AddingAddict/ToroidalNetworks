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

parser.add_argument('--struct_idx', '-s',  help='which structured', type=int, default=0)
parser.add_argument('--J_idx', '-j',  help='which J', type=int, default=0)
parser.add_argument('--fix_r_mode', '-m',  help='whether we are fixing r or simulating', type=ast.literal_eval, default=0)
args = vars(parser.parse_args())
print(parser.parse_args())
struct_idx= args['struct_idx']
J_idx= args['J_idx']
fix_r_mode= bool(args['fix_r_mode'])

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
CVh0 = res_dict['best_monk_eX']
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
fix_r_seeds = np.arange(10)

r_target = 63.512434943863234
sigr_target = 54.44306897905883

structs = np.arange(0,8+1)/8
Js = J*10**(np.arange(0,8+1)/8-0.5)

print('simulating struct # '+str(struct_idx+1))
print('')
struct = structs[struct_idx]

print('simulating J # '+str(J_idx+1))
print('')
newJ = Js[J_idx]

cA = aXs[-1]/bX
rX0 = bX

def find_rX_to_fix_r(prms,rX0,CVh0,cA,min_rfact=0.7,max_rfact=1.1,min_cfact=0.9,max_cfact=1.3):
    rXs = rX0*((max_rfact-min_rfact)*np.arange(0,4+1)/4 + min_rfact)
    CVhs = CVh0*((max_cfact-min_cfact)*np.arange(0,4+1)/4 + min_cfact)
    vsm_rs = np.zeros((len(rXs),len(CVhs),len(fix_r_seeds)))
    vsm_sigrs = np.zeros((len(rXs),len(CVhs),len(fix_r_seeds)))

    for rX_idx,rX in enumerate(rXs):
        print('simulating rX # '+str(rX_idx+1))
        print('')
        for CVh_idx,CVh in enumerate(CVhs):
            print('simulating CVh # '+str(CVh_idx+1))
            print('')
            for seed_idx,seed in enumerate(fix_r_seeds):
                print('simulating seed # '+str(seed_idx+1))
                print('')
                
                start = time.process_time()

                net,this_M,this_H,this_B,this_LAS,this_EPS = su.gen_ring_disorder_tensor(seed,prms,CVh)

                print("Generating disorder took ",time.process_time() - start," s")
                print('')

                start = time.process_time()

                sol,_ = integ.sim_dyn_tensor(ri,T,0.0,this_M,rX*(this_B+cA*this_H)*this_EPS,
                                            this_LAS,net.C_conds[0],mult_tau=True,max_min=30)
                rates = np.mean(sol[:,mask_time].cpu().numpy(),-1)
                vsm_rs[rX_idx,CVh_idx,seed_idx] = np.mean(rates[net.get_oriented_neurons()])
                vsm_sigrs[rX_idx,CVh_idx,seed_idx] = np.std(rates[net.get_oriented_neurons()])

                print("Integrating base network took ",time.process_time() - start," s")
                print('this vsm_rs =',vsm_rs[rX_idx,CVh_idx,seed_idx])
                print('this vsm_sigrs =',vsm_sigrs[rX_idx,CVh_idx,seed_idx])
                print('')
            
    vsm_rs = np.mean(vsm_rs,-1)
    vsm_sigrs = np.mean(vsm_sigrs,-1)

    vsm_rs_itp = RegularGridInterpolator((rXs,CVhs), vsm_rs)
    vsm_sigrs_itp = RegularGridInterpolator((rXs,CVhs), vsm_sigrs)
    
    def residuals(x):
        pred_vsm_r = vsm_rs_itp(x)
        pred_vsm_sigr = vsm_sigrs_itp(x)
        res = np.array([r_target - pred_vsm_r,sigr_target - pred_vsm_sigr])
        return res.ravel()
    
    print('rXs =',rXs)
    print('CVhs =',CVhs)
    print('vsm_rs =',vsm_rs)
    print('vsm_sigrs =',vsm_sigrs)
    
    xmin = (rXs[0],CVhs[0])
    xmax = (rXs[-1],CVhs[-1])
    x0 = [0.5*(rXs[0]+rXs[-1]),0.5*(CVhs[0]+CVhs[-1])]
    results = least_squares(residuals,x0,bounds=(xmin,xmax))
    
    print('best fit rX =',results.x[0])
    print('best fit CVh =',results.x[1])
    print('predicted vsm_r =',vsm_rs_itp(results.x))
    print('predicted vsm_sigr =',vsm_sigrs_itp(results.x))
    
    return results.x

def simulate_networks(prms,rX,cA,CVh):
    N = prms.get('Nori',180) * (prms.get('NE',4) + prms.get('NI',1))
    rs = np.zeros((len(seeds),2,N))
    mus = np.zeros((len(seeds),2,N))
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
        Ls[seed_idx,0] = np.max(integ.calc_lyapunov_exp_tensor(ri,T[T>=3*Nt],0.0,this_M,
                                                               rX*(this_B+cA*this_H)*this_EPS,this_LAS,
                                                               net.C_conds[0],base_sol[:,T>=3*Nt],10,1*Nt,2*ri.tE,
                                                               mult_tau=True).cpu().numpy())
        rs[seed_idx,0] = np.mean(base_sol[:,mask_time].cpu().numpy(),-1)
        TOs[seed_idx,0] = base_timeout

        print("Integrating base network took ",time.process_time() - start," s")
        print('')

        start = time.process_time()
        
        opto_sol,opto_timeout = integ.sim_dyn_tensor(ri,T,1.0,this_M,rX*(this_B+cA*this_H)*this_EPS,
                                                     this_LAS,net.C_conds[0],mult_tau=True,max_min=30)
        Ls[seed_idx,1] = np.max(integ.calc_lyapunov_exp_tensor(ri,T[T>=3*Nt],1.0,this_M,
                                                               rX*(this_B+cA*this_H)*this_EPS,this_LAS,
                                                               net.C_conds[0],opto_sol[:,T>=3*Nt],10,1*Nt,2*ri.tE,
                                                               mult_tau=True).cpu().numpy())
        rs[seed_idx,1] = np.mean(opto_sol[:,mask_time].cpu().numpy(),-1)
        TOs[seed_idx,1] = opto_timeout

        print("Integrating opto network took ",time.process_time() - start," s")
        print('')

        start = time.process_time()

        muEs[seed_idx,0] = M[:,net.C_all[0]]@rs[seed_idx,0,net.C_all[0]] + H
        muIs[seed_idx,0] = M[:,net.C_all[1]]@rs[seed_idx,0,net.C_all[1]]
        muEs[seed_idx,0,net.C_all[0]] *= ri.tE
        muEs[seed_idx,0,net.C_all[1]] *= ri.tI
        muIs[seed_idx,0,net.C_all[0]] *= ri.tE
        muIs[seed_idx,0,net.C_all[1]] *= ri.tI
        mus[seed_idx,0] = muEs[seed_idx,0] + muIs[seed_idx,0]

        muEs[seed_idx,1] = M[:,net.C_all[0]]@rs[seed_idx,1,net.C_all[0]] + H
        muIs[seed_idx,1] = M[:,net.C_all[1]]@rs[seed_idx,1,net.C_all[1]]
        muEs[seed_idx,1,net.C_all[0]] *= ri.tE
        muEs[seed_idx,1,net.C_all[1]] *= ri.tI
        muIs[seed_idx,1,net.C_all[0]] *= ri.tE
        muIs[seed_idx,1,net.C_all[1]] *= ri.tI
        muEs[seed_idx,1] = muEs[seed_idx,1] + LAS
        mus[seed_idx,1] = muEs[seed_idx,1] + muIs[seed_idx,1]

        print("Calculating statistics took ",time.process_time() - start," s")
        print('')

    return net,rs,mus,muEs,muIs,Ls,TOs

# Simulate network where structure is removed by increasing baseline fraction
print('simulating baseline fraction network')
print('')
this_prms = prms.copy()
this_prms['J'] = newJ
this_prms['basefrac'] = 1-struct

res_dict = {}

if fix_r_mode:
    if J_idx == 0:
        rX,CVh = find_rX_to_fix_r(this_prms,bX,CVh0,cA,1.6,1.7)
    elif J_idx == 1:
        rX,CVh = find_rX_to_fix_r(this_prms,bX,CVh0,cA,1.3,1.5)
    elif J_idx == 2:
        rX,CVh = find_rX_to_fix_r(this_prms,bX,CVh0,cA,1.1,1.3)
    elif J_idx == 3:
        rX,CVh = find_rX_to_fix_r(this_prms,bX,CVh0,cA,1.0,1.2)
    elif J_idx == 4:
        rX,CVh = find_rX_to_fix_r(this_prms,bX,CVh0,cA)#,0.92,1.12)
    elif J_idx == 5:
        rX,CVh = find_rX_to_fix_r(this_prms,bX,CVh0,cA,0.85,1.05)
    elif J_idx == 6:
        rX,CVh = find_rX_to_fix_r(this_prms,bX,CVh0,cA,0.8,1.0)
    elif J_idx == 7:
        rX,CVh = find_rX_to_fix_r(this_prms,bX,CVh0,cA,0.75,0.95)
    else:
        rX,CVh = find_rX_to_fix_r(this_prms,bX,CVh0,cA,0.72,0.92)
    print(rX)
    print(CVh)
    
    try:
        with open('./../results/vary_fixed_r_sigr_struct_{:d}_J_{:d}'.format(struct_idx,J_idx)+'.pkl', 'rb') as handle:
            res_dict = pickle.load(handle)
    except:
        res_dict = {}
    res_dict['rX'] = rX
    res_dict['CVh'] = CVh
    res_dict['ran_fix_r_mode'] = True

else:
    with open('./../results/vary_fixed_r_sigr_struct_{:d}_J_{:d}'.format(struct_idx,J_idx)+'.pkl', 'rb') as handle:
        res_dict = pickle.load(handle)
    rX = res_dict['rX']
    CVh = res_dict['CVh']
    if not res_dict['ran_fix_r_mode']: raise Exception('Did not run fixed r mode first')
        
    μrEs = np.zeros((len(seeds),3,Nori))
    μrIs = np.zeros((len(seeds),3,Nori))
    ΣrEs = np.zeros((len(seeds),4,Nori))
    ΣrIs = np.zeros((len(seeds),4,Nori))
    μhEs = np.zeros((len(seeds),3,Nori))
    μhIs = np.zeros((len(seeds),3,Nori))
    ΣhEs = np.zeros((len(seeds),4,Nori))
    ΣhIs = np.zeros((len(seeds),4,Nori))
    balEs = np.zeros((len(seeds),2,Nori))
    balIs = np.zeros((len(seeds),2,Nori))
    Lexps = np.zeros((len(seeds),2))
    timeouts = np.zeros((len(seeds),2)).astype(bool)
    # μrEs = np.zeros((2,len(seeds),3,Nori))
    # μrIs = np.zeros((2,len(seeds),3,Nori))
    # ΣrEs = np.zeros((2,len(seeds),4,Nori))
    # ΣrIs = np.zeros((2,len(seeds),4,Nori))
    # μhEs = np.zeros((2,len(seeds),3,Nori))
    # μhIs = np.zeros((2,len(seeds),3,Nori))
    # ΣhEs = np.zeros((2,len(seeds),4,Nori))
    # ΣhIs = np.zeros((2,len(seeds),4,Nori))
    # balEs = np.zeros((2,len(seeds),2,Nori))
    # balIs = np.zeros((2,len(seeds),2,Nori))
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

    net,rs,mus,muEs,muIs,Ls,TOs = simulate_networks(this_prms,rX,cA,CVh)

    start = time.process_time()

    for nloc in range(Nori):
        μrEs[:,:2,nloc] = np.mean(rs[:,:,net.C_idxs[0][nloc]],axis=-1)
        μrIs[:,:2,nloc] = np.mean(rs[:,:,net.C_idxs[1][nloc]],axis=-1)
        ΣrEs[:,:2,nloc] = np.var(rs[:,:,net.C_idxs[0][nloc]],axis=-1)
        ΣrIs[:,:2,nloc] = np.var(rs[:,:,net.C_idxs[1][nloc]],axis=-1)
        μhEs[:,:2,nloc] = np.mean(mus[:,:,net.C_idxs[0][nloc]],axis=-1)
        μhIs[:,:2,nloc] = np.mean(mus[:,:,net.C_idxs[1][nloc]],axis=-1)
        ΣhEs[:,:2,nloc] = np.var(mus[:,:,net.C_idxs[0][nloc]],axis=-1)
        ΣhIs[:,:2,nloc] = np.var(mus[:,:,net.C_idxs[1][nloc]],axis=-1)
        balEs[:,:,nloc] = np.mean(np.abs(mus[:,:,net.C_idxs[0][nloc]])/muEs[:,:,net.C_idxs[0][nloc]],axis=-1)
        balIs[:,:,nloc] = np.mean(np.abs(mus[:,:,net.C_idxs[1][nloc]])/muEs[:,:,net.C_idxs[1][nloc]],axis=-1)

        μrEs[:,2,nloc] = np.mean(rs[:,1,net.C_idxs[0][nloc]]-rs[:,0,net.C_idxs[0][nloc]],axis=-1)
        μrIs[:,2,nloc] = np.mean(rs[:,1,net.C_idxs[1][nloc]]-rs[:,0,net.C_idxs[1][nloc]],axis=-1)
        ΣrEs[:,2,nloc] = np.var(rs[:,1,net.C_idxs[0][nloc]]-rs[:,0,net.C_idxs[0][nloc]],axis=-1)
        ΣrIs[:,2,nloc] = np.var(rs[:,1,net.C_idxs[1][nloc]]-rs[:,0,net.C_idxs[1][nloc]],axis=-1)
        μhEs[:,2,nloc] = np.mean(mus[:,1,net.C_idxs[0][nloc]]-mus[:,0,net.C_idxs[0][nloc]],axis=-1)
        μhIs[:,2,nloc] = np.mean(mus[:,1,net.C_idxs[1][nloc]]-mus[:,0,net.C_idxs[1][nloc]],axis=-1)
        ΣhEs[:,2,nloc] = np.var(mus[:,1,net.C_idxs[0][nloc]]-mus[:,0,net.C_idxs[0][nloc]],axis=-1)
        ΣhIs[:,2,nloc] = np.var(mus[:,1,net.C_idxs[1][nloc]]-mus[:,0,net.C_idxs[1][nloc]],axis=-1)

        for seed_idx in range(len(seeds)):
            ΣrEs[seed_idx,3,nloc] = np.cov(rs[seed_idx,0,net.C_idxs[0][nloc]],
                                        rs[seed_idx,1,net.C_idxs[0][nloc]]-rs[seed_idx,0,net.C_idxs[0][nloc]])[0,1]
            ΣrIs[seed_idx,3,nloc] = np.cov(rs[seed_idx,0,net.C_idxs[1][nloc]],
                                        rs[seed_idx,1,net.C_idxs[1][nloc]]-rs[seed_idx,0,net.C_idxs[1][nloc]])[0,1]
    Lexps[:,:] = Ls
    timeouts[:,:] = TOs

    seed_mask = np.logical_not(np.any(timeouts,axis=-1))
    vsm_mask = net.get_oriented_neurons()[0]

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

    vsm_base_means = np.mean(base_rates[seed_mask,:][:,vsm_mask])
    vsm_base_stds = np.std(base_rates[seed_mask,:][:,vsm_mask])
    vsm_opto_means = np.mean(opto_rates[seed_mask,:][:,vsm_mask])
    vsm_opto_stds = np.std(opto_rates[seed_mask,:][:,vsm_mask])
    vsm_diff_means = np.mean(diff_rates[seed_mask,:][:,vsm_mask])
    vsm_diff_stds = np.std(diff_rates[seed_mask,:][:,vsm_mask])
    vsm_norm_covs = np.cov(base_rates[seed_mask,:][:,vsm_mask].flatten(),
        diff_rates[seed_mask,:][:,vsm_mask].flatten())[0,1] / vsm_diff_stds**2

    print("Saving statistics took ",time.process_time() - start," s")
    print('')

    # start = time.process_time()

    # for nloc in range(Nori):
    #     μrEs[0,:,:2,nloc] = np.mean(rs[:,:,net.C_idxs[0][nloc]],axis=-1)
    #     μrIs[0,:,:2,nloc] = np.mean(rs[:,:,net.C_idxs[1][nloc]],axis=-1)
    #     ΣrEs[0,:,:2,nloc] = np.var(rs[:,:,net.C_idxs[0][nloc]],axis=-1)
    #     ΣrIs[0,:,:2,nloc] = np.var(rs[:,:,net.C_idxs[1][nloc]],axis=-1)
    #     μhEs[0,:,:2,nloc] = np.mean(mus[:,:,net.C_idxs[0][nloc]],axis=-1)
    #     μhIs[0,:,:2,nloc] = np.mean(mus[:,:,net.C_idxs[1][nloc]],axis=-1)
    #     ΣhEs[0,:,:2,nloc] = np.var(mus[:,:,net.C_idxs[0][nloc]],axis=-1)
    #     ΣhIs[0,:,:2,nloc] = np.var(mus[:,:,net.C_idxs[1][nloc]],axis=-1)
    #     balEs[0,:,:,nloc] = np.mean(np.abs(mus[:,:,net.C_idxs[0][nloc]])/muEs[:,:,net.C_idxs[0][nloc]],axis=-1)
    #     balIs[0,:,:,nloc] = np.mean(np.abs(mus[:,:,net.C_idxs[1][nloc]])/muEs[:,:,net.C_idxs[1][nloc]],axis=-1)

    #     μrEs[0,:,2,nloc] = np.mean(rs[:,1,net.C_idxs[0][nloc]]-rs[:,0,net.C_idxs[0][nloc]],axis=-1)
    #     μrIs[0,:,2,nloc] = np.mean(rs[:,1,net.C_idxs[1][nloc]]-rs[:,0,net.C_idxs[1][nloc]],axis=-1)
    #     ΣrEs[0,:,2,nloc] = np.var(rs[:,1,net.C_idxs[0][nloc]]-rs[:,0,net.C_idxs[0][nloc]],axis=-1)
    #     ΣrIs[0,:,2,nloc] = np.var(rs[:,1,net.C_idxs[1][nloc]]-rs[:,0,net.C_idxs[1][nloc]],axis=-1)
    #     μhEs[0,:,2,nloc] = np.mean(mus[:,1,net.C_idxs[0][nloc]]-mus[:,0,net.C_idxs[0][nloc]],axis=-1)
    #     μhIs[0,:,2,nloc] = np.mean(mus[:,1,net.C_idxs[1][nloc]]-mus[:,0,net.C_idxs[1][nloc]],axis=-1)
    #     ΣhEs[0,:,2,nloc] = np.var(mus[:,1,net.C_idxs[0][nloc]]-mus[:,0,net.C_idxs[0][nloc]],axis=-1)
    #     ΣhIs[0,:,2,nloc] = np.var(mus[:,1,net.C_idxs[1][nloc]]-mus[:,0,net.C_idxs[1][nloc]],axis=-1)

    #     for seed_idx in range(len(seeds)):
    #         ΣrEs[0,seed_idx,3,nloc] = np.cov(rs[seed_idx,0,net.C_idxs[0][nloc]],
    #                                          rs[seed_idx,1,net.C_idxs[0][nloc]]-rs[seed_idx,0,net.C_idxs[0][nloc]])[0,1]
    #         ΣrIs[0,seed_idx,3,nloc] = np.cov(rs[seed_idx,0,net.C_idxs[1][nloc]],
    #                                          rs[seed_idx,1,net.C_idxs[1][nloc]]-rs[seed_idx,0,net.C_idxs[1][nloc]])[0,1]
    # Lexps[0,:,:] = Ls
    # timeouts[0,:,:] = TOs

    # seed_mask = np.logical_not(np.any(timeouts[0],axis=-1))
    # vsm_mask = net.get_oriented_neurons()[0]

    # base_rates = rs[:,0,:]
    # opto_rates = rs[:,1,:]
    # diff_rates = opto_rates - base_rates

    # all_base_means[0] = np.mean(base_rates[seed_mask,:])
    # all_base_stds[0] = np.std(base_rates[seed_mask,:])
    # all_opto_means[0] = np.mean(opto_rates[seed_mask,:])
    # all_opto_stds[0] = np.std(opto_rates[seed_mask,:])
    # all_diff_means[0] = np.mean(diff_rates[seed_mask,:])
    # all_diff_stds[0] = np.std(diff_rates[seed_mask,:])
    # all_norm_covs[0] = np.cov(base_rates[seed_mask,:].flatten(),
    #     diff_rates[seed_mask,:].flatten())[0,1] / all_diff_stds[0]**2

    # vsm_base_means[0] = np.mean(base_rates[seed_mask,:][:,vsm_mask])
    # vsm_base_stds[0] = np.std(base_rates[seed_mask,:][:,vsm_mask])
    # vsm_opto_means[0] = np.mean(opto_rates[seed_mask,:][:,vsm_mask])
    # vsm_opto_stds[0] = np.std(opto_rates[seed_mask,:][:,vsm_mask])
    # vsm_diff_means[0] = np.mean(diff_rates[seed_mask,:][:,vsm_mask])
    # vsm_diff_stds[0] = np.std(diff_rates[seed_mask,:][:,vsm_mask])
    # vsm_norm_covs[0] = np.cov(base_rates[seed_mask,:][:,vsm_mask].flatten(),
    #     diff_rates[seed_mask,:][:,vsm_mask].flatten())[0,1] / vsm_diff_stds[0]**2

    # print("Saving statistics took ",time.process_time() - start," s")
    # print('')

    # # Simulate network where structure is removed by decreasing recurrent width relative to input width
    # print('simulating baseline fraction network')
    # print('')
    # this_prms = prms.copy()
    # this_prms['J'] = newJ
    # this_prms['SoriE'] = np.fmax(SoriE*struct,1e-12)
    # this_prms['SoriI'] = np.fmax(SoriI*struct,1e-12)

    # net,rs,mus,muEs,muIs,Ls,TOs = simulate_networks(this_prms,rX,cA,CVh)

    # start = time.process_time()

    # for nloc in range(Nori):
    #     μrEs[1,:,:2,nloc] = np.mean(rs[:,:,net.C_idxs[0][nloc]],axis=-1)
    #     μrIs[1,:,:2,nloc] = np.mean(rs[:,:,net.C_idxs[1][nloc]],axis=-1)
    #     ΣrEs[1,:,:2,nloc] = np.var(rs[:,:,net.C_idxs[0][nloc]],axis=-1)
    #     ΣrIs[1,:,:2,nloc] = np.var(rs[:,:,net.C_idxs[1][nloc]],axis=-1)
    #     μhEs[1,:,:2,nloc] = np.mean(mus[:,:,net.C_idxs[0][nloc]],axis=-1)
    #     μhIs[1,:,:2,nloc] = np.mean(mus[:,:,net.C_idxs[1][nloc]],axis=-1)
    #     ΣhEs[1,:,:2,nloc] = np.var(mus[:,:,net.C_idxs[0][nloc]],axis=-1)
    #     ΣhIs[1,:,:2,nloc] = np.var(mus[:,:,net.C_idxs[1][nloc]],axis=-1)
    #     balEs[1,:,:,nloc] = np.mean(np.abs(mus[:,:,net.C_idxs[0][nloc]])/muEs[:,:,net.C_idxs[0][nloc]],axis=-1)
    #     balIs[1,:,:,nloc] = np.mean(np.abs(mus[:,:,net.C_idxs[1][nloc]])/muEs[:,:,net.C_idxs[1][nloc]],axis=-1)

    #     μrEs[1,:,2,nloc] = np.mean(rs[:,1,net.C_idxs[0][nloc]]-rs[:,0,net.C_idxs[0][nloc]],axis=-1)
    #     μrIs[1,:,2,nloc] = np.mean(rs[:,1,net.C_idxs[1][nloc]]-rs[:,0,net.C_idxs[1][nloc]],axis=-1)
    #     ΣrEs[1,:,2,nloc] = np.var(rs[:,1,net.C_idxs[0][nloc]]-rs[:,0,net.C_idxs[0][nloc]],axis=-1)
    #     ΣrIs[1,:,2,nloc] = np.var(rs[:,1,net.C_idxs[1][nloc]]-rs[:,0,net.C_idxs[1][nloc]],axis=-1)
    #     μhEs[1,:,2,nloc] = np.mean(mus[:,1,net.C_idxs[0][nloc]]-mus[:,0,net.C_idxs[0][nloc]],axis=-1)
    #     μhIs[1,:,2,nloc] = np.mean(mus[:,1,net.C_idxs[1][nloc]]-mus[:,0,net.C_idxs[1][nloc]],axis=-1)
    #     ΣhEs[1,:,2,nloc] = np.var(mus[:,1,net.C_idxs[0][nloc]]-mus[:,0,net.C_idxs[0][nloc]],axis=-1)
    #     ΣhIs[1,:,2,nloc] = np.var(mus[:,1,net.C_idxs[1][nloc]]-mus[:,0,net.C_idxs[1][nloc]],axis=-1)

    #     for seed_idx in range(len(seeds)):
    #         ΣrEs[1,seed_idx,3,nloc] = np.cov(rs[seed_idx,0,net.C_idxs[0][nloc]],
    #                                          rs[seed_idx,1,net.C_idxs[0][nloc]]-rs[seed_idx,0,net.C_idxs[0][nloc]])[0,1]
    #         ΣrIs[1,seed_idx,3,nloc] = np.cov(rs[seed_idx,0,net.C_idxs[1][nloc]],
    #                                          rs[seed_idx,1,net.C_idxs[1][nloc]]-rs[seed_idx,0,net.C_idxs[1][nloc]])[0,1]
    # Lexps[1,:,:] = Ls
    # timeouts[1,:,:] = TOs

    # seed_mask = np.logical_not(np.any(timeouts[1],axis=-1))
    # vsm_mask = net.get_oriented_neurons()[0]

    # base_rates = rs[:,0,:]
    # opto_rates = rs[:,1,:]
    # diff_rates = opto_rates - base_rates

    # all_base_means[1] = np.mean(base_rates[seed_mask,:])
    # all_base_stds[1] = np.std(base_rates[seed_mask,:])
    # all_opto_means[1] = np.mean(opto_rates[seed_mask,:])
    # all_opto_stds[1] = np.std(opto_rates[seed_mask,:])
    # all_diff_means[1] = np.mean(diff_rates[seed_mask,:])
    # all_diff_stds[1] = np.std(diff_rates[seed_mask,:])
    # all_norm_covs[1] = np.cov(base_rates[seed_mask,:].flatten(),
    #     diff_rates[seed_mask,:].flatten())[0,1] / all_diff_stds[1]**2

    # vsm_base_means[1] = np.mean(base_rates[seed_mask,:][:,vsm_mask])
    # vsm_base_stds[1] = np.std(base_rates[seed_mask,:][:,vsm_mask])
    # vsm_opto_means[1] = np.mean(opto_rates[seed_mask,:][:,vsm_mask])
    # vsm_opto_stds[1] = np.std(opto_rates[seed_mask,:][:,vsm_mask])
    # vsm_diff_means[1] = np.mean(diff_rates[seed_mask,:][:,vsm_mask])
    # vsm_diff_stds[1] = np.std(diff_rates[seed_mask,:][:,vsm_mask])
    # vsm_norm_covs[1] = np.cov(base_rates[seed_mask,:][:,vsm_mask].flatten(),
    #     diff_rates[seed_mask,:][:,vsm_mask].flatten())[0,1] / vsm_diff_stds[1]**2

    # print("Saving statistics took ",time.process_time() - start," s")
    # print('')
        
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

with open('./../results/vary_fixed_r_sigr_id_{:s}_struct_{:d}_J_{:d}'.format(str(id),struct_idx,J_idx)+'.pkl', 'wb') as handle:
    pickle.dump(res_dict,handle)