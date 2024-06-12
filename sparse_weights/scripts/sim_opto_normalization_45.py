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
parser.add_argument('--o_idx', '-o',  help='whether opto is on', type=int, default=0)
parser.add_argument('--base_mult', '-b',  help='baseline multiplier', type=float, default=1.0)
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
c1_idx= args['c1_idx']
c2_idx= args['c2_idx']
o_idx= args['o_idx']
base_mult= args['base_mult']
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

id = None
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
bX = base_mult*res_dict['best_monk_bX']
if bX <= 1e-8:
    bX = 1e-8
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
Nori = 20
NE = 4*(N//Nori)//5
NI = 1*(N//Nori)//5

prms['Nori'] = Nori
prms['NE'] = NE
prms['NI'] = NI

seeds = np.arange(50)

print('simulating baseline mult {:.1f}'.format(base_mult))
print('')

print('simulating contrast # '+str(c1_idx+1)+' for peak 1, contrast # '+str(c2_idx+1)+' for peak 2')
print('')
cA1 = aXs[c1_idx]/bX
cA2 = aXs[c2_idx]/bX
rX = bX

μrEs = np.zeros((len(seeds),5,Nori))
μrIs = np.zeros((len(seeds),5,Nori))
ΣrEs = np.zeros((len(seeds),7,Nori))
ΣrIs = np.zeros((len(seeds),7,Nori))
μhEs = np.zeros((len(seeds),5,Nori))
μhIs = np.zeros((len(seeds),5,Nori))
ΣhEs = np.zeros((len(seeds),7,Nori))
ΣhIs = np.zeros((len(seeds),7,Nori))
balEs = np.zeros((len(seeds),3,Nori))
balIs = np.zeros((len(seeds),3,Nori))
Lexps = np.zeros((len(seeds),3))
timeouts = np.zeros((len(seeds),3)).astype(bool)

def simulate_networks(prms,rX,cA1,cA2,CVh):
    N = prms.get('Nori',180) * (prms.get('NE',4) + prms.get('NI',1))
    rs = np.zeros((len(seeds),3,N))
    mus = np.zeros((len(seeds),3,N))
    muEs = np.zeros((len(seeds),3,N))
    muIs = np.zeros((len(seeds),3,N))
    Ls = np.zeros((len(seeds),3))
    TOs = np.zeros((len(seeds),3))

    for seed_idx,seed in enumerate(seeds):
        print('simulating seed # '+str(seed_idx+1))
        print('')
        
        start = time.process_time()

        net,this_M,this_H1,this_B,this_LAS,this_EPS = su.gen_ring_disorder_tensor(seed,prms,CVh)
        this_H2 = torch.roll(this_H1,N//4).to(device)
        M = this_M.cpu().numpy()
        H = (rX*(this_B+cA1*this_H1+cA2*this_H2)*this_EPS).cpu().numpy()
        LAS = this_LAS.cpu().numpy()

        print("Generating disorder took ",time.process_time() - start," s")
        print('')

        start = time.process_time()

        peak1_sol,peak1_timeout = integ.sim_dyn_tensor(ri,T,o_idx,this_M,rX*(this_B+cA1*this_H1)*this_EPS,
                                                     this_LAS,net.C_conds[0],mult_tau=True,max_min=30)
        Ls[seed_idx,0] = np.max(integ.calc_lyapunov_exp_tensor(ri,T[T>=3*Nt],o_idx,this_M,
                                                               rX*(this_B+cA1*this_H1)*this_EPS,this_LAS,
                                                               net.C_conds[0],peak1_sol[:,T>=3*Nt],10,1*Nt,2*ri.tE,
                                                               mult_tau=True).cpu().numpy())
        rs[seed_idx,0] = np.mean(peak1_sol[:,mask_time].cpu().numpy(),-1)
        TOs[seed_idx,0] = peak1_timeout

        print("Integrating peak 1 network took ",time.process_time() - start," s")
        print('')

        start = time.process_time()

        peak2_sol,peak2_timeout = integ.sim_dyn_tensor(ri,T,o_idx,this_M,rX*(this_B+cA2*this_H2)*this_EPS,
                                                     this_LAS,net.C_conds[0],mult_tau=True,max_min=30)
        Ls[seed_idx,1] = np.max(integ.calc_lyapunov_exp_tensor(ri,T[T>=3*Nt],o_idx,this_M,
                                                               rX*(this_B+cA2*this_H2)*this_EPS,this_LAS,
                                                               net.C_conds[0],peak2_sol[:,T>=3*Nt],10,1*Nt,2*ri.tE,
                                                               mult_tau=True).cpu().numpy())
        rs[seed_idx,1] = np.mean(peak2_sol[:,mask_time].cpu().numpy(),-1)
        TOs[seed_idx,1] = peak2_timeout

        print("Integrating peak 2 network took ",time.process_time() - start," s")
        print('')

        start = time.process_time()
        
        norm_sol,norm_timeout = integ.sim_dyn_tensor(ri,T,o_idx,this_M,rX*(this_B+cA1*this_H1+cA2*this_H2)*this_EPS,
                                                     this_LAS,net.C_conds[0],mult_tau=True,max_min=30)
        Ls[seed_idx,2] = np.max(integ.calc_lyapunov_exp_tensor(ri,T[T>=3*Nt],o_idx,this_M,
                                                               rX*(this_B+cA1*this_H1+cA2*this_H2)*this_EPS,this_LAS,
                                                               net.C_conds[0],norm_sol[:,T>=3*Nt],10,1*Nt,2*ri.tE,
                                                               mult_tau=True).cpu().numpy())
        rs[seed_idx,2] = np.mean(norm_sol[:,mask_time].cpu().numpy(),-1)
        TOs[seed_idx,2] = norm_timeout

        print("Integrating norm network took ",time.process_time() - start," s")
        print('')

        start = time.process_time()

        muEs[seed_idx] = (M[:,net.C_all[0]]@rs[seed_idx,:,net.C_all[0]]).T + H[None,:]
        muIs[seed_idx] = (M[:,net.C_all[1]]@rs[seed_idx,:,net.C_all[1]]).T
        muEs[seed_idx,:,net.C_all[0]] *= ri.tE
        muEs[seed_idx,:,net.C_all[1]] *= ri.tI
        muIs[seed_idx,:,net.C_all[0]] *= ri.tE
        muIs[seed_idx,:,net.C_all[1]] *= ri.tI
        muEs[seed_idx] = muEs[seed_idx] + o_idx*LAS
        mus[seed_idx] = muEs[seed_idx] + muIs[seed_idx]

        print("Calculating statistics took ",time.process_time() - start," s")
        print('')

    return net,rs,mus,muEs,muIs,Ls,TOs

# Simulate network
this_prms = prms.copy()

net,rs,mus,muEs,muIs,Ls,TOs = simulate_networks(this_prms,rX,cA1,cA2,CVh)

start = time.process_time()

for nloc in range(Nori):
    μrEs[:,:3,nloc] = np.mean(rs[:,:,net.C_idxs[0][nloc]],axis=-1)
    μrIs[:,:3,nloc] = np.mean(rs[:,:,net.C_idxs[1][nloc]],axis=-1)
    ΣrEs[:,:3,nloc] = np.var(rs[:,:,net.C_idxs[0][nloc]],axis=-1)
    ΣrIs[:,:3,nloc] = np.var(rs[:,:,net.C_idxs[1][nloc]],axis=-1)
    μhEs[:,:3,nloc] = np.mean(mus[:,:,net.C_idxs[0][nloc]],axis=-1)
    μhIs[:,:3,nloc] = np.mean(mus[:,:,net.C_idxs[1][nloc]],axis=-1)
    ΣhEs[:,:3,nloc] = np.var(mus[:,:,net.C_idxs[0][nloc]],axis=-1)
    ΣhIs[:,:3,nloc] = np.var(mus[:,:,net.C_idxs[1][nloc]],axis=-1)
    balEs[:,:,nloc] = np.mean(np.abs(mus[:,:,net.C_idxs[0][nloc]])/muEs[:,:,net.C_idxs[0][nloc]],axis=-1)
    balIs[:,:,nloc] = np.mean(np.abs(mus[:,:,net.C_idxs[1][nloc]])/muEs[:,:,net.C_idxs[1][nloc]],axis=-1)

    μrEs[:,3,nloc] = np.mean(rs[:,2,net.C_idxs[0][nloc]]-rs[:,0,net.C_idxs[0][nloc]],axis=-1)
    μrEs[:,4,nloc] = np.mean(rs[:,2,net.C_idxs[0][nloc]]-rs[:,1,net.C_idxs[0][nloc]],axis=-1)
    μrIs[:,3,nloc] = np.mean(rs[:,2,net.C_idxs[1][nloc]]-rs[:,0,net.C_idxs[1][nloc]],axis=-1)
    μrIs[:,4,nloc] = np.mean(rs[:,2,net.C_idxs[1][nloc]]-rs[:,1,net.C_idxs[1][nloc]],axis=-1)
    ΣrEs[:,3,nloc] = np.var(rs[:,2,net.C_idxs[0][nloc]]-rs[:,0,net.C_idxs[0][nloc]],axis=-1)
    ΣrEs[:,4,nloc] = np.var(rs[:,2,net.C_idxs[0][nloc]]-rs[:,1,net.C_idxs[0][nloc]],axis=-1)
    ΣrIs[:,3,nloc] = np.var(rs[:,2,net.C_idxs[1][nloc]]-rs[:,0,net.C_idxs[1][nloc]],axis=-1)
    ΣrIs[:,4,nloc] = np.var(rs[:,2,net.C_idxs[1][nloc]]-rs[:,1,net.C_idxs[1][nloc]],axis=-1)
    μhEs[:,3,nloc] = np.mean(mus[:,2,net.C_idxs[0][nloc]]-mus[:,0,net.C_idxs[0][nloc]],axis=-1)
    μhEs[:,4,nloc] = np.mean(mus[:,2,net.C_idxs[0][nloc]]-mus[:,1,net.C_idxs[0][nloc]],axis=-1)
    μhIs[:,3,nloc] = np.mean(mus[:,2,net.C_idxs[1][nloc]]-mus[:,0,net.C_idxs[1][nloc]],axis=-1)
    μhIs[:,4,nloc] = np.mean(mus[:,2,net.C_idxs[1][nloc]]-mus[:,1,net.C_idxs[1][nloc]],axis=-1)
    ΣhEs[:,3,nloc] = np.var(mus[:,2,net.C_idxs[0][nloc]]-mus[:,0,net.C_idxs[0][nloc]],axis=-1)
    ΣhEs[:,4,nloc] = np.var(mus[:,2,net.C_idxs[0][nloc]]-mus[:,1,net.C_idxs[0][nloc]],axis=-1)
    ΣhIs[:,3,nloc] = np.var(mus[:,2,net.C_idxs[1][nloc]]-mus[:,0,net.C_idxs[1][nloc]],axis=-1)
    ΣhIs[:,4,nloc] = np.var(mus[:,2,net.C_idxs[1][nloc]]-mus[:,1,net.C_idxs[1][nloc]],axis=-1)

    for seed_idx in range(len(seeds)):
        ΣrEs[seed_idx,5,nloc] = np.cov(rs[seed_idx,0,net.C_idxs[0][nloc]],
                                       rs[seed_idx,2,net.C_idxs[0][nloc]]-rs[seed_idx,0,net.C_idxs[0][nloc]])[0,1]
        ΣrEs[seed_idx,6,nloc] = np.cov(rs[seed_idx,1,net.C_idxs[0][nloc]],
                                       rs[seed_idx,2,net.C_idxs[0][nloc]]-rs[seed_idx,1,net.C_idxs[0][nloc]])[0,1]
        ΣrIs[seed_idx,5,nloc] = np.cov(rs[seed_idx,0,net.C_idxs[1][nloc]],
                                       rs[seed_idx,2,net.C_idxs[1][nloc]]-rs[seed_idx,0,net.C_idxs[1][nloc]])[0,1]
        ΣrIs[seed_idx,6,nloc] = np.cov(rs[seed_idx,1,net.C_idxs[1][nloc]],
                                       rs[seed_idx,2,net.C_idxs[1][nloc]]-rs[seed_idx,1,net.C_idxs[1][nloc]])[0,1]
Lexps[:,:] = Ls
timeouts[:,:] = TOs

seed_mask = np.logical_not(np.any(timeouts,axis=-1))
vsm1_mask = net.get_oriented_neurons()[0]
vsm2_mask = net.get_oriented_neurons(vis_ori=45)[0]

peak1_rates = rs[:,0,:]
peak2_rates = rs[:,1,:]
norm_rates = rs[:,2,:]
diff1_rates = norm_rates - peak1_rates
diff2_rates = norm_rates - peak2_rates

all_peak1_means = np.mean(peak1_rates[seed_mask,:])
all_peak1_stds = np.std(peak1_rates[seed_mask,:])
all_norm_means = np.mean(norm_rates[seed_mask,:])
all_norm_stds = np.std(norm_rates[seed_mask,:])
all_diff1_means = np.mean(diff1_rates[seed_mask,:])
all_diff2_means = np.mean(diff2_rates[seed_mask,:])
all_diff1_stds = np.std(diff1_rates[seed_mask,:])
all_diff2_stds = np.std(diff2_rates[seed_mask,:])
all_norm1_covs = np.cov(peak1_rates[seed_mask,:].flatten(),
    diff1_rates[seed_mask,:].flatten())[0,1] / all_diff1_stds**2
all_norm2_covs = np.cov(peak2_rates[seed_mask,:].flatten(),
    diff2_rates[seed_mask,:].flatten())[0,1] / all_diff2_stds**2

vsm1_peak1_means = np.mean(peak1_rates[seed_mask,:][:,vsm1_mask])
vsm1_peak1_stds = np.std(peak1_rates[seed_mask,:][:,vsm1_mask])
vsm1_norm_means = np.mean(norm_rates[seed_mask,:][:,vsm1_mask])
vsm1_norm_stds = np.std(norm_rates[seed_mask,:][:,vsm1_mask])
vsm1_diff1_means = np.mean(diff1_rates[seed_mask,vsm1_mask])
vsm1_diff2_means = np.mean(diff2_rates[seed_mask,vsm1_mask])
vsm1_diff1_stds = np.std(diff1_rates[seed_mask,vsm1_mask])
vsm1_diff2_stds = np.std(diff2_rates[seed_mask,vsm1_mask])
vsm1_norm1_covs = np.cov(peak1_rates[seed_mask,vsm1_mask].flatten(),
    diff1_rates[seed_mask,vsm1_mask].flatten())[0,1] / vsm1_diff1_stds**2
vsm1_norm2_covs = np.cov(peak2_rates[seed_mask,vsm1_mask].flatten(),
    diff2_rates[seed_mask,vsm1_mask].flatten())[0,1] / vsm1_diff2_stds**2

vsm2_peak1_means = np.mean(peak1_rates[seed_mask,:][:,vsm2_mask])
vsm2_peak1_stds = np.std(peak1_rates[seed_mask,:][:,vsm2_mask])
vsm2_norm_means = np.mean(norm_rates[seed_mask,:][:,vsm2_mask])
vsm2_norm_stds = np.std(norm_rates[seed_mask,:][:,vsm2_mask])
vsm2_diff1_means = np.mean(diff1_rates[seed_mask,vsm2_mask])
vsm2_diff2_means = np.mean(diff2_rates[seed_mask,vsm2_mask])
vsm2_diff1_stds = np.std(diff1_rates[seed_mask,vsm2_mask])
vsm2_diff2_stds = np.std(diff2_rates[seed_mask,vsm2_mask])
vsm2_norm1_covs = np.cov(peak1_rates[seed_mask,vsm2_mask].flatten(),
    diff1_rates[seed_mask,vsm2_mask].flatten())[0,1] / vsm2_diff1_stds**2
vsm2_norm2_covs = np.cov(peak2_rates[seed_mask,vsm2_mask].flatten(),
    diff2_rates[seed_mask,vsm2_mask].flatten())[0,1] / vsm2_diff2_stds**2

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
res_dict['all_peak1_means'] = all_peak1_means
res_dict['all_peak1_stds'] = all_peak1_stds
res_dict['all_norm_means'] = all_norm_means
res_dict['all_norm_stds'] = all_norm_stds
res_dict['all_diff1_means'] = all_diff1_means
res_dict['all_diff2_means'] = all_diff2_means
res_dict['all_diff1_stds'] = all_diff1_stds
res_dict['all_diff2_stds'] = all_diff2_stds
res_dict['all_norm1_covs'] = all_norm1_covs
res_dict['all_norm2_covs'] = all_norm2_covs
res_dict['vsm1_peak1_means'] = vsm1_peak1_means
res_dict['vsm1_peak1_stds'] = vsm1_peak1_stds
res_dict['vsm1_norm_means'] = vsm1_norm_means
res_dict['vsm1_norm_stds'] = vsm1_norm_stds
res_dict['vsm1_diff1_means'] = vsm1_diff1_means
res_dict['vsm1_diff2_means'] = vsm1_diff2_means
res_dict['vsm1_diff1_stds'] = vsm1_diff1_stds
res_dict['vsm1_diff2_stds'] = vsm1_diff2_stds
res_dict['vsm1_norm1_covs'] = vsm1_norm1_covs
res_dict['vsm1_norm2_covs'] = vsm1_norm2_covs
res_dict['vsm2_peak1_means'] = vsm2_peak1_means
res_dict['vsm2_peak1_stds'] = vsm2_peak1_stds
res_dict['vsm2_norm_means'] = vsm2_norm_means
res_dict['vsm2_norm_stds'] = vsm2_norm_stds
res_dict['vsm2_diff1_means'] = vsm2_diff1_means
res_dict['vsm2_diff2_means'] = vsm2_diff2_means
res_dict['vsm2_diff1_stds'] = vsm2_diff1_stds
res_dict['vsm2_diff2_stds'] = vsm2_diff2_stds
res_dict['vsm2_norm1_covs'] = vsm2_norm1_covs
res_dict['vsm2_norm2_covs'] = vsm2_norm2_covs
res_dict['timeouts'] = timeouts

res_file = './../results/opto_norm_45_id_{:s}'.format(str(id))
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
res_file = res_file + '_base_{:.1f}_c1_{:d}_c2_{:d}'.format(base_mult,c1_idx,c2_idx)

with open(res_file+'.pkl', 'wb') as handle:
    pickle.dump(res_dict,handle)