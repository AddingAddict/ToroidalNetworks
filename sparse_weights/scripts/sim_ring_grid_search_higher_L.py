import argparse
import os
try:
    import pickle5 as pickle
except:
    import pickle
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares
import time

import ring_network as network
import sim_util as su
import ricciardi as ric
import integrate as integ

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description=('This python script takes results from sampled spatial model parameters, '
    'trains a net to interpolate the results, and finds parameters that best fit the experimental results'))

parser.add_argument('--njob', '-n',  help='which number job', type=int, default=0)
args = vars(parser.parse_args())
print(parser.parse_args())
njob= args['njob']

ri = ric.Ricciardi()
ri.set_up_nonlinearity('./phi_int')
ri.set_up_nonlinearity_tensor()

NtE = 75
# T = np.linspace(0,NtE*ri.tE,round(NtE*ri.tE/(ri.tI/3))+1)
T = torch.linspace(0,NtE*ri.tE,round(NtE*ri.tE/(ri.tI/3))+1)
mask_time = T>(NtE/2*ri.tE)

aXs = np.arange(0,30+2,2)
bXs = np.arange(1,19+2,2)
eXs = np.arange(0,0.6+0.05,0.05)

net = network.RingNetwork(seed=0,NC=[4,1],Nori=1200)

eps = torch.zeros((len(eXs),net.N),dtype=torch.float32)
for eX_idx,eX in enumerate(eXs):
    if eX == 0.0:
        eps[eX_idx] = torch.ones(net.N,dtype=torch.float32)
    else:
        shape = 1/eX**2
        scale = 1/shape
        this_eps = np.random.default_rng(0).gamma(shape,scale=scale,size=net.N).astype(np.float32)
        eps[eX_idx] = torch.from_numpy(this_eps)
eps = eps.to(device)

mous_base_means =       np.array([ 6.21,  6.71,  7.16,  7.66,  7.99, 10.98, 16.73])
mous_base_stds =        np.array([ 5.79,  6.63,  6.92,  7.14,  7.07,  9.00, 13.64])
mous_opto_means =       np.array([10.50, 10.73, 11.20, 12.14, 12.92, 14.86, 20.20])
mous_opto_stds =        np.array([10.15, 10.15, 10.56, 10.80, 11.43, 11.83, 16.06])
mous_diff_means =       np.array([ 4.30,  4.02,  4.04,  4.48,  4.93,  3.88,  3.47])
mous_diff_stds =        np.array([ 7.94,  7.93,  8.30,  8.38,  8.96,  8.39, 10.27])
mous_norm_covs =        np.array([ 0.0427, -0.0387, -0.0466, -0.0370, -0.0051, -0.0971, -0.1277])

mous_base_means_err =   np.array([ 0.83,  0.96,  0.99,  1.02,  1.01,  1.29,  1.99])
mous_base_stds_err =    np.array([ 0.49,  0.79,  0.96,  0.81,  0.78,  1.08,  2.68])
mous_opto_means_err =   np.array([ 1.46,  1.46,  1.52,  1.55,  1.64,  1.70,  2.32])
mous_opto_stds_err =    np.array([ 1.60,  1.69,  1.86,  1.71,  1.64,  1.33,  2.56])
mous_diff_means_err =   np.array([ 1.15,  1.15,  1.21,  1.22,  1.30,  1.23,  1.51])
mous_diff_stds_err =    np.array([ 1.37,  1.33,  1.77,  1.54,  1.63,  1.56,  1.88])
mous_norm_covs_err =0.5*np.array([ 0.1254,  0.1669,  0.1576,  0.1608,  0.1438,  0.1892,  0.4293])

monk_base_means =       np.array([20.38, 43.32, 54.76, 64.54, 70.97, 72.69])
monk_base_stds =        np.array([17.06, 32.41, 38.93, 42.76, 45.17, 48.61])
monk_opto_means =       np.array([30.82, 44.93, 53.36, 60.46, 64.09, 68.87])
monk_opto_stds =        np.array([36.36, 42.87, 45.13, 49.31, 47.53, 52.24])
monk_diff_means =       np.array([10.44,  1.61, -1.41, -4.08, -6.88, -3.82])
monk_diff_stds =        np.array([37.77, 42.48, 42.24, 45.43, 41.78, 41.71])
monk_norm_covs =        np.array([-0.1456, -0.2999, -0.3792, -0.3831, -0.4664, -0.4226])

monk_base_means_err =   np.array([ 2.39,  4.49,  5.38,  5.90,  6.22,  6.69])
monk_base_stds_err =    np.array([ 2.29,  3.67,  4.48,  5.10,  5.61,  5.32])
monk_opto_means_err =   np.array([ 5.03,  5.86,  6.16,  6.74,  6.50,  7.15])
monk_opto_stds_err =    np.array([ 7.73,  6.47,  5.90,  6.20,  4.93,  4.74])
monk_diff_means_err =   np.array([ 5.28,  5.90,  5.84,  6.28,  5.75,  5.76])
monk_diff_stds_err =    np.array([ 8.36,  8.74,  8.01, 10.04,  8.51,  8.94])
monk_norm_covs_err =0.5*np.array([ 0.1075,  0.1354,  0.1579,  0.1496,  0.1717,  0.1665])

mous_nc = len(mous_base_means)
monk_nc = len(monk_base_means)

def gen_prms(seed):
    prm_dict = {}

    rng = np.random.default_rng(seed)
    prm_dict['K'] = 500
    prm_dict['SoriE'] = rng.uniform(15,45)
    prm_dict['SoriI'] = rng.uniform(15,45)
    prm_dict['SoriF'] = rng.uniform(15,45)
    prm_dict['J'] = 10**rng.uniform(-4,-3)
    prm_dict['beta'] = 10**rng.uniform(-1,0.7)
    prm_dict['gE'] = rng.uniform(3,7)
    prm_dict['gI'] = rng.uniform(2,prm_dict['gE']-0.5)
    prm_dict['hE'] = rng.uniform(0.2,3.5)
    prm_dict['hI'] = rng.uniform(0.01,1.5)
    prm_dict['L'] = 3.8+rng.uniform(0.2,4)
    prm_dict['CVL'] = 10**rng.uniform(-0.5,1.5)

    return prm_dict

def gen_disorder(prm_dict):
    K = prm_dict['K']
    SoriE = prm_dict['SoriE']
    SoriI = prm_dict['SoriI']
    SoriF = prm_dict['SoriF']
    J = prm_dict['J']
    beta = prm_dict['beta']
    gE = prm_dict['gE']
    gI = prm_dict['gI']
    hE = prm_dict['hE']
    hI = prm_dict['hI']
    L = prm_dict['L']
    CVL = prm_dict['CVL']

    WMat = J*np.array([[1,-gE],[1./beta,-gI/beta]],dtype=np.float32)
    HVec = K*J*np.array([hE,hI/beta],dtype=np.float32)

    net.set_seed(0)
    net.generate_disorder(WMat,np.array([[SoriE,SoriI],[SoriE,SoriI]]),HVec,SoriF*np.ones(2),K)
    net.generate_tensors()

    B = torch.where(net.C_conds[0],HVec[0],HVec[1])

    LAS = torch.zeros(net.N,dtype=torch.float32)
    sigma_l = np.sqrt(np.log(1+CVL**2))
    mu_l = np.log(1e-3*L)-sigma_l**2/2
    LAS_E = np.random.default_rng(0).lognormal(mu_l, sigma_l, net.NC[0]*net.Nloc).astype(np.float32)
    LAS[net.C_conds[0]] = torch.from_numpy(LAS_E)

    B = B.to(device)
    LAS = LAS.to(device)

    return net,net.M_torch,net.H_torch,B,LAS

resultsdir='./../results/'
name_results='results_ring_'+str(200+njob)+'.pkl'
this_results=resultsdir+name_results
print('Saving all results in '+  name_results)
print(' ')

if not os.path.exists(resultsdir):
    os.makedirs(resultsdir)

try:
    with open(this_results, 'rb') as handle:
        res_dict = pickle.load(handle)
    first_rep = max(list(res_dict.keys()))+1
except:
    res_dict = {}
    first_rep = 0

nrep = 10000
for idx_rep in range(first_rep,nrep):
    print('repetition number ',idx_rep)
    print('')

    start = time.process_time()

    seed = (njob*nrep+idx_rep)%2**32

    prm_dict = gen_prms(seed)
    net,M,H,B,LAS = gen_disorder(prm_dict)

    print("Disorder generation took ",time.process_time() - start," s")
    print('')

    start = time.process_time()

    all_base_means = np.zeros((len(aXs),len(bXs),len(eXs)))
    all_base_stds = np.zeros((len(aXs),len(bXs),len(eXs)))
    all_opto_means = np.zeros((len(aXs),len(bXs),len(eXs)))
    all_opto_stds = np.zeros((len(aXs),len(bXs),len(eXs)))
    all_diff_means = np.zeros((len(aXs),len(bXs),len(eXs)))
    all_diff_stds = np.zeros((len(aXs),len(bXs),len(eXs)))
    all_norm_covs = np.zeros((len(aXs),len(bXs),len(eXs)))

    vsm_base_means = np.zeros((len(aXs),len(bXs),len(eXs)))
    vsm_base_stds = np.zeros((len(aXs),len(bXs),len(eXs)))
    vsm_opto_means = np.zeros((len(aXs),len(bXs),len(eXs)))
    vsm_opto_stds = np.zeros((len(aXs),len(bXs),len(eXs)))
    vsm_diff_means = np.zeros((len(aXs),len(bXs),len(eXs)))
    vsm_diff_stds = np.zeros((len(aXs),len(bXs),len(eXs)))
    vsm_norm_covs = np.zeros((len(aXs),len(bXs),len(eXs)))

    timeouts = np.zeros((len(aXs),len(bXs),len(eXs))).astype(bool)

    for aX_idx,aX in enumerate(aXs):
        if aX_idx > 0 and np.all(timeouts[aX_idx-1,:,:]):
            all_base_means[aX_idx,:,:] = all_base_means[aX_idx-1,:,:] + aX
            all_base_stds[aX_idx,:,:] = all_base_stds[aX_idx-1,:,:] + aX
            all_opto_means[aX_idx,:,:] = all_opto_means[aX_idx-1,:,:] + aX
            all_opto_stds[aX_idx,:,:] = all_opto_stds[aX_idx-1,:,:] + aX
            all_diff_means[aX_idx,:,:] = all_diff_means[aX_idx-1,:,:] + aX
            all_diff_stds[aX_idx,:,:] = all_diff_stds[aX_idx-1,:,:] + aX
            all_norm_covs[aX_idx,:,:] = all_norm_covs[aX_idx-1,:,:] + aX
            vsm_base_means[aX_idx,:,:] = vsm_base_means[aX_idx-1,:,:] + aX
            vsm_base_stds[aX_idx,:,:] = vsm_base_stds[aX_idx-1,:,:] + aX
            vsm_opto_means[aX_idx,:,:] = vsm_opto_means[aX_idx-1,:,:] + aX
            vsm_opto_stds[aX_idx,:,:] = vsm_opto_stds[aX_idx-1,:,:] + aX
            vsm_diff_means[aX_idx,:,:] = vsm_diff_means[aX_idx-1,:,:] + aX
            vsm_diff_stds[aX_idx,:,:] = vsm_diff_stds[aX_idx-1,:,:] + aX
            vsm_norm_covs[aX_idx,:,:] = vsm_norm_covs[aX_idx-1,:,:] + aX
            timeouts[aX_idx,:,:] = True
            continue

        for bX_idx,bX in enumerate(bXs):
            if bX_idx > 0 and np.all(timeouts[aX_idx,bX_idx-1,:]):
                all_base_means[aX_idx,bX_idx,:] = all_base_means[aX_idx,bX_idx-1,:] + bX
                all_base_stds[aX_idx,bX_idx,:] = all_base_stds[aX_idx,bX_idx-1,:] + bX
                all_opto_means[aX_idx,bX_idx,:] = all_opto_means[aX_idx,bX_idx-1,:] + bX
                all_opto_stds[aX_idx,bX_idx,:] = all_opto_stds[aX_idx,bX_idx-1,:] + bX
                all_diff_means[aX_idx,bX_idx,:] = all_diff_means[aX_idx,bX_idx-1,:] + bX
                all_diff_stds[aX_idx,bX_idx,:] = all_diff_stds[aX_idx,bX_idx-1,:] + bX
                all_norm_covs[aX_idx,bX_idx,:] = all_norm_covs[aX_idx,bX_idx-1,:] + bX
                vsm_base_means[aX_idx,bX_idx,:] = vsm_base_means[aX_idx,bX_idx-1,:] + bX
                vsm_base_stds[aX_idx,bX_idx,:] = vsm_base_stds[aX_idx,bX_idx-1,:] + bX
                vsm_opto_means[aX_idx,bX_idx,:] = vsm_opto_means[aX_idx,bX_idx-1,:] + bX
                vsm_opto_stds[aX_idx,bX_idx,:] = vsm_opto_stds[aX_idx,bX_idx-1,:] + bX
                vsm_diff_means[aX_idx,bX_idx,:] = vsm_diff_means[aX_idx,bX_idx-1,:] + bX
                vsm_diff_stds[aX_idx,bX_idx,:] = vsm_diff_stds[aX_idx,bX_idx-1,:] + bX
                vsm_norm_covs[aX_idx,bX_idx,:] = vsm_norm_covs[aX_idx,bX_idx-1,:] + bX
                timeouts[aX_idx,bX_idx,:] = True
                continue

            for eX_idx,eX in enumerate(eXs):
                if eX_idx > 0 and timeouts[aX_idx,bX_idx,eX_idx-1]:
                    all_base_means[aX_idx,bX_idx,eX_idx] = all_base_means[aX_idx,bX_idx,eX_idx-1] + eX
                    all_base_stds[aX_idx,bX_idx,eX_idx] = all_base_stds[aX_idx,bX_idx,eX_idx-1] + eX
                    all_opto_means[aX_idx,bX_idx,eX_idx] = all_opto_means[aX_idx,bX_idx,eX_idx-1] + eX
                    all_opto_stds[aX_idx,bX_idx,eX_idx] = all_opto_stds[aX_idx,bX_idx,eX_idx-1] + eX
                    all_diff_means[aX_idx,bX_idx,eX_idx] = all_diff_means[aX_idx,bX_idx,eX_idx-1] + eX
                    all_diff_stds[aX_idx,bX_idx,eX_idx] = all_diff_stds[aX_idx,bX_idx,eX_idx-1] + eX
                    all_norm_covs[aX_idx,bX_idx,eX_idx] = all_norm_covs[aX_idx,bX_idx,eX_idx-1] + eX
                    vsm_base_means[aX_idx,bX_idx,eX_idx] = vsm_base_means[aX_idx,bX_idx,eX_idx-1] + eX
                    vsm_base_stds[aX_idx,bX_idx,eX_idx] = vsm_base_stds[aX_idx,bX_idx,eX_idx-1] + eX
                    vsm_opto_means[aX_idx,bX_idx,eX_idx] = vsm_opto_means[aX_idx,bX_idx,eX_idx-1] + eX
                    vsm_opto_stds[aX_idx,bX_idx,eX_idx] = vsm_opto_stds[aX_idx,bX_idx,eX_idx-1] + eX
                    vsm_diff_means[aX_idx,bX_idx,eX_idx] = vsm_diff_means[aX_idx,bX_idx,eX_idx-1] + eX
                    vsm_diff_stds[aX_idx,bX_idx,eX_idx] = vsm_diff_stds[aX_idx,bX_idx,eX_idx-1] + eX
                    vsm_norm_covs[aX_idx,bX_idx,eX_idx] = vsm_norm_covs[aX_idx,bX_idx,eX_idx-1] + eX
                    timeouts[aX_idx,bX_idx,eX_idx] = True
                    continue

                base_sol,base_timeout = integ.sim_dyn_tensor(ri,T,0.0,M,(bX*B+aX*H)*eps[eX_idx],LAS,net.C_conds[0],
                                                             mult_tau=True,max_min=15)
                opto_sol,opto_timeout = integ.sim_dyn_tensor(ri,T,1.0,M,(bX*B+aX*H)*eps[eX_idx],LAS,net.C_conds[0],
                                                             mult_tau=True,max_min=15)
                diff_sol = opto_sol - base_sol
                timeout = base_timeout or opto_timeout

                base_rates = np.mean(base_sol[:,mask_time].cpu().numpy(),axis=1)
                opto_rates = np.mean(opto_sol[:,mask_time].cpu().numpy(),axis=1)
                diff_rates = np.mean(diff_sol[:,mask_time].cpu().numpy(),axis=1)

                all_base_means[aX_idx,bX_idx,eX_idx] = np.mean(base_rates)
                all_base_stds[aX_idx,bX_idx,eX_idx] = np.std(base_rates)
                all_opto_means[aX_idx,bX_idx,eX_idx] = np.mean(opto_rates)
                all_opto_stds[aX_idx,bX_idx,eX_idx] = np.std(opto_rates)
                all_diff_means[aX_idx,bX_idx,eX_idx] = np.mean(diff_rates)
                all_diff_stds[aX_idx,bX_idx,eX_idx] = np.std(diff_rates)
                all_norm_covs[aX_idx,bX_idx,eX_idx] = np.cov(base_rates,
                    diff_rates)[0,1] / all_diff_stds[aX_idx,bX_idx,eX_idx]**2

                vsm_base_means[aX_idx,bX_idx,eX_idx] = np.mean(base_rates[net.get_oriented_neurons()])
                vsm_base_stds[aX_idx,bX_idx,eX_idx] = np.std(base_rates[net.get_oriented_neurons()])
                vsm_opto_means[aX_idx,bX_idx,eX_idx] = np.mean(opto_rates[net.get_oriented_neurons()])
                vsm_opto_stds[aX_idx,bX_idx,eX_idx] = np.std(opto_rates[net.get_oriented_neurons()])
                vsm_diff_means[aX_idx,bX_idx,eX_idx] = np.mean(diff_rates[net.get_oriented_neurons()])
                vsm_diff_stds[aX_idx,bX_idx,eX_idx] = np.std(diff_rates[net.get_oriented_neurons()])
                vsm_norm_covs[aX_idx,bX_idx,eX_idx] = np.cov(base_rates[net.get_oriented_neurons()],
                    diff_rates[net.get_oriented_neurons()])[0,1] / vsm_diff_stds[aX_idx,bX_idx,eX_idx]**2
                
                if timeout:
                    all_norm_covs[aX_idx,bX_idx,eX_idx] = 1000
                    vsm_norm_covs[aX_idx,bX_idx,eX_idx] = 1000
                
                timeouts[aX_idx,bX_idx,eX_idx] = timeout

    print("Simulating inputs took ",time.process_time() - start," s")
    print('')

    start = time.process_time()

    this_res_dict = {}
    this_res_dict['prms'] = prm_dict
    this_res_dict['all_base_means'] = all_base_means
    this_res_dict['all_base_stds'] = all_base_stds
    this_res_dict['all_opto_means'] = all_opto_means
    this_res_dict['all_opto_stds'] = all_opto_stds
    this_res_dict['all_diff_means'] = all_diff_means
    this_res_dict['all_diff_stds'] = all_diff_stds
    this_res_dict['all_norm_covs'] = all_norm_covs
    this_res_dict['vsm_base_means'] = vsm_base_means
    this_res_dict['vsm_base_stds'] = vsm_base_stds
    this_res_dict['vsm_opto_means'] = vsm_opto_means
    this_res_dict['vsm_opto_stds'] = vsm_opto_stds
    this_res_dict['vsm_diff_means'] = vsm_diff_means
    this_res_dict['vsm_diff_stds'] = vsm_diff_stds
    this_res_dict['vsm_norm_covs'] = vsm_norm_covs

    all_base_mean_itp = RegularGridInterpolator((aXs,bXs,eXs), all_base_means)
    all_base_std_itp = RegularGridInterpolator((aXs,bXs,eXs), all_base_stds)
    all_opto_mean_itp = RegularGridInterpolator((aXs,bXs,eXs), all_opto_means)
    all_opto_std_itp = RegularGridInterpolator((aXs,bXs,eXs), all_opto_stds)
    all_diff_mean_itp = RegularGridInterpolator((aXs,bXs,eXs), all_diff_means)
    all_diff_std_itp = RegularGridInterpolator((aXs,bXs,eXs), all_diff_stds)
    all_norm_cov_itp = RegularGridInterpolator((aXs,bXs,eXs), all_norm_covs)

    vsm_base_mean_itp = RegularGridInterpolator((aXs,bXs,eXs), vsm_base_means)
    vsm_base_std_itp = RegularGridInterpolator((aXs,bXs,eXs), vsm_base_stds)
    vsm_opto_mean_itp = RegularGridInterpolator((aXs,bXs,eXs), vsm_opto_means)
    vsm_opto_std_itp = RegularGridInterpolator((aXs,bXs,eXs), vsm_opto_stds)
    vsm_diff_mean_itp = RegularGridInterpolator((aXs,bXs,eXs), vsm_diff_means)
    vsm_diff_std_itp = RegularGridInterpolator((aXs,bXs,eXs), vsm_diff_stds)
    vsm_norm_cov_itp = RegularGridInterpolator((aXs,bXs,eXs), vsm_norm_covs)

    def fit_best_mous_inputs(eX):
        def residuals(x):
            this_bX = x[0]
            this_aXs = np.concatenate(([0],x[1:]))
            pred_base_means = all_base_mean_itp(np.vstack((this_aXs,this_bX*np.ones(mous_nc),eX*np.ones(mous_nc))).T)
            pred_base_stds = all_base_std_itp(np.vstack((this_aXs,this_bX*np.ones(mous_nc),eX*np.ones(mous_nc))).T)
            pred_opto_means = all_opto_mean_itp(np.vstack((this_aXs,this_bX*np.ones(mous_nc),eX*np.ones(mous_nc))).T)
            pred_opto_stds = all_opto_std_itp(np.vstack((this_aXs,this_bX*np.ones(mous_nc),eX*np.ones(mous_nc))).T)
            pred_diff_means = all_diff_mean_itp(np.vstack((this_aXs,this_bX*np.ones(mous_nc),eX*np.ones(mous_nc))).T)
            pred_diff_stds = all_diff_std_itp(np.vstack((this_aXs,this_bX*np.ones(mous_nc),eX*np.ones(mous_nc))).T)
            pred_norm_covs = all_norm_cov_itp(np.vstack((this_aXs,this_bX*np.ones(mous_nc),eX*np.ones(mous_nc))).T)
            res = np.array([(pred_base_means-mous_base_means)/mous_base_means_err,
                            (pred_base_stds-mous_base_stds)/mous_base_stds_err,
                            (pred_opto_means-mous_opto_means)/mous_opto_means_err,
                            (pred_opto_stds-mous_opto_stds)/mous_opto_stds_err,
                            (pred_diff_means-mous_diff_means)/mous_diff_means_err,
                            (pred_diff_stds-mous_diff_stds)/mous_diff_stds_err,
                            (pred_norm_covs-mous_norm_covs)/mous_norm_covs_err])
            return res.ravel()
        xmin = np.concatenate(([bXs[ 0]],aXs[ 0]*np.ones(mous_nc-1)))
        xmax = np.concatenate(([bXs[-1]],aXs[-1]*np.ones(mous_nc-1)))
        x0 = np.concatenate(([2],np.linspace(1,12,mous_nc-1)))
        results = least_squares(residuals,x0,bounds=(xmin,xmax))
        this_bX = results.x[0]
        this_aXs = np.concatenate(([0],results.x[1:]))
        return (this_bX,this_aXs,all_base_mean_itp(np.vstack((this_aXs,this_bX*np.ones(mous_nc),eX*np.ones(mous_nc))).T),
                all_base_std_itp(np.vstack((this_aXs,this_bX*np.ones(mous_nc),eX*np.ones(mous_nc))).T),
                all_opto_mean_itp(np.vstack((this_aXs,this_bX*np.ones(mous_nc),eX*np.ones(mous_nc))).T),
                all_opto_std_itp(np.vstack((this_aXs,this_bX*np.ones(mous_nc),eX*np.ones(mous_nc))).T),
                all_diff_mean_itp(np.vstack((this_aXs,this_bX*np.ones(mous_nc),eX*np.ones(mous_nc))).T),
                all_diff_std_itp(np.vstack((this_aXs,this_bX*np.ones(mous_nc),eX*np.ones(mous_nc))).T),
                all_norm_cov_itp(np.vstack((this_aXs,this_bX*np.ones(mous_nc),eX*np.ones(mous_nc))).T),results.cost)

    def fit_best_mous_input_var():
        def residuals(x):
            _,_,_,_,_,_,_,_,_,cost = fit_best_mous_inputs(x[0])
            return [cost]
        xmin,xmax = (eXs[0]),(eXs[-1])
        x0 = np.array([0.2])
        results = least_squares(residuals,x0,bounds=(xmin,xmax))
        return (results.x[0],*fit_best_mous_inputs(results.x[0]))

    def fit_best_monk_inputs(eX):
        def residuals(x):
            this_bX = x[0]
            this_aXs = np.concatenate(([0],x[1:]))
            pred_base_means = vsm_base_mean_itp(np.vstack((this_aXs,this_bX*np.ones(monk_nc),eX*np.ones(monk_nc))).T)
            pred_base_stds = vsm_base_std_itp(np.vstack((this_aXs,this_bX*np.ones(monk_nc),eX*np.ones(monk_nc))).T)
            pred_opto_means = vsm_opto_mean_itp(np.vstack((this_aXs,this_bX*np.ones(monk_nc),eX*np.ones(monk_nc))).T)
            pred_opto_stds = vsm_opto_std_itp(np.vstack((this_aXs,this_bX*np.ones(monk_nc),eX*np.ones(monk_nc))).T)
            pred_diff_means = vsm_diff_mean_itp(np.vstack((this_aXs,this_bX*np.ones(monk_nc),eX*np.ones(monk_nc))).T)
            pred_diff_stds = vsm_diff_std_itp(np.vstack((this_aXs,this_bX*np.ones(monk_nc),eX*np.ones(monk_nc))).T)
            pred_norm_covs = vsm_norm_cov_itp(np.vstack((this_aXs,this_bX*np.ones(monk_nc),eX*np.ones(monk_nc))).T)
            res = np.array([(pred_base_means-monk_base_means)/monk_base_means_err,
                            (pred_base_stds-monk_base_stds)/monk_base_stds_err,
                            (pred_opto_means-monk_opto_means)/monk_opto_means_err,
                            (pred_opto_stds-monk_opto_stds)/monk_opto_stds_err,
                            (pred_diff_means-monk_diff_means)/monk_diff_means_err,
                            (pred_diff_stds-monk_diff_stds)/monk_diff_stds_err,
                            (pred_norm_covs-monk_norm_covs)/monk_norm_covs_err])
            return res.ravel()
        xmin = np.concatenate(([bXs[ 0]],aXs[ 0]*np.ones(monk_nc-1)))
        xmax = np.concatenate(([bXs[-1]],aXs[-1]*np.ones(monk_nc-1)))
        x0 = np.concatenate(([2],np.linspace(1,12,monk_nc-1)))
        results = least_squares(residuals,x0,bounds=(xmin,xmax))
        this_bX = results.x[0]
        this_aXs = np.concatenate(([0],results.x[1:]))
        return (this_bX,this_aXs,vsm_base_mean_itp(np.vstack((this_aXs,this_bX*np.ones(monk_nc),eX*np.ones(monk_nc))).T),
                vsm_base_std_itp(np.vstack((this_aXs,this_bX*np.ones(monk_nc),eX*np.ones(monk_nc))).T),
                vsm_opto_mean_itp(np.vstack((this_aXs,this_bX*np.ones(monk_nc),eX*np.ones(monk_nc))).T),
                vsm_opto_std_itp(np.vstack((this_aXs,this_bX*np.ones(monk_nc),eX*np.ones(monk_nc))).T),
                vsm_diff_mean_itp(np.vstack((this_aXs,this_bX*np.ones(monk_nc),eX*np.ones(monk_nc))).T),
                vsm_diff_std_itp(np.vstack((this_aXs,this_bX*np.ones(monk_nc),eX*np.ones(monk_nc))).T),
                vsm_norm_cov_itp(np.vstack((this_aXs,this_bX*np.ones(monk_nc),eX*np.ones(monk_nc))).T),results.cost)

    def fit_best_monk_input_var():
        def residuals(x):
            _,_,_,_,_,_,_,_,_,cost = fit_best_monk_inputs(x[0])
            return [cost]
        xmin,xmax = (eXs[0]),(eXs[-1])
        x0 = np.array([0.2])
        results = least_squares(residuals,x0,bounds=(xmin,xmax))
        return (results.x[0],*fit_best_monk_inputs(results.x[0]))

    best_mous_eX,best_mous_bX,best_mous_aXs,best_mous_base_means,best_mous_base_stds,\
        best_mous_opto_means,best_mous_opto_stds,best_mous_diff_means,best_mous_diff_stds,\
        best_mous_norm_covs,best_mous_cost = fit_best_mous_input_var()

    best_monk_eX,best_monk_bX,best_monk_aXs,best_monk_base_means,best_monk_base_stds,\
        best_monk_opto_means,best_monk_opto_stds,best_monk_diff_means,best_monk_diff_stds,\
        best_monk_norm_covs,best_monk_cost = fit_best_monk_input_var()

    this_res_dict['best_mous_eX'] = best_mous_eX
    this_res_dict['best_mous_bX'] = best_mous_bX
    this_res_dict['best_mous_aXs'] = best_mous_aXs
    this_res_dict['best_mous_base_means'] = best_mous_base_means
    this_res_dict['best_mous_base_stds'] = best_mous_base_stds
    this_res_dict['best_mous_opto_means'] = best_mous_opto_means
    this_res_dict['best_mous_opto_stds'] = best_mous_opto_stds
    this_res_dict['best_mous_diff_means'] = best_mous_diff_means
    this_res_dict['best_mous_diff_stds'] = best_mous_diff_stds
    this_res_dict['best_mous_norm_covs'] = best_mous_norm_covs
    this_res_dict['best_mous_cost'] = best_mous_cost

    this_res_dict['best_monk_eX'] = best_monk_eX
    this_res_dict['best_monk_bX'] = best_monk_bX
    this_res_dict['best_monk_aXs'] = best_monk_aXs
    this_res_dict['best_monk_base_means'] = best_monk_base_means
    this_res_dict['best_monk_base_stds'] = best_monk_base_stds
    this_res_dict['best_monk_opto_means'] = best_monk_opto_means
    this_res_dict['best_monk_opto_stds'] = best_monk_opto_stds
    this_res_dict['best_monk_diff_means'] = best_monk_diff_means
    this_res_dict['best_monk_diff_stds'] = best_monk_diff_stds
    this_res_dict['best_monk_norm_covs'] = best_monk_norm_covs
    this_res_dict['best_monk_cost'] = best_monk_cost

    print("Fitting best inputs took ",time.process_time() - start," s")
    print('')
    print(prm_dict)
    print(best_mous_cost)
    print(best_monk_cost)
    print('')

    res_dict[idx_rep] = this_res_dict

    with open(this_results, 'wb') as handle:
        pickle.dump(res_dict,handle)
