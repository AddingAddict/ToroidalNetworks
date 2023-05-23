import argparse
import os
import pickle
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares
import time

import spat_ring_network as r_network
import spat_snp_network as m_network
import sim_util as su
import ricciardi as ric
import integrate as integ

parser = argparse.ArgumentParser(description=('This python script takes results from sampled spatial model parameters, '
    'trains a net to interpolate the results, and finds parameters that best fit the experimental results'))

parser.add_argument('--njob', '-n',  help='which number job', type=int, default=0)
args = vars(parser.parse_args())
print(parser.parse_args())
njob= args['njob']

ri = ric.Ricciardi()
ri.set_up_nonlinearity('./../scripts/phi_int')

NtE = 100
T = np.linspace(0,NtE*ri.tE,round(NtE*ri.tE/(ri.tI/3))+1)
mask_time = T>(NtE/2*ri.tE)

Ls = np.arange(0.5,8.0+0.5,0.5)
CVLs = 10**np.arange(-1.0,1.0+0.25,0.25)

net = m_network.network(seed=0,NC=[4,1],Nrf=48,Nori=9,Lrf=80)

LAMs = np.zeros((len(CVLs),net.N))
for CVL_idx,CVL in enumerate(CVLs):
    sigma_l = np.sqrt(np.log(1+CVL**2))
    mu_l = np.log(1e-3)-sigma_l**2/2
    LAMs[CVL_idx,net.C_all[0]] = np.random.lognormal(mu_l, sigma_l, net.NC[0]*net.Nloc)

data_opto_means = np.array([10.56, 10.79, 11.25, 12.19, 12.96, 14.89, 20.25])
data_opto_stds =  np.array([10.20, 10.20, 10.62, 10.85, 11.47, 11.84, 16.12])
data_diff_means = np.array([ 4.32,  4.05,  4.07,  4.51,  4.95,  3.90,  3.48])
data_diff_stds =  np.array([ 7.98,  7.96,  8.34,  8.42,  8.99,  8.41, 10.30])
data_norm_covs =  np.array([0.04409,-0.03578,-0.04356,-0.03427,-0.00142,-0.09287,-0.12237])

data_opto_means_err = np.array([1.46, 1.47, 1.53, 1.56, 1.65, 1.69, 2.32])
data_opto_stds_err =  np.array([1.63, 1.71, 1.88, 1.73, 1.66, 1.33, 2.57])
data_diff_means_err = np.array([1.15, 1.15, 1.22, 1.22, 1.31, 1.22, 1.50])
data_diff_stds_err =  np.array([1.37, 1.32, 1.76, 1.53, 1.61, 1.53, 1.86])
data_norm_covs_err =  np.array([0.1270, 0.1679, 0.1588, 0.1608, 0.1424, 0.1883, 0.4295])

nc = len(data_opto_means)

def gen_disorder(prm_dict,eX):
    SrfE = prm_dict['SrfE']
    SrfI = prm_dict['SrfI']
    SrfF = prm_dict['SrfF']
    SoriE = prm_dict['SoriE']
    SoriI = prm_dict['SoriI']
    SoriF = prm_dict['SoriF']
    fEE = prm_dict['fEE']
    fEI = prm_dict['fEI']
    fIE = prm_dict['fIE']
    fII = prm_dict['fII']
    fFI = prm_dict['fFI']

    net.set_seed(0)
    net.generate_disorder(1e-3*np.array([[fEE*0.1,-fEI*0.8],[fIE*0.3,-fII*0.7]]),
                      np.array([[SrfE,SrfI],[SrfE,SrfI]]),np.array([[SoriE,SoriI],[SoriE,SoriI]]),
                      500*1e-3*np.array([0.25,fFI*0.2]),
                      SrfF*np.ones(2),SoriF*np.ones(2),500)
    B = np.zeros_like(net.H)
    B[net.C_all[0]] = 500*1e-3*0.25
    B[net.C_all[1]] = 500*1e-3*fFI*0.2

    eps = np.zeros_like(net.H)
    if eX == 0.0:
        eps = np.ones(net.N)
    else:
        shape = 1/eX**2
        scale = 1/shape
        eps = np.random.default_rng(0).gamma(shape,scale=scale,size=net.N)

    return net,net.M,net.H,B,eps

resultsdir='./../results/'
base_results='results_base_'+str(njob)+'.pkl'
name_results='results_opto_'+str(njob)+'.pkl'
this_base_results=resultsdir+base_results
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

try:
    with open(this_base_results, 'rb') as handle:
        base_res_dict = pickle.load(handle)
    last_rep = max(list(base_res_dict.keys()))+1
except:
    base_res_dict = {}
    last_rep = 0

print((first_rep,last_rep))

nrep = 10000
for idx_rep in range(first_rep,last_rep):
    print('repetition number ',idx_rep)
    print('')

    start = time.process_time()

    prm_dict = base_res_dict[idx_rep]['prms']
    eX = base_res_dict[idx_rep]['best_eX']
    bX = base_res_dict[idx_rep]['best_bX']
    aXs = base_res_dict[idx_rep]['best_aXs']

    net,M,H,B,eps = gen_disorder(prm_dict,eX)

    print("Disorder generation took ",time.process_time() - start," s")
    print('')

    start = time.process_time()

    opto_means = np.zeros((len(Ls),len(CVLs),len(aXs)))
    opto_stds = np.zeros((len(Ls),len(CVLs),len(aXs)))
    diff_means = np.zeros((len(Ls),len(CVLs),len(aXs)))
    diff_stds = np.zeros((len(Ls),len(CVLs),len(aXs)))
    norm_covs = np.zeros((len(Ls),len(CVLs),len(aXs)))

    for aX_idx,aX in enumerate(aXs):
        base_sol,_ = integ.sim_dyn(ri,T,0.0,M,(bX*B+aX*H)*eps,H,net.C_all[0],net.C_all[1],
                              mult_tau=True,max_min=30)
        for L_idx,L in enumerate(Ls):
            for CVL_idx,CVL in enumerate(CVLs):
                LAM = LAMs[CVL_idx]
                opto_sol,_ = integ.sim_dyn(ri,T,L,M,(bX*B+aX*H)*eps,LAM,net.C_all[0],net.C_all[1],
                                      mult_tau=True,max_min=30)
                diff_sol = opto_sol - base_sol
                opto_means[L_idx,CVL_idx,aX_idx] = np.mean(opto_sol[net.get_centered_neurons(),-1])
                opto_stds[L_idx,CVL_idx,aX_idx] = np.std(opto_sol[net.get_centered_neurons(),-1])
                diff_means[L_idx,CVL_idx,aX_idx] = np.mean(diff_sol[net.get_centered_neurons(),-1])
                diff_stds[L_idx,CVL_idx,aX_idx] = np.std(diff_sol[net.get_centered_neurons(),-1])
                norm_covs[L_idx,CVL_idx,aX_idx] = np.cov(base_sol[net.get_centered_neurons(),-1],
                    diff_sol[net.get_centered_neurons(),-1])[0,1] / diff_stds[L_idx,CVL_idx,aX_idx]**2

    print("Simulating inputs took ",time.process_time() - start," s")
    print('')

    start = time.process_time()

    this_res_dict = {}
    this_res_dict['prms'] = prm_dict
    this_res_dict['opto_means'] = opto_means
    this_res_dict['opto_stds'] = opto_stds
    this_res_dict['diff_means'] = diff_means
    this_res_dict['diff_stds'] = diff_stds
    this_res_dict['norm_covs'] = norm_covs

    opto_mean_itp = RegularGridInterpolator((Ls,CVLs,aXs), opto_means)
    opto_std_itp = RegularGridInterpolator((Ls,CVLs,aXs), opto_stds)
    diff_mean_itp = RegularGridInterpolator((Ls,CVLs,aXs), diff_means)
    diff_std_itp = RegularGridInterpolator((Ls,CVLs,aXs), diff_stds)
    norm_cov_itp = RegularGridInterpolator((Ls,CVLs,aXs), norm_covs)

    def fit_best_opto_inputs():
        def residuals(x):
            this_L = x[0]
            this_CVL = x[1]
            pred_opto_means = opto_mean_itp(np.vstack((this_L*np.ones(nc),this_CVL*np.ones(nc),aXs)).T)
            pred_opto_stds = opto_std_itp(np.vstack((this_L*np.ones(nc),this_CVL*np.ones(nc),aXs)).T)
            pred_diff_means = diff_mean_itp(np.vstack((this_L*np.ones(nc),this_CVL*np.ones(nc),aXs)).T)
            pred_diff_stds = diff_std_itp(np.vstack((this_L*np.ones(nc),this_CVL*np.ones(nc),aXs)).T)
            pred_norm_covs = norm_cov_itp(np.vstack((this_L*np.ones(nc),this_CVL*np.ones(nc),aXs)).T)
            res = np.array([(pred_opto_means-data_opto_means)/data_opto_means_err,
                            (pred_opto_stds-data_opto_stds)/data_opto_stds_err,
                            (pred_diff_means-data_diff_means)/data_diff_means_err,
                            (pred_diff_stds-data_diff_stds)/data_diff_stds_err,
                            (pred_norm_covs-data_norm_covs)/data_norm_covs_err])
            return res.ravel()
        xmin = (Ls[ 0],CVLs[ 0])
        xmax = (Ls[-1],CVLs[-1])
        x0 = [5,5]
        results = least_squares(residuals,x0,bounds=(xmin,xmax))
        this_L = results.x[0]
        this_CVL = results.x[1]
        return (this_L,this_CVL,opto_mean_itp(np.vstack((this_L*np.ones(nc),this_CVL*np.ones(nc),aXs)).T),
                opto_std_itp(np.vstack((this_L*np.ones(nc),this_CVL*np.ones(nc),aXs)).T),
                diff_mean_itp(np.vstack((this_L*np.ones(nc),this_CVL*np.ones(nc),aXs)).T),
                diff_std_itp(np.vstack((this_L*np.ones(nc),this_CVL*np.ones(nc),aXs)).T),
                norm_cov_itp(np.vstack((this_L*np.ones(nc),this_CVL*np.ones(nc),aXs)).T),results.cost)

    best_L,best_CVL,best_opto_means,best_opto_stds,best_diff_means,best_diff_stds,best_norm_covs,best_cost = fit_best_opto_inputs()
    this_res_dict['best_L'] = best_L
    this_res_dict['best_CVL'] = best_CVL
    this_res_dict['best_opto_means'] = best_opto_means
    this_res_dict['best_opto_stds'] = best_opto_stds
    this_res_dict['best_diff_means'] = best_diff_means
    this_res_dict['best_diff_stds'] = best_diff_stds
    this_res_dict['best_norm_covs'] = best_norm_covs
    this_res_dict['best_cost'] = best_cost

    print("Fitting best inputs took ",time.process_time() - start," s")
    print('')
    print(prm_dict)
    print(best_cost)
    print('')

    res_dict[idx_rep] = this_res_dict

    with open(this_results, 'wb') as handle:
        pickle.dump(res_dict,handle)
