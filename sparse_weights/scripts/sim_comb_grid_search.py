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

NtE = 75
T = np.linspace(0,NtE*ri.tE,round(NtE*ri.tE/(ri.tI/3))+1)
mask_time = T>(NtE/2*ri.tE)

aXs = np.arange(0,24+3,3)
bXs = np.arange(1,11+2,2)
eXs = np.arange(0,0.6+0.05,0.05)

net = m_network.network(seed=0,NC=[4,1],Nrf=48,Nori=9,Lrf=80)

eps = np.zeros((len(eXs),net.N))
for eX_idx,eX in enumerate(eXs):
    if eX == 0.0:
        eps[eX_idx] = np.ones(net.N)
    else:
        shape = 1/eX**2
        scale = 1/shape
        eps[eX_idx] = np.random.default_rng(0).gamma(shape,scale=scale,size=net.N)

data_base_means = np.array([ 6.22,  6.72,  7.17,  7.67,  8.00, 10.97, 16.7 ])
data_base_stds =  np.array([ 5.79,  6.64,  6.93,  7.15,  7.07,  8.98, 13.6 ])
data_opto_means = np.array([10.56, 10.79, 11.25, 12.19, 12.96, 14.89, 20.25])
data_opto_stds =  np.array([10.20, 10.20, 10.62, 10.85, 11.47, 11.84, 16.12])
data_diff_means = np.array([ 4.32,  4.05,  4.07,  4.51,  4.95,  3.90,  3.48])
data_diff_stds =  np.array([ 7.98,  7.96,  8.34,  8.42,  8.99,  8.41, 10.30])
data_norm_covs =  np.array([0.04409,-0.03578,-0.04356,-0.03427,-0.00142,-0.09287,-0.12237])

data_base_means_err = np.array([0.83, 0.96, 1.00, 1.03, 1.02, 1.30, 2.00])
data_base_stds_err =  np.array([0.48, 0.78, 0.96, 0.81, 0.78, 1.08, 2.63])
data_opto_means_err = np.array([1.46, 1.47, 1.53, 1.56, 1.65, 1.69, 2.32])
data_opto_stds_err =  np.array([1.63, 1.71, 1.88, 1.73, 1.66, 1.33, 2.57])
data_diff_means_err = np.array([1.15, 1.15, 1.22, 1.22, 1.31, 1.22, 1.50])
data_diff_stds_err =  np.array([1.37, 1.32, 1.76, 1.53, 1.61, 1.53, 1.86])
data_norm_covs_err =  np.array([0.1270, 0.1679, 0.1588, 0.1608, 0.1424, 0.1883, 0.4295])

nc = len(data_base_means)

def gen_prms(seed):
    prm_dict = {}

    rng = np.random.default_rng(seed)
    prm_dict['SrfE'] = rng.uniform(5,20)
    prm_dict['SrfI'] = rng.uniform(5,20)
    prm_dict['SrfF'] = 30#rng.uniform(5,20)
    prm_dict['SoriE'] = rng.uniform(15,45)
    prm_dict['SoriI'] = rng.uniform(15,45)
    prm_dict['SoriF'] = rng.uniform(15,45)
    prm_dict['fEE'] = rng.uniform(0.6,1.4)
    prm_dict['fEI'] = rng.uniform(0.6,1.4)
    prm_dict['fIE'] = rng.uniform(0.6,1.4)
    prm_dict['fII'] = rng.uniform(0.6,1.4)
    prm_dict['fFI'] = rng.uniform(0.6,1.4)
    prm_dict['L'] = rng.uniform(0.5,8)
    prm_dict['CVL'] = 10**rng.uniform(-1.0,1.0)

    return prm_dict

def gen_disorder(prm_dict):
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
    L = prm_dict['L']
    CVL = prm_dict['CVL']

    net.set_seed(0)
    net.generate_disorder(1e-3*np.array([[fEE*0.1,-fEI*0.8],[fIE*0.3,-fII*0.7]]),
                      np.array([[SrfE,SrfI],[SrfE,SrfI]]),np.array([[SoriE,SoriI],[SoriE,SoriI]]),
                      500*1e-3*np.array([0.25,fFI*0.2]),
                      SrfF*np.ones(2),SoriF*np.ones(2),500)
    B = np.zeros_like(net.H)
    B[net.C_all[0]] = 500*1e-3*0.25
    B[net.C_all[1]] = 500*1e-3*fFI*0.2

    LAS = np.zeros_like(net.H)
    sigma_l = np.sqrt(np.log(1+CVL**2))
    mu_l = np.log(1e-3*L)-sigma_l**2/2
    LAS[net.C_all[0]] = np.random.default_rng(0).lognormal(mu_l, sigma_l, net.NC[0]*net.Nloc)

    return net,net.M,net.H,B,LAS

resultsdir='./../results/'
name_results='results_comb_'+str(njob)+'.pkl'
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

    base_means = np.zeros((len(aXs),len(bXs),len(eXs)))
    base_stds = np.zeros((len(aXs),len(bXs),len(eXs)))
    opto_means = np.zeros((len(aXs),len(bXs),len(eXs)))
    opto_stds = np.zeros((len(aXs),len(bXs),len(eXs)))
    diff_means = np.zeros((len(aXs),len(bXs),len(eXs)))
    diff_stds = np.zeros((len(aXs),len(bXs),len(eXs)))
    norm_covs = np.zeros((len(aXs),len(bXs),len(eXs)))

    for aX_idx,aX in enumerate(aXs):
        for bX_idx,bX in enumerate(bXs):
            for eX_idx,eX in enumerate(eXs):
                base_sol,_ = integ.sim_dyn(ri,T,0.0,M,(bX*B+aX*H)*eps[eX_idx],LAS,net.C_all[0],net.C_all[1],
                                      mult_tau=True,max_min=20)
                opto_sol,_ = integ.sim_dyn(ri,T,1.0,M,(bX*B+aX*H)*eps[eX_idx],LAS,net.C_all[0],net.C_all[1],
                                      mult_tau=True,max_min=20)
                diff_sol = opto_sol - base_sol
                base_rates = np.mean(base_sol[:,mask_time],axis=1)
                opto_rates = np.mean(opto_sol[:,mask_time],axis=1)
                diff_rates = np.mean(diff_sol[:,mask_time],axis=1)
                base_means[aX_idx,bX_idx,eX_idx] = np.mean(base_rates[net.get_centered_neurons()])
                base_stds[aX_idx,bX_idx,eX_idx] = np.std(base_rates[net.get_centered_neurons()])
                opto_means[aX_idx,bX_idx,eX_idx] = np.mean(opto_rates[net.get_centered_neurons()])
                opto_stds[aX_idx,bX_idx,eX_idx] = np.std(opto_rates[net.get_centered_neurons()])
                diff_means[aX_idx,bX_idx,eX_idx] = np.mean(diff_rates[net.get_centered_neurons()])
                diff_stds[aX_idx,bX_idx,eX_idx] = np.std(diff_rates[net.get_centered_neurons()])
                norm_covs[aX_idx,bX_idx,eX_idx] = np.cov(base_rates[net.get_centered_neurons()],
                    diff_sol[net.get_centered_neurons(),-1])[0,1] / diff_stds[aX_idx,bX_idx,eX_idx]**2

    print("Simulating inputs took ",time.process_time() - start," s")
    print('')

    start = time.process_time()

    this_res_dict = {}
    this_res_dict['prms'] = prm_dict
    this_res_dict['base_means'] = base_means
    this_res_dict['base_stds'] = base_stds
    this_res_dict['opto_means'] = opto_means
    this_res_dict['opto_stds'] = opto_stds
    this_res_dict['diff_means'] = diff_means
    this_res_dict['diff_stds'] = diff_stds
    this_res_dict['norm_covs'] = norm_covs

    base_mean_itp = RegularGridInterpolator((aXs,bXs,eXs), base_means)
    base_std_itp = RegularGridInterpolator((aXs,bXs,eXs), base_stds)
    opto_mean_itp = RegularGridInterpolator((aXs,bXs,eXs), opto_means)
    opto_std_itp = RegularGridInterpolator((aXs,bXs,eXs), opto_stds)
    diff_mean_itp = RegularGridInterpolator((aXs,bXs,eXs), diff_means)
    diff_std_itp = RegularGridInterpolator((aXs,bXs,eXs), diff_stds)
    norm_cov_itp = RegularGridInterpolator((aXs,bXs,eXs), norm_covs)

    def fit_best_inputs(eX):
        def residuals(x):
            this_bX = x[0]
            this_aXs = np.concatenate(([0],x[1:]))
            pred_base_means = base_mean_itp(np.vstack((this_aXs,this_bX*np.ones(nc),eX*np.ones(nc))).T)
            pred_base_stds = base_std_itp(np.vstack((this_aXs,this_bX*np.ones(nc),eX*np.ones(nc))).T)
            pred_opto_means = opto_mean_itp(np.vstack((this_aXs,this_bX*np.ones(nc),eX*np.ones(nc))).T)
            pred_opto_stds = opto_std_itp(np.vstack((this_aXs,this_bX*np.ones(nc),eX*np.ones(nc))).T)
            pred_diff_means = diff_mean_itp(np.vstack((this_aXs,this_bX*np.ones(nc),eX*np.ones(nc))).T)
            pred_diff_stds = diff_std_itp(np.vstack((this_aXs,this_bX*np.ones(nc),eX*np.ones(nc))).T)
            pred_norm_covs = norm_cov_itp(np.vstack((this_aXs,this_bX*np.ones(nc),eX*np.ones(nc))).T)
            res = np.array([(pred_base_means-data_base_means)/data_base_means_err,
                            (pred_base_stds-data_base_stds)/data_base_stds_err,
                            (pred_opto_means-data_opto_means)/data_opto_means_err,
                            (pred_opto_stds-data_opto_stds)/data_opto_stds_err,
                            (pred_diff_means-data_diff_means)/data_diff_means_err,
                            (pred_diff_stds-data_diff_stds)/data_diff_stds_err,
                            (pred_norm_covs-data_norm_covs)/data_norm_covs_err])
            return res.ravel()
        xmin = np.concatenate(([bXs[ 0]],aXs[ 0]*np.ones(nc-1)))
        xmax = np.concatenate(([bXs[-1]],aXs[-1]*np.ones(nc-1)))
        x0 = np.concatenate(([2],np.linspace(1,12,nc-1)))
        results = least_squares(residuals,x0,bounds=(xmin,xmax))
        this_bX = results.x[0]
        this_aXs = np.concatenate(([0],results.x[1:]))
        return (this_bX,this_aXs,base_mean_itp(np.vstack((this_aXs,this_bX*np.ones(nc),eX*np.ones(nc))).T),
                base_std_itp(np.vstack((this_aXs,this_bX*np.ones(nc),eX*np.ones(nc))).T),
                opto_mean_itp(np.vstack((this_aXs,this_bX*np.ones(nc),eX*np.ones(nc))).T),
                opto_std_itp(np.vstack((this_aXs,this_bX*np.ones(nc),eX*np.ones(nc))).T),
                diff_mean_itp(np.vstack((this_aXs,this_bX*np.ones(nc),eX*np.ones(nc))).T),
                diff_std_itp(np.vstack((this_aXs,this_bX*np.ones(nc),eX*np.ones(nc))).T),
                norm_cov_itp(np.vstack((this_aXs,this_bX*np.ones(nc),eX*np.ones(nc))).T),results.cost)

    def fit_best_input_var():
        def residuals(x):
            _,_,_,_,_,_,_,_,_,cost = fit_best_inputs(x[0])
            return [cost]
        xmin,xmax = (eXs[0]),(eXs[-1])
        x0 = np.array([0.2])
        results = least_squares(residuals,x0,bounds=(xmin,xmax))
        return (results.x[0],*fit_best_inputs(results.x[0]))

    best_eX,best_bX,best_aXs,best_base_means,best_base_stds,best_opto_means,best_opto_stds,\
        best_diff_means,best_diff_stds,best_norm_covs,best_cost = fit_best_input_var()
    this_res_dict['best_eX'] = best_eX
    this_res_dict['best_bX'] = best_bX
    this_res_dict['best_aXs'] = best_aXs
    this_res_dict['best_base_means'] = best_base_means
    this_res_dict['best_base_stds'] = best_base_stds
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
