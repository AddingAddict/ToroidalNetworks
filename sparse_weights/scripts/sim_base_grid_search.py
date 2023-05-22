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

aXs = np.arange(0,16+2,2)
bXs = np.arange(1,9+2,2)
eXs = np.arange(0,0.5+0.05,0.05)

net = m_network.network(seed=0,NC=[4,1],Nrf=56,Nori=9,Lrf=90)

eps = np.zeros((len(eXs),net.N))
for eX_idx,eX in enumerate(eXs):
    if eX == 0.0:
        eps[eX_idx] = np.ones(net.N)
    else:
        shape = 1/eX**2
        scale = 1/shape
        eps[eX_idx] = np.random.default_rng(0).gamma(shape,scale=scale,size=net.N)

data_means = np.array([6.22, 6.72, 7.17, 7.67, 8.,  10.97, 16.7])
data_stds =  np.array([5.79, 6.64, 6.93, 7.15, 7.07, 8.98, 13.6])

data_means_err = np.array([0.83, 0.96, 1.00, 1.03, 1.02, 1.30, 2.00])
data_stds_err =  np.array([0.48, 0.78, 0.96, 0.81, 0.78, 1.08, 2.63])

nc = len(data_means)

def gen_prms(seed):
    prm_dict = {}

    rng = np.random.default_rng(seed)
    prm_dict['SrfE'] = rng.uniform(5,20)
    prm_dict['SrfI'] = rng.uniform(5,20)
    prm_dict['SrfF'] = 30#rng.uniform(5,20)
    prm_dict['SoriE'] = rng.uniform(15,45)
    prm_dict['SoriI'] = rng.uniform(15,45)
    prm_dict['SoriF'] = rng.uniform(15,45)
    prm_dict['fEEs'] = rng.uniform(0.6,1.4)
    prm_dict['fEIs'] = rng.uniform(0.6,1.4)
    prm_dict['fIEs'] = rng.uniform(0.6,1.4)
    prm_dict['fIIs'] = rng.uniform(0.6,1.4)
    prm_dict['fFIs'] = rng.uniform(0.6,1.4)

    return prm_dict

def gen_disorder(prm_dict):
    SrfE = prm_dict['SrfE']
    SrfI = prm_dict['SrfI']
    SrfF = prm_dict['SrfF']
    SoriE = prm_dict['SoriE']
    SoriI = prm_dict['SoriI']
    SoriF = prm_dict['SoriF']
    fEEs = prm_dict['fEEs']
    fEIs = prm_dict['fEIs']
    fIEs = prm_dict['fIEs']
    fIIs = prm_dict['fIIs']
    fFIs = prm_dict['fFIs']

    net.set_seed(0)
    net.generate_disorder(1e-3*np.array([[fEE*0.1,-fEIs*0.8],[fIEs*0.3,-fIIs*0.7]]),
                      np.array([[SrfE,SrfI],[SrfE,SrfI]]),np.array([[SoriE,SoriI],[SoriE,SoriI]]),
                      500*1e-3*np.array([0.25,fFIs*0.2]),
                      SrfF*np.ones(2),SoriF*np.ones(2),500)
    B = np.zeros_like(net.H)
    B[net.C_all[0]] = 500*1e-3*0.25
    B[net.C_all[0]] = 500*1e-3*fFIs*0.2

    return net,net.M,net.H,B

resultsdir='./../results/'
name_results='results_base_'+str(njob)+'.pkl'
this_results=resultsdir+name_results
print('Saving all results in '+  name_results)
print(' ')

if not os.path.exists(resultsdir):
    os.makedirs(resultsdir)

init = True
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
    net,M,H,B = gen_disorder(prm_dict)

    print("Disorder generation took ",time.process_time() - start," s")
    print('')

    start = time.process_time()

    means = np.zeros((len(aXs),len(bXs),len(eXs)))
    stds = np.zeros((len(aXs),len(bXs),len(eXs)))

    for aX_idx,aX in enumerate(aXs):
        for bX_idx,bX in enumerate(bXs):
            for eX_idx,eX in enumerate(eXs):
                sol,_ = integ.sim_dyn(ri,T,0.0,M,(bX*B+aX*H)*eps[eX_idx],H,net.C_all[0],net.C_all[1],
                                      mult_tau=True,max_min=30)
                means[aX_idx,bX_idx,eX_idx] = np.mean(sol[net.get_centered_neurons(),-1])
                stds[aX_idx,bX_idx,eX_idx] = np.std(sol[net.get_centered_neurons(),-1])

    print("Simulating inputs took ",time.process_time() - start," s")
    print('')

    start = time.process_time()

    this_res_dict = {}
    this_res_dict['prms'] = prm_dict
    this_res_dict['means'] = means
    this_res_dict['stds'] = stds

    mean_itp = RegularGridInterpolator((aXs,bXs,eXs), means)
    std_itp = RegularGridInterpolator((aXs,bXs,eXs), stds)

    def fit_best_inputs(eX):
        def residuals(x):
            this_bX = x[0]
            this_aXs = np.concatenate(([0],x[1:]))
            pred_means = mean_itp(np.vstack((this_aXs,this_bX*np.ones(nc),eX*np.ones(nc))).T)
            pred_stds = std_itp(np.vstack((this_aXs,this_bX*np.ones(nc),eX*np.ones(nc))).T)
            res = np.array([(pred_means-data_means)/data_means_err, (pred_stds-data_stds)/data_stds_err])
            return res.ravel()
        xmin = np.concatenate(([bXs[ 0]],aXs[ 0]*np.ones(nc-1)))
        xmax = np.concatenate(([bXs[-1]],aXs[-1]*np.ones(nc-1)))
        x0 = np.concatenate(([2],np.linspace(1,12,nc-1)))
        results = least_squares(residuals,x0,bounds=(xmin,xmax))
        this_bX = results.x[0]
        this_aXs = np.concatenate(([0],results.x[1:]))
        return (this_bX,this_aXs,mean_itp(np.vstack((this_aXs,this_bX*np.ones(nc),eX*np.ones(nc))).T),
                std_itp(np.vstack((this_aXs,this_bX*np.ones(nc),eX*np.ones(nc))).T),results.cost)

    def fit_best_input_var():
        def residuals(x):
            _,_,_,_,cost = fit_best_inputs(x[0])
            return [cost]
        xmin,xmax = (eXs[0]),(eXs[-1])
        x0 = np.array([0.2])
        results = least_squares(residuals,x0,bounds=(xmin,xmax))
        return (results.x[0],*fit_best_inputs(results.x[0]))

    best_eX,best_bX,best_aXs,best_means,best_stds,best_cost = fit_best_input_var()
    this_res_dict['best_eX'] = best_eX
    this_res_dict['best_bX'] = best_bX
    this_res_dict['best_aXs'] = best_aXs
    this_res_dict['best_means'] = best_means
    this_res_dict['best_stds'] = best_stds
    this_res_dict['best_cost'] = best_cost

    print("Fitting best inputs took ",time.process_time() - start," s")
    print('')
    print(prm_dict)
    print(best_cost)
    print('')

    res_dict[idx_rep] = this_res_dict

    with open(this_results, 'wb') as handle:
        pickle.dump(res_dict,handle)
