import argparse
import pickle
import numpy as np
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

eps = np.random.default_rng(0).standard_normal(6480)

def gen_prms(seed):
    prm_dict = {}

    rng = np.random.default_rng(seed)
    prm_dict['SrfE'] = rng.uniform(5,20)
    prm_dict['SrfI'] = rng.uniform(5,20)
    prm_dict['SrfF'] = rng.uniform(5,20)
    prm_dict['SoriE'] = rng.uniform(15,45)
    prm_dict['SoriI'] = rng.uniform(15,45)
    prm_dict['SoriF'] = rng.uniform(15,45)
    prm_dict['fEIs'] = rng.uniform(0.8,1.2)
    prm_dict['fIEs'] = rng.uniform(0.8,1.2)
    prm_dict['fIIs'] = rng.uniform(0.8,1.2)
    prm_dict['fFIs'] = rng.uniform(0.8,1.2)

    return prm_dict

def gen_disorder(prm_dict):
    SrfE = prm_dict['SrfE']
    SrfI = prm_dict['SrfI']
    SrfF = prm_dict['SrfF']
    SoriE = prm_dict['SoriE']
    SoriI = prm_dict['SoriI']
    SoriF = prm_dict['SoriF']
    fEIs = prm_dict['fEIs']
    fIEs = prm_dict['fIEs']
    fIIs = prm_dict['fIIs']
    fFIs = prm_dict['fFIs']

    net = m_network.network(seed=0,NC=[4,1],Nrf=36,Nori=9)
    net.generate_disorder(1e-3*np.array([[0.1,-fEIs*0.8],[fIEs*0.3,-fIIs*0.7]]),
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

    # aXs = np.arange(1,15+2,2)
    # bXs = np.arange(1,7+2,2)
    # eXs = np.arange(0,0.3+0.05,0.05)

    aXs = np.arange(1,15+2,4)
    bXs = np.arange(1,7+2,4)
    eXs = np.arange(0,0.3+0.05,0.1)

    means = np.zeros((len(aXs),len(bXs),len(eXs)))
    stds = np.zeros((len(aXs),len(bXs),len(eXs)))

    for aX_idx,aX in enumerate(aXs):
        for bX_idx,bX in enumerate(bXs):
            for eX_idx,eX in enumerate(eXs):
                sol,_ = integ.sim_dyn(ri,T,0.0,M,(bX*B+aX*H)*(1+eX*eps),H,net.C_all[0],net.C_all[1],
                                      mult_tau=True,max_min=30)
                means[aX_idx,bX_idx,eX_idx] = np.mean(sol[net.get_centered_neurons(),-1])
                stds[aX_idx,bX_idx,eX_idx] = np.std(sol[net.get_centered_neurons(),-1])

    print("Simulating inputs took ",time.process_time() - start," s")
    print('')

    this_res_dict = {}
    this_res_dict['prms'] = prm_dict
    this_res_dict['means'] = means
    this_res_dict['stds'] = stds

    res_dict[idx_rep] = this_res_dict

    with open(this_results, 'wb') as handle:
        pickle.dump(res_dict,handle)