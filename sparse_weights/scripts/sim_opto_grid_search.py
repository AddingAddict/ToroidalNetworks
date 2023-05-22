import argparse
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

Ls = np.arange(0,1.4+0.2,0.2)
CVLs = np.arange(1,9+2,2)

net = m_network.network(seed=0,NC=[4,1],Nrf=56,Nori=9,Lrf=90)

eps = np.zeros((len(eXs),net.N))
for eX_idx,eX in enumerate(eXs):
    if eX == 0.0:
        eps[eX_idx] = np.ones(net.N)
    else:
        shape = 1/eX**2
        scale = 1/shape
        eps[eX_idx] = np.random.default_rng(0).gamma(shape,scale=scale,size=net.N)

opto_means = np.array([10.56, 10.79, 11.25, 12.19, 12.96, 14.89, 20.25])
opto_stds =  np.array([10.20, 10.20, 10.62, 10.85, 11.47, 11.84, 16.12])
diff_means = np.array([ 4.32,  4.05,  4.07,  4.51,  4.95,  3.90,  3.48])
diff_stds =  np.array([ 7.98,  7.96,  8.34,  8.42,  8.99,  8.41, 10.30])
norm_covs =  np.array([0.04409,-0.03578,-0.04356,-0.03427,-0.00142,-0.09287,-0.12237])

opto_means_err = np.array([1.46, 1.47, 1.53, 1.56, 1.65, 1.69, 2.32])
opto_stds_err =  np.array([1.63, 1.71, 1.88, 1.73, 1.66, 1.33, 2.57])
diff_means_err = np.array([1.15, 1.15, 1.22, 1.22, 1.31, 1.22, 1.50])
diff_stds_err =  np.array([1.37, 1.32, 1.76, 1.53, 1.61, 1.53, 1.86])
norm_covs_err =  np.array([0.1270, 0.1679, 0.1588, 0.1608, 0.1424, 0.1883, 0.4295])

nc = len(opto_means)

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
        eps[eX_idx] = np.ones(net.N)
    else:
        shape = 1/eX**2
        scale = 1/shape
        eps[eX_idx] = np.random.default_rng(0).gamma(shape,scale=scale,size=net.N)

    return net,net.M,net.H,B,eps

def gen_lam(CVL):
    sigma_l = np.sqrt(np.log(1+CV_Lam**2))
    mu_l = np.log(Lam)-sigma_l**2/2
    LAM = np.zeros(net.N)
    LAM[net.allE] = np.random.lognormal(mu_l, sigma_l, net.NE*net.Nloc)

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
    with open(base_results, 'rb') as handle:
        base_res_dict = pickle.load(handle)
    last_rep = max(list(res_dict.keys()))+1
except:
    base_res_dict = {}
    last_rep = 0

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

    means = np.zeros((len(Ls),len(CVLs),len(aXs)))
    stds = np.zeros((len(Ls),len(CVLs),len(aXs)))

    for L_idx,L in enumerate(Ls):
        for CVL_idx,CVL in enumerate(CVLs):
            for aX_idx,aX in enumerate(aXs):
                base_sol,_ = integ.sim_dyn(ri,T,0.0,M,(bX*B+aX*H)*eps[eX_idx],H,net.C_all[0],net.C_all[1],
                                      mult_tau=True,max_min=30)
                base_sol,_ = integ.sim_dyn(ri,T,L,M,(bX*B+aX*H)*eps[eX_idx],H,net.C_all[0],net.C_all[1],
                                      mult_tau=True,max_min=30)
                means[L_idx,CVL_idx,aX_idx] = np.mean(sol[net.get_centered_neurons(),-1])
                stds[L_idx,CVL_idx,aX_idx] = np.std(sol[net.get_centered_neurons(),-1])

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
            res = np.array([(pred_means-opto_means)/opto_means_err, (pred_stds-opto_stds)/opto_stds_err])
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
