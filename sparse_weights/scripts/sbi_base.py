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

from sbi.utils import BoxUniform,MultipleIndependent
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)

parser = argparse.ArgumentParser()
parser.add_argument('--job_id', '-i', help='completely arbitrary job id label',type=int, default=0)
parser.add_argument('--num_samp', '-ns', help='number of samples',type=int, default=50)
parser.add_argument('--bayes_iter', '-bi', help='bayessian inference interation (0 = use prior, 1 = use first posterior)',type=int, default=0)
args = vars(parser.parse_args())
job_id = int(args['job_id'])
num_samp = int(args['num_samp'])
bayes_iter = int(args['bayes_iter'])

print("Bayesian iteration:", bayes_iter)
print("Job ID:", job_id)

# Define where to save results
res_dir = './../results/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_dir = res_dir + 'sbi_base/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_file = res_dir + 'bayes_iter={:d}_job={:d}.pkl'.format(bayes_iter, job_id)

SoriE = 30
SoriI = 30
SoriF = 30

# create prior distribution
if bayes_iter == 0:
    '''
    theta[:,0] = det(J)/(|Jei| * |Jie|) = 1 - (|Jee| * |Jii|) / (|Jei| * |Jie|)
    theta[:,1] = (|Jee|-|Jii|)/(|Jei| + |Jie|)
    theta[:,2] = (log10[|Jei|] + log10[|Jie|]) / 2
    theta[:,3] = (log10[|Jei|] - log10[|Jie|]) / 2
    theta[:,4] = hbE
    theta[:,5] = hbI
    theta[:,6] = L
    theta[:,7] = log10[CVL]
    '''
    # load posterior of phase ring connectivity parameters
    full_prior = BoxUniform(
        low =torch.tensor([0.0,-2.0,-2.0,-2.0, 0.2, 0.2, 0.2,-0.5]),
        high=torch.tensor([1.0, 2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 1.5]),
    )
else:
    with open(f'./../notebooks/sbi_base_{bayes_iter:d}.pkl','rb') as handle:
        full_prior = pickle.load(handle)

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

prms = {}
prms['Nori'] = Nori
prms['NE'] = NE
prms['NI'] = NI
prms['K'] = 500
prms['SoriE'] = SoriE
prms['SoriI'] = SoriI
prms['SoriF'] = SoriF

rng = np.random.default_rng(seed=job_id)

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

def simulate_network(theta):
    rs = np.zeros((2))
    varrs = np.zeros((2))
    
    Jee,Jei,Jie,Jii = get_J(theta)
    
    theta_prms = {
        'JEE': Jee.item(),
        'JEI': -Jei.item(),
        'JIE': Jie.item(),
        'JII': -Jii.item(),
        'hE': theta[4].item(),
        'hI': theta[5].item(),
        'L': theta[6].item(),
        'CVL': 10**theta[7].item(),
    }
    
    start = time.process_time()

    net,this_M,_,this_B,this_LAS,_ = su.gen_ring_disorder_tensor(rng.integers(100),
                                                                        prms | theta_prms,0,exact_params=True)
    M = this_M.cpu().numpy()
    H = this_B.cpu().numpy()
    LAS = this_LAS.cpu().numpy()

    print("Generating disorder took ",time.process_time() - start," s")
    print('')

    start = time.process_time()

    base_sol,base_timeout = integ.sim_dyn_tensor(ri,T,0.0,this_M,this_B,
                                                    this_LAS,net.C_conds[0],mult_tau=True,max_min=30)
    base_r = np.mean(base_sol[:,mask_time].cpu().numpy(),-1)
    if base_timeout:
        rs[0] = np.nan
        varrs[0] = np.nan
        bal = np.nan
    else:
        rs[0] = np.mean(base_r)
        varrs[0] = np.var(base_r)
        muE = H + M[:,net.C_all[0]]@base_r[net.C_all[0]]
        muI = M[:,net.C_all[1]]@base_r[net.C_all[1]]
        muE[net.C_all[0]] *= ri.tE
        muE[net.C_all[1]] *= ri.tI
        muI[net.C_all[0]] *= ri.tE
        muI[net.C_all[1]] *= ri.tI
        bal = np.mean(np.abs(muE + muI)/muE)

    print("Integrating base network took ",time.process_time() - start," s")
    print('')

    start = time.process_time()
    
    opto_sol,opto_timeout = integ.sim_dyn_tensor(ri,T,1.0,this_M,this_B,
                                                    this_LAS,net.C_conds[0],mult_tau=True,max_min=30)
    opto_r = np.mean(opto_sol[:,mask_time].cpu().numpy(),-1)
    if opto_timeout:
        rs[1] = np.nan
        varrs[1] = np.nan
    else:
        rs[1] = np.mean(opto_r)
        varrs[1] = np.var(opto_r)
        
    vardr = np.var(opto_r - base_r)

    return rs,varrs,vardr,bal

def get_obs(thetas):
    len_t = thetas.shape[0]
    out = torch.zeros((len_t,6),dtype=torch.float32,device=thetas.device)
    for i in range(len_t):
        print('simulating sample # '+str(i+1))
        print('')
        theta = thetas[i]
        
        rs,varrs,vardr,bal = simulate_network(theta)
        out[i] = torch.tensor([rs[0],rs[1],varrs[0],varrs[1],vardr,bal],dtype=theta.dtype).to(theta.device)
    return out

thetas = torch.zeros((0,8),dtype=torch.float32,device=torch.device('cpu'))
xs = torch.zeros((0,6),dtype=torch.float32,device=torch.device('cpu'))
while thetas.shape[0] < num_samp:
    this_samps = min(1, num_samp - thetas.shape[0])
    
    start = time.process_time()
    # sample from prior
    theta = full_prior.sample((this_samps,))
    # simulate sheet
    x = get_obs(theta)

    thetas = torch.cat([thetas,theta],dim=0)
    xs = torch.cat([xs,x],dim=0)

    print(f'Simulating samples took',time.process_time() - start,'s\n')

    # save results
    with open(res_file, 'wb') as handle:
        pickle.dump({
            'theta': thetas,
            'x': xs,
        }, handle)
