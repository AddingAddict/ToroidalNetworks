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
from sbi_func import PostTimesBoxUniform

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

res_dir = res_dir + 'sbi_match/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

res_file = res_dir + 'bayes_iter={:d}_job={:d}.pkl'.format(bayes_iter, job_id)

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
    theta[:,8] = hmE - hbE
    theta[:,9] = hmI - hbI
    theta[:,10] = SoriE
    theta[:,11] = SoriI
    theta[:,12] = SoriF
    '''
    # load posterior of phase ring connectivity parameters
    with open(f'./../notebooks/sbi_base_6.pkl','rb') as handle:
        post = pickle.load(handle)
    
    full_prior = PostTimesBoxUniform(post,
        post_low =torch.tensor([0.0,-2.5,-4.0,-2.0, 0.2, 0.2, 0.2,-0.5]),
        post_high=torch.tensor([1.0, 2.0, 2.0, 2.5, 5.0, 5.0, 7.0, 2.0]),
        low =torch.tensor([ 1.0, 1.0,20,20,20],dtype=torch.float32),
        high=torch.tensor([10.0,10.0,40,40,40],dtype=torch.float32),
    )
else:
    with open(f'./../notebooks/sbi_match_{bayes_iter:d}.pkl','rb') as handle:
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
    rbs = np.zeros((2))
    rms = np.zeros((2))
    varrbs = np.zeros((2))
    varrms = np.zeros((2))
    
    Jee,Jei,Jie,Jii = get_J(theta)
    
    theta_prms = {
        'JEE': Jee.item(),
        'JEI': -Jei.item(),
        'JIE': Jie.item(),
        'JII': -Jii.item(),
        'bE': theta[4].item(),
        'bI': theta[5].item(),
        'hE': theta[8].item(),
        'hI': theta[9].item(),
        'L': theta[6].item(),
        'CVL': 10**theta[7].item(),
        'SoriE': theta[10].item(),
        'SoriI': theta[11].item(),
        'SoriF': theta[12].item(),
    }
    
    start = time.process_time()

    net,this_M,this_H,this_B,this_LAS,_ = su.gen_ring_disorder_tensor(rng.integers(100),
                                                                        prms | theta_prms,0,exact_params=True,diff_base=True)
    vsm_mask = net.get_oriented_neurons(delta_ori=4.5)[0]
    M = this_M.cpu().numpy()
    H = this_H.cpu().numpy()
    B = this_B.cpu().numpy()
    LAS = this_LAS.cpu().numpy()

    print("Generating disorder took ",time.process_time() - start," s")
    print('')

    start = time.process_time()

    init_base_sol,init_base_timeout = integ.sim_dyn_tensor(ri,T,0.0,this_M,this_B,
                                                    this_LAS,net.C_conds[0],mult_tau=True,max_min=30)
    init_base_r = np.mean(init_base_sol[:,mask_time].cpu().numpy(),-1)
    if init_base_timeout:
        print('init base network integration timed out')
        rbs[0] = np.nan
        varrbs[0] = np.nan
        balb = np.nan
    else:
        rbs[0] = np.mean(init_base_r)
        varrbs[0] = np.var(init_base_r)
        muE = B + M[:,net.C_all[0]]@init_base_r[net.C_all[0]]
        muI = M[:,net.C_all[1]]@init_base_r[net.C_all[1]]
        muE[net.C_all[0]] *= ri.tE
        muE[net.C_all[1]] *= ri.tI
        muI[net.C_all[0]] *= ri.tE
        muI[net.C_all[1]] *= ri.tI
        balb = np.mean(np.abs(muE + muI)/muE)

    print("Integrating init base network took ",time.process_time() - start," s")
    print('')

    start = time.process_time()
    
    opto_base_sol,opto_base_timeout = integ.sim_dyn_tensor(ri,T,1.0,this_M,this_B,
                                                    this_LAS,net.C_conds[0],mult_tau=True,max_min=30)
    opto_base_r = np.mean(opto_base_sol[:,mask_time].cpu().numpy(),-1)
    if opto_base_timeout:
        print('opto base network integration timed out')
        rbs[1] = np.nan
        varrbs[1] = np.nan
    else:
        rbs[1] = np.mean(opto_base_r)
        varrbs[1] = np.var(opto_base_r)
        
    vardrb = np.var(opto_base_r - init_base_r)

    start = time.process_time()

    init_match_sol,init_match_timeout = integ.sim_dyn_tensor(ri,T,0.0,this_M,this_B + this_H,
                                                    this_LAS,net.C_conds[0],mult_tau=True,max_min=30)    
    if init_match_timeout:
        print('init match network integration timed out')
        rms[0] = np.nan
        varrms[0] = np.nan
        balm = np.nan
        init_norm_min = np.array([np.nan,np.nan])
    else:
        rms[0] = np.mean(init_match_r[vsm_mask])
        varrms[0] = np.var(init_match_r[vsm_mask])
        muE = B + H + M[:,net.C_all[0]]@init_match_r[net.C_all[0]]
        muI = M[:,net.C_all[1]]@init_match_r[net.C_all[1]]
        muE[net.C_all[0]] *= ri.tE
        muE[net.C_all[1]] *= ri.tI
        muI[net.C_all[0]] *= ri.tE
        muI[net.C_all[1]] *= ri.tI
        balm = np.mean((np.abs(muE + muI)/muE)[vsm_mask])
        
        init_match_r = np.mean(init_match_sol[:,mask_time].cpu().numpy(),-1)
        m_init_match_r = np.zeros((2,Nori))
        for i in range(Nori):
            m_init_match_r[0,i] = np.mean(init_match_r[net.C_idxs[0][i]])
            m_init_match_r[1,i] = np.mean(init_match_r[net.C_idxs[1][i]])
        init_norm_min = np.min(m_init_match_r,1)
        init_norm_min -= m_init_match_r[:,Nori//2]
        init_norm_min /= (m_init_match_r[:,0] - m_init_match_r[:,Nori//2])

    print("Integrating init match network took ",time.process_time() - start," s")
    print('')

    start = time.process_time()
    
    opto_match_sol,opto_match_timeout = integ.sim_dyn_tensor(ri,T,1.0,this_M,this_B + this_H,
                                                    this_LAS,net.C_conds[0],mult_tau=True,max_min=30)
    opto_match_r = np.mean(opto_match_sol[:,mask_time].cpu().numpy(),-1)
    if opto_match_timeout:
        print('opto match network integration timed out')
        rms[1] = np.nan
        varrms[1] = np.nan
        opto_norm_min = np.array([np.nan,np.nan])
    else:
        rms[1] = np.mean(opto_match_r[vsm_mask])
        varrms[1] = np.var(opto_match_r[vsm_mask])
        
        opto_match_r = np.mean(opto_match_sol[:,mask_time].cpu().numpy(),-1)
        m_opto_match_r = np.zeros((2,Nori))
        for i in range(Nori):
            m_opto_match_r[0,i] = np.mean(opto_match_r[net.C_idxs[0][i]])
            m_opto_match_r[1,i] = np.mean(opto_match_r[net.C_idxs[1][i]])
        opto_norm_min = np.min(m_opto_match_r,1)
        opto_norm_min -= m_opto_match_r[:,Nori//2]
        opto_norm_min /= (m_opto_match_r[:,0] - m_opto_match_r[:,Nori//2])
        
    vardrm = np.var(opto_match_r[vsm_mask] - init_base_r[vsm_mask])

    return rbs,rms,varrbs,varrms,vardrb,vardrm,balb,balm,init_norm_min,opto_norm_min

def get_obs(thetas):
    len_t = thetas.shape[0]
    out = torch.zeros((len_t,16),dtype=torch.float32,device=thetas.device)
    for i in range(len_t):
        print('simulating sample # '+str(i+1))
        print('')
        theta = thetas[i]
        
        rbs,rms,varrbs,varrms,vardrb,vardrm,balb,balm,init_norm_min,opto_norm_min = simulate_network(theta)
        out[i] = torch.tensor([rbs[0],rbs[1],varrbs[0],varrbs[1],vardrb,rms[0],rms[1],varrms[0],varrms[1],vardrm,balb,balm,init_norm_min[0],init_norm_min[1],opto_norm_min[0],opto_norm_min[1]],dtype=theta.dtype).to(theta.device)
    return out

thetas = torch.zeros((0,13),dtype=torch.float32,device=torch.device('cpu'))
xs = torch.zeros((0,16),dtype=torch.float32,device=torch.device('cpu'))
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
