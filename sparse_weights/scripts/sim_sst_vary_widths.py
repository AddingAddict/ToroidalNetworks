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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NtE = 50
Nt = NtE*ri.tE
dt = ri.tI/5
T = torch.linspace(0,5*Nt,round(5*Nt/dt)+1)
mask_time = T>(4*Nt)
T_mask = T.cpu().numpy()[mask_time]

N = 12000
Nori = 20
NE = 12*(N//Nori)//15
NP = 2*(N//Nori)//15
NS = 1*(N//Nori)//15

W = 1e-3*np.array([[0.15,-1,-0.2],[1.0,-3,-0.5],[0.2,-0.35,0.0]])
sW = np.array([[30,20,20],[30,20,20],[15,10,20]])
H = np.array([0.25,0.3,0.00])
sH = 20*np.ones(3)
K = 500
CVh = 0.2
rX = 6
cA = 4

net = network.RingNetwork(NC=[NE,NP,NS],Nori=Nori)

B = np.zeros(net.N,dtype=np.float32)
B[net.C_all[0]] = H[0]
B[net.C_all[1]] = H[1]
B_torch = torch.from_numpy(B).to(device)

Eps = np.random.default_rng(0).gamma(1/CVh**2,scale=CVh**2,size=net.N).astype(np.float32)
Eps_torch = torch.from_numpy(Eps).to(device)

net.generate_disorder(W,sW,H,sH,500)
net.generate_tensors()
E_cond = torch.logical_or(net.C_conds[0],net.C_conds[2]).to(device)

Inp = rX*(B+cA*net.H)*Eps
Inp_torch = torch.from_numpy(Inp).to(device)