import argparse
import os
try:
    import pickle5 as pickle
except:
    import pickle
import numpy as np
import torch
import torch_interpolations as torchitp
from torchquad import Simpson, set_up_backend

import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_up_backend("torch", data_type="float32")
simp = Simpson()

parser = argparse.ArgumentParser(description=('This python script takes results from sampled spatial model parameters, '
    'trains a net to interpolate the results, and finds parameters that best fit the experimental results'))

parser.add_argument('--L', '-L',  help='Laser strength', type=float, default=1.0)
parser.add_argument('--CVL', '-CVL',  help='Laser strength', type=float, default=1.0)
args = vars(parser.parse_args())
print(parser.parse_args())
L= args['L']
CVL= args['CVL']

resultsdir='./../results/'
print('Saving all results in '+  resultsdir)
print(' ')

if not os.path.exists(resultsdir):
    os.makedirs(resultsdir)

λL = 1e-3*L
LNΣ = np.log(1+CVL**2)
LNσ = np.sqrt(LNΣ)
LNμ = np.log(λL)-0.5*LNΣ

start = time.process_time()

with open(resultsdir+'itp_ranges'+'.pkl', 'rb') as handle:
    ranges_dict = pickle.load(handle)

φxrange = ranges_dict['Ph']['xrange']
φxs = np.linspace(φxrange[0],φxrange[1],round(φxrange[2])).astype(np.float32)
φEs = np.load(resultsdir+'PhE_itp.npy').astype(np.float32)
φIs = np.load(resultsdir+'PhI_itp.npy').astype(np.float32)
φxs_torch = torch.from_numpy(φxs).to(device)
φEs_torch = torch.from_numpy(φEs).to(device)
φIs_torch = torch.from_numpy(φIs).to(device)

Mxrange = ranges_dict['M']['xrange']
Mxs = np.linspace(Mxrange[0],Mxrange[1],round(Mxrange[2])).astype(np.float32)
Mσrange = ranges_dict['M']['σrange']
Mσs = np.linspace(Mσrange[0],Mσrange[1],round(Mσrange[2])).astype(np.float32)
MEs = np.load(resultsdir+'ME_itp.npy').astype(np.float32)
MIs = np.load(resultsdir+'MI_itp.npy').astype(np.float32)
Mxs_torch = torch.from_numpy(Mxs).to(device)
Mσs_torch = torch.from_numpy(Mσs).to(device)
MEs_torch = torch.from_numpy(MEs).to(device)
MIs_torch = torch.from_numpy(MIs).to(device)

Cxrange = ranges_dict['C']['xrange']
Cxs = np.linspace(Cxrange[0],Cxrange[1],round(Cxrange[2])).astype(np.float32)
Cσrange = ranges_dict['C']['σrange']
Cσs = np.linspace(Cσrange[0],Cσrange[1],round(Cσrange[2])).astype(np.float32)
Ccrange = ranges_dict['C']['crange']
Ccs = np.linspace(Ccrange[0],Ccrange[1],round(Ccrange[2])).astype(np.float32)
CEs = np.load(resultsdir+'CE_itp.npy').astype(np.float32)
CIs = np.load(resultsdir+'CI_itp.npy').astype(np.float32)
Cxs_torch = torch.from_numpy(Cxs).to(device)
Cσs_torch = torch.from_numpy(Cσs).to(device)
Ccs_torch = torch.from_numpy(Ccs).to(device)
CEs_torch = torch.from_numpy(CEs).to(device)
CIs_torch = torch.from_numpy(CIs).to(device)

φE_itp = torchitp.RegularGridInterpolator((φxs_torch,),φEs_torch)
φI_itp = torchitp.RegularGridInterpolator((φxs_torch,),φIs_torch)

ME_itp = torchitp.RegularGridInterpolator((Mσs_torch,Mxs_torch),MEs_torch)
MI_itp = torchitp.RegularGridInterpolator((Mσs_torch,Mxs_torch),MIs_torch)

CE_itp = torchitp.RegularGridInterpolator((Ccs_torch,Cσs_torch,Cxs_torch),CEs_torch)
CI_itp = torchitp.RegularGridInterpolator((Ccs_torch,Cσs_torch,Cxs_torch),CIs_torch)

sr2 = np.sqrt(2)
sr2π = np.sqrt(2*np.pi)

def μtox(μ):
    return np.sign(μ/100-0.2)*np.abs(μ/100-0.2)**0.5

def xtoμ(x):
    return 100*(np.sign(x)*np.abs(x)**2.0+0.2)

def μtox_torch(μ):
    return torch.sign(μ/100-0.2)*torch.abs(μ/100-0.2)**0.5

def xtoμ_torch(x):
    return 100*(torch.sign(x)*torch.abs(x)**2.0+0.2)

def φE(μ):
    try:
        return φE_itp(μtox_torch(1e3*μ)[None,:])
    except:
        return φE_itp(torch.tensor([μtox_torch(1e3*μ)]))
def φI(μ):
    try:
        return φI_itp(μtox_torch(1e3*μ)[None,:])
    except:
        return φI_itp(torch.tensor([μtox_torch(1e3*μ)]))

def ME(μ,Σ):
    try:
        return ME_itp(torch.row_stack(torch.broadcast_tensors(1e3*torch.sqrt(Σ),μtox_torch(1e3*μ))))
    except:
        return ME_itp(torch.tensor([[1e3*torch.sqrt(Σ)],[μtox_torch(1e3*μ)]]))
def MI(μ,Σ):
    try:
        return MI_itp(torch.row_stack(torch.broadcast_tensors(1e3*torch.sqrt(Σ),μtox_torch(1e3*μ))))
    except:
        return MI_itp(torch.tensor([[1e3*torch.sqrt(Σ)],[μtox_torch(1e3*μ)]]))
def ME_vecμ(μ,Σ):
    return ME_itp(torch.stack((1e3*torch.sqrt(Σ)*torch.ones_like(μ),μtox_torch(1e3*μ)),dim=0))
def MI_vecμ(μ,Σ):
    return MI_itp(torch.stack((1e3*torch.sqrt(Σ)*torch.ones_like(μ),μtox_torch(1e3*μ)),dim=0))

def CE(μ,Σ,k):
    c = torch.sign(k)*torch.fmin(torch.abs(k)/Σ,torch.tensor(1))
    try:
        return CE_itp(torch.row_stack(torch.broadcast_tensors(c,1e3*torch.sqrt(Σ),μtox_torch(1e3*μ))))
    except:
        return CE_itp(torch.tensor([[c],[1e3*torch.sqrt(Σ)],[μtox_torch(1e3*μ)]]))
def CI(μ,Σ,k):
    c = torch.sign(k)*torch.fmin(torch.abs(k)/Σ,torch.tensor(1))
    try:
        return CI_itp(torch.row_stack(torch.broadcast_tensors(c,1e3*torch.sqrt(Σ),μtox_torch(1e3*μ))))
    except:
        return CI_itp(torch.tensor([[c],[1e3*torch.sqrt(Σ)],[μtox_torch(1e3*μ)]]))
def CE_vecμ(μ,Σ,k):
    c = torch.sign(k)*torch.fmin(torch.abs(k)/Σ,torch.tensor(1))
    return CE_itp(torch.stack((c*torch.ones_like(μ),1e3*torch.sqrt(Σ)*torch.ones_like(μ),μtox_torch(1e3*μ)),dim=0))
def CI_vecμ(μ,Σ,k):
    c = torch.sign(k)*torch.fmin(torch.abs(k)/Σ,torch.tensor(1))
    return CI_itp(torch.stack((c*torch.ones_like(μ),1e3*torch.sqrt(Σ)*torch.ones_like(μ),μtox_torch(1e3*μ)),dim=0))

Nint = 1000001
# Nint = 101

def φLint(μ):
    return simp.integrate(lambda x: torch.exp(-0.5*((torch.log(x)-LNμ)/LNσ)**2)/(sr2π*LNσ*x)*φE(μ+x),
        dim=1,N=Nint,integration_domain=[[1e-12,200*λL]],backend='torch').cpu().numpy()

def MLint(μ,Σ):
    return simp.integrate(lambda x: torch.exp(-0.5*((torch.log(x)-LNμ)/LNσ)**2)/(sr2π*LNσ*x)*ME_vecμ(μ+x,Σ),
        dim=1,N=Nint,integration_domain=[[1e-12,200*λL]],backend='torch').cpu().numpy()

def CLint(μ,Σ,k):
    return simp.integrate(lambda x: torch.exp(-0.5*((torch.log(x)-LNμ)/LNσ)**2)/(sr2π*LNσ*x)*CE_vecμ(μ+x,Σ,k),
        dim=1,N=Nint,integration_domain=[[1e-12,200*λL]],backend='torch').cpu().numpy()

print("Interpolating base moments took ",time.process_time() - start," s")
print('')

start = time.process_time()
    
φxrange = ranges_dict['PhL']['xrange']
φxs = np.linspace(φxrange[0],φxrange[1],round(φxrange[2])).astype(np.float32)
φxs_torch = torch.from_numpy(φxs).to(device)

try:
    φLs = np.load(resultsdir+'PhL_itp'+'_L={:.2f}'.format(L)+'_CVL={:.2f}'.format(CVL)+'.npy').astype(np.float32)
except:
    φLs = np.zeros((len(φxs)),dtype=np.float32)

    for x_idx,x in enumerate(φxs_torch):
        φLs[x_idx] = φLint(xtoμ_torch(x)*1e-3)
        
    # φL_itp = torchitp.RegularGridInterpolator((φxs,),φLs)

    np.save(resultsdir+'PhL_itp'+'_L={:.2f}'.format(L)+'_CVL={:.2f}'.format(CVL),φLs)

print("Interpolating φL took ",time.process_time() - start," s")
print('')

start = time.process_time()

Mxrange = ranges_dict['ML']['xrange']
Mxs = np.linspace(Mxrange[0],Mxrange[1],round(Mxrange[2])).astype(np.float32)
Mσrange = ranges_dict['ML']['σrange']
Mσs = np.linspace(Mσrange[0],Mσrange[1],round(Mσrange[2])).astype(np.float32)
Mxs_torch = torch.from_numpy(Mxs).to(device)
Mσs_torch = torch.from_numpy(Mσs).to(device)

try:
    MLs = np.load(resultsdir+'ML_itp'+'_L={:.2f}'.format(L)+'_CVL={:.2f}'.format(CVL)+'.npy').astype(np.float32)
except:
    MLs = np.zeros((len(Mσs),len(Mxs)),dtype=np.float32)

    for σ_idx,σ in enumerate(Mσs_torch):
        for x_idx,x in enumerate(Mxs_torch):
            MLs[σ_idx,x_idx] = MLint(xtoμ_torch(x)*1e-3,(σ*1e-3)**2)
        
    # ML_itp = torchitp.RegularGridInterpolator((Mσs,Mxs),MLs)

    np.save(resultsdir+'ML_itp'+'_L={:.2f}'.format(L)+'_CVL={:.2f}'.format(CVL),MLs)

print("Interpolating ML took ",time.process_time() - start," s")
print('')

start = time.process_time()

Cxrange = ranges_dict['CL']['xrange']
Cxs = np.linspace(Cxrange[0],Cxrange[1],round(Cxrange[2])).astype(np.float32)
Cσrange = ranges_dict['CL']['σrange']
Cσs = np.linspace(Cσrange[0],Cσrange[1],round(Cσrange[2])).astype(np.float32)
Ccrange = ranges_dict['CL']['crange']
Ccs = np.linspace(Ccrange[0],Ccrange[1],round(Ccrange[2])).astype(np.float32)
Cxs_torch = torch.from_numpy(Cxs).to(device)
Cσs_torch = torch.from_numpy(Cσs).to(device)
Ccs_torch = torch.from_numpy(Ccs).to(device)

try:
    CLs = np.load(resultsdir+'CL_itp'+'_L={:.2f}'.format(L)+'_CVL={:.2f}'.format(CVL)+'.npy').astype(np.float32)
except:
    CLs = np.zeros((len(Ccs),len(Cσs),len(Cxs)),dtype=np.float32)

    for c_idx,c in enumerate(Ccs_torch):
        for σ_idx,σ in enumerate(Cσs_torch):
            for x_idx,x in enumerate(Cxs_torch):
                CLs[c_idx,σ_idx,x_idx] = CLint(xtoμ_torch(x)*1e-3,(σ*1e-3)**2,c*(σ*1e-3)**2)
        
    # CL_itp = torchitp.RegularGridInterpolator((Ccs,Cσs,Cxs),CLs)

    np.save(resultsdir+'CL_itp'+'_L={:.2f}'.format(L)+'_CVL={:.2f}'.format(CVL),CLs)

print("Interpolating CL took ",time.process_time() - start," s")
print('')

# def φL(μ):
#     try:
#         return φL_itp(μtox(1e3*μ)[:,None])
#     except:
#         return φL_itp([μtox(1e3*μ)])

# def ML(μ,Σ):
#     return ML_itp(np.vstack((1e3*np.sqrt(Σ),μtox(1e3*μ))).T)

# def CL(μ,Σ,k):
#     c = np.sign(k)*np.fmin(np.abs(k)/Σ,1)
#     return CL_itp(np.vstack((c,1e3*np.sqrt(Σ),μtox(1e3*μ))).T)