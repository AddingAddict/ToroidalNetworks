import os
import pickle
import numpy as np
from scipy.linalg import toeplitz
from scipy.interpolate import interp1d,RegularGridInterpolator
from scipy.integrate import quad,simpson
from scipy.special import erf
from mpmath import fp
import time

dmu = 1e-4
sr2pi = np.sqrt(2*np.pi)

jtheta = np.vectorize(fp.jtheta, 'D')

def wrapnormdens(x,s,L=180):
    return np.real(jtheta(3,x*np.pi/L,np.exp(-(s*(2*np.pi)/L)**2/2)))/(2*np.pi)

def basesubwrapnorm(x,s,L=180):
    return (wrapnormdens(x,s,L)-wrapnormdens(L/2,s,L))/(wrapnormdens(0,s,L)-wrapnormdens(L/2,s,L))

def mutox(mu):
    return np.sign(mu/100-0.2)*np.abs(mu/100-0.2)**0.5

def xtomu(x):
    return 100*(np.sign(x)*np.abs(x)**2.0+0.2)
    
def base_itp_moments(res_dir):
    with open(os.path.join(res_dir,'itp_ranges.pkl'), 'rb') as handle:
        ranges_dict = pickle.load(handle)
    
    Fxrange = ranges_dict['Ph']['xrange']
    Fxs = np.linspace(Fxrange[0],Fxrange[1],round(Fxrange[2])).astype(np.float32)
    FEs = np.load(os.path.join(res_dir,'PhE_itp.npy')).astype(np.float32)
    FIs = np.load(os.path.join(res_dir,'PhI_itp.npy')).astype(np.float32)
    
    Mxrange = ranges_dict['M']['xrange']
    Mxs = np.linspace(Mxrange[0],Mxrange[1],round(Mxrange[2])).astype(np.float32)
    Msrange = ranges_dict['M']['σrange']
    Mss = np.linspace(Msrange[0],Msrange[1],round(Msrange[2])).astype(np.float32)
    MEs = np.load(os.path.join(res_dir,'ME_itp.npy')).astype(np.float32)
    MIs = np.load(os.path.join(res_dir,'MI_itp.npy')).astype(np.float32)
    
    Cxrange = ranges_dict['C']['xrange']
    Cxs = np.linspace(Cxrange[0],Cxrange[1],round(Cxrange[2])).astype(np.float32)
    Csrange = ranges_dict['C']['σrange']
    Css = np.linspace(Csrange[0],Csrange[1],round(Csrange[2])).astype(np.float32)
    Ccrange = ranges_dict['C']['crange']
    Ccs = np.linspace(Ccrange[0],Ccrange[1],round(Ccrange[2])).astype(np.float32)
    CEs = np.load(os.path.join(res_dir,'CE_itp.npy')).astype(np.float32)
    CIs = np.load(os.path.join(res_dir,'CI_itp.npy')).astype(np.float32)
    
    FE_itp = RegularGridInterpolator((Fxs,),FEs,bounds_error=False,fill_value=None)
    FI_itp = RegularGridInterpolator((Fxs,),FIs,bounds_error=False,fill_value=None)
    
    ME_itp = RegularGridInterpolator((Mss,Mxs),MEs,bounds_error=False,fill_value=None)
    MI_itp = RegularGridInterpolator((Mss,Mxs),MIs,bounds_error=False,fill_value=None)
    
    CE_itp = RegularGridInterpolator((Ccs,Css,Cxs),CEs,bounds_error=False,fill_value=None)
    CI_itp = RegularGridInterpolator((Ccs,Css,Cxs),CIs,bounds_error=False,fill_value=None)
    
    def FE(mu):
        try:
            return FE_itp(mutox(1e3*mu)[:,None])
        except:
            return FE_itp([mutox(1e3*mu)])
    def FI(mu):
        try:
            return FI_itp(mutox(1e3*mu)[:,None])
        except:
            return FI_itp([mutox(1e3*mu)])
    
    def ME(mu,Sig):
        return ME_itp(np.row_stack(list(np.broadcast(1e3*np.sqrt(Sig),mutox(1e3*mu)))))
    def MI(mu,Sig):
        return MI_itp(np.row_stack(list(np.broadcast(1e3*np.sqrt(Sig),mutox(1e3*mu)))))
    
    def CE(mu,Sig,k):
        c = np.sign(k)*np.fmin(np.abs(k)/Sig,1)
        return CE_itp(np.row_stack(list(np.broadcast(c,1e3*np.sqrt(Sig),mutox(1e3*mu)))))
    def CI(mu,Sig,k):
        c = np.sign(k)*np.fmin(np.abs(k)/Sig,1)
        return CI_itp(np.row_stack(list(np.broadcast(c,1e3*np.sqrt(Sig),mutox(1e3*mu)))))
    
    return FE,FI,ME,MI,CE,CI
    
def opto_itp_moments(res_dir,L,CVL):
    with open(os.path.join(res_dir,'itp_ranges.pkl'), 'rb') as handle:
        ranges_dict = pickle.load(handle)
    
    Fxrange = ranges_dict['PhL']['xrange']
    Fxs = np.linspace(Fxrange[0],Fxrange[1],round(Fxrange[2])).astype(np.float32)
    FLs = np.load(os.path.join(res_dir,'PhL_itp_L={:.2f}_CVL={:.2f}.npy'.format(L,CVL))).astype(np.float32)
    
    Mxrange = ranges_dict['ML']['xrange']
    Mxs = np.linspace(Mxrange[0],Mxrange[1],round(Mxrange[2])).astype(np.float32)
    Msrange = ranges_dict['ML']['σrange']
    Mss = np.linspace(Msrange[0],Msrange[1],round(Msrange[2])).astype(np.float32)
    MLs = np.load(os.path.join(res_dir,'ML_itp_L={:.2f}_CVL={:.2f}.npy'.format(L,CVL))).astype(np.float32)
    
    Cxrange = ranges_dict['CL']['xrange']
    Cxs = np.linspace(Cxrange[0],Cxrange[1],round(Cxrange[2])).astype(np.float32)
    Csrange = ranges_dict['CL']['σrange']
    Css = np.linspace(Csrange[0],Csrange[1],round(Csrange[2])).astype(np.float32)
    Ccrange = ranges_dict['CL']['crange']
    Ccs = np.linspace(Ccrange[0],Ccrange[1],round(Ccrange[2])).astype(np.float32)
    CLs = np.load(os.path.join(res_dir,'CL_itp_L={:.2f}_CVL={:.2f}.npy'.format(L,CVL))).astype(np.float32)
    
    FL_itp = RegularGridInterpolator((Fxs,),FLs,bounds_error=False,fill_value=None)
    
    ML_itp = RegularGridInterpolator((Mss,Mxs),MLs,bounds_error=False,fill_value=None)
    
    CL_itp = RegularGridInterpolator((Ccs,Css,Cxs),CLs,bounds_error=False,fill_value=None)
    
    def FL(mu):
        try:
            return FL_itp(mutox(1e3*mu)[:,None])
        except:
            return FL_itp([mutox(1e3*mu)])
    
    def ML(mu,Sig):
        return ML_itp(np.row_stack(list(np.broadcast(1e3*np.sqrt(Sig),mutox(1e3*mu)))))
    
    def CL(mu,Sig,k):
        c = np.sign(k)*np.fmin(np.abs(k)/Sig,1)
        return CL_itp(np.row_stack(list(np.broadcast(c,1e3*np.sqrt(Sig),mutox(1e3*mu)))))
    
    return FL,ML,CL

def R(M1,M2,mu1,mu2,Sig1,Sig2,k):
    c = np.sign(k)*np.fmin(np.abs(k)/np.sqrt(Sig1*Sig2),1)
    sig1 = np.sign(c)*np.sqrt(Sig1*np.abs(c))
    sig2 = np.sqrt(Sig2*np.abs(c))
    Del1 = Sig1*(1-np.abs(c))
    Del2 = Sig2*(1-np.abs(c))
    return quad(lambda x: np.exp(-0.5*x**2)/sr2pi*\
                M1(mu1+sig1*x,Del1)*\
                M2(mu2+sig2*x,Del2),-8,8)[0]

def R_int(M1,M2,mu1,mu2,Sig1,Sig2,k,x):
    c = np.sign(k)*np.fmin(np.abs(k)/np.sqrt(Sig1*Sig2),1)
    sig1 = np.sign(c)*np.sqrt(Sig1*np.abs(c))
    sig2 = np.sqrt(Sig2*np.abs(c))
    Del1 = Sig1*(1-np.abs(c))
    Del2 = Sig2*(1-np.abs(c))
    return np.exp(-0.5*x**2)/sr2pi*\
                M1(mu1+sig1*x,Del1)*\
                M2(mu2+sig2*x,Del2)

def R_simp(M1,M2,mu1,mu2,Sig1,Sig2,k):
    xs = np.linspace(-8,8,1001)
    return simpson(R_int(M1,M2,mu1,mu2,Sig1,Sig2,k,xs),xs)

def doub_vec(A):
    return np.concatenate([A,A])

def doub_mat(A):
    return np.block([[A,np.zeros_like(A)],[np.zeros_like(A),A]])

def each_diag(A,k=0):
    if k == 0:
        return np.einsum('...jj->...j',A)
    else:
        new_shape = np.array(A.shape[:-1])
        new_shape[-1] -= k
        out = np.zeros(new_shape)
        mult_shape = A.shape[:-2]
        for i in range(np.prod(mult_shape)):
            mult_idx = np.unravel_index(i,mult_shape)
            out[mult_idx] = np.diag(A[mult_idx],k)
        return out

def each_matmul(A,B):
    return np.einsum('ijk,jk->ik',A,B)

def grid_stat(stat,A,Tstat,dt):
    Nsav = A.shape[-1]
    Nstat = round(Tstat/dt)+1
    new_shape = np.array(A.shape)
    new_shape[-1] = Nstat
    A_ext = np.zeros(new_shape)
    if Nsav < Nstat:
        A_ext[...,:Nsav] = A
        A_ext[...,Nsav:] = A[...,-1:]
    else:
        A_ext = A[...,:Nstat]
    mult_shape = A.shape[:-1]
    new_shape = np.concatenate([new_shape,[Nstat]])
    A_mat = np.zeros(new_shape)
    for i in range(np.prod(mult_shape)):
        mult_idx = np.unravel_index(i,mult_shape)
        A_mat[mult_idx] = toeplitz(A_ext[mult_idx])
    return stat(A_mat,axis=(-1,-2))

def d2_stencil(Tsav,dt):
    Nsav = round(Tsav/dt)+1
    d2_mat = np.zeros((Nsav,Nsav))
    d2_mat[(np.arange(Nsav), np.arange(Nsav))] = -2/dt**2
    d2_mat[(np.arange(Nsav-1), np.arange(1,Nsav))] = 1/dt**2
    d2_mat[(np.arange(1,Nsav), np.arange(Nsav-1))] = 1/dt**2
    d2_mat[0,1] = 2/dt**2
    d2_mat[-1,-1] = -1/dt**2
    return d2_mat

def get_time_freq_func(f):
    N = f.shape[-1]
    new_shape = np.array(f.shape)
    new_shape[-1] += N-2
    ft = np.zeros(new_shape)
    ft[...,:N] = f
    ft[...,N:] = f[...,-1:1:-1]
    fo = np.real(np.fft.fft(ft))
    return ft,fo

def smooth_func(f,dt,fcut=17,beta=1):
    N = f.shape[-1]
    _,fo = get_time_freq_func(f)
    fo *= 1/(1 + np.exp((np.abs(np.fft.fftfreq(2*(N-1),dt)) - fcut)*beta))
    return np.real(np.fft.ifft(fo))[...,:N]

def gauss_dmft(tau,muW,SigW,muH,SigH,M_fn,C_fn,Twrm,Tsav,dt,r0=None,Cr0=None):
    Ntyp = len(muH)
    Nint = round((Twrm+Tsav)/dt)+1
    Nclc = round(1.5*Tsav/dt)+1
    Nsav = round(Tsav/dt)+1
    
    r = np.zeros((Ntyp,Nint),dtype=np.float32)
    Cr = np.zeros((Ntyp,Nint,Nint),dtype=np.float32)
    
    if r0 is None:
        r0 = 1e-8*np.ones((Ntyp),dtype=np.float32)
    if Cr0 is None:
        Cr0 = 1e2*np.ones((Ntyp,1),dtype=np.float32)
        
    tauinv = 1/tau
    dttauinv = dt/tau
    dttauinv2 = dttauinv**2
    
    r[:,0] = r0
    
    NCr0 = Cr0.shape[1]
    if Nclc > NCr0:
        Cr[:,0,:NCr0] = Cr0
        Cr[:,0,NCr0:Nclc] = Cr0[:,-1:]
        Cr[:,:NCr0,0] = Cr0
        Cr[:,NCr0:Nclc,0] = Cr0[:,-1:]
    else:
        Cr[:,0,:Nclc] = Cr0[:,:Nclc]
        Cr[:,:Nclc,0] = Cr0[:,:Nclc]
        
    Mphi = np.empty((Ntyp,),dtype=np.float32)
    Cphi = np.empty((Ntyp,),dtype=np.float32)
    
    def drdt(ri,Sigii):
        mui = muW@ri + muH
        M_fn(mui,Sigii,Mphi)
        return tauinv*(-ri + Mphi)
    
    for i in range(Nint-1):
        Crii = Cr[:,i,i]
        Sigii = SigW@Crii + SigH[:,0]
        
        k1 = drdt(r[:,i]          ,Sigii)
        k2 = drdt(r[:,i]+0.5*dt*k1,Sigii)
        k3 = drdt(r[:,i]+0.5*dt*k2,Sigii)
        k4 = drdt(r[:,i]+    dt*k3,Sigii)
        
        r[:,i+1] = r[:,i] + dt/6*(k1+2*k2+2*k3+k4)
        ri = r[:,i]
        mui = muW@ri + muH
        
        if np.any(np.abs(r[:,i+1]) > 1e10) or np.any(np.isnan(r[:,i+1])):
            print("system diverged when integrating r")
            return r,Cr,False

        if i > Nclc-1:
            Cr[:,i+1,i-Nclc] = Cr[:,i,i-Nclc]
            
        for j in range(max(0,i-Nclc),i+1):
            ij_idx = np.fmin(i-j,Nsav-1)
            
            Crij = Cr[:,i,j]
            Sigij = SigW@Crij + SigH[:,ij_idx]
            C_fn(mui,Sigii,Sigij,Cphi)
            Cr[:,i+1,j+1] = Cr[:,i,j+1]+Cr[:,i+1,j]-Cr[:,i,j] +\
                dttauinv*(-Cr[:,i+1,j]-Cr[:,i,j+1]+2*Cr[:,i,j]) + dttauinv2*(-Cr[:,i,j]+Cphi)
                
            Cr[:,i+1,j+1] = np.maximum(Cr[:,i+1,j+1],ri**2)
            
            if np.any(np.abs(Cr[:,i+1,j+1]) > 1e10) or np.any(np.isnan(Cr[:,i+1,j+1])):
                print("system diverged when integrating Cr")
                return r,Cr,False
                
            Cr[:,j+1,i+1] = Cr[:,i+1,j+1]
            
        Ndiv = 5
        if (Ndiv*(i+1)) % (Nint-1) == 0:
            print("{:.2f}% completed".format((i+1)/(Nint-1)))
            
    Cr_diag = each_diag(Cr)
    
    return r,Cr,\
        (np.max(Cr_diag[:,-Nsav:],axis=1)-np.min(Cr_diag[:,-Nsav:],axis=1))/\
            np.mean(Cr_diag[:,-Nsav:],axis=1) < 1e-3

def diff_gauss_dmft(tau,muW,SigW,muH,SigH,R_fn,Twrm,Tsav,dt,r,Cr,Cdr0=None):
    Ntyp = len(muH)
    Nint = round((Twrm+Tsav)/dt)+1
    Nclc = round(1.5*Tsav/dt)+1
    Nsav = round(Tsav/dt)+1
    
    dr = r[Ntyp:] - r[:Ntyp]
    
    Cdr = np.zeros((Ntyp,Nint,Nint),dtype=np.float32)
    
    if Cdr0 is None:
        # Cdr0 = Cr[:Ntyp]
        Cdr0 = dr.astype(np.float32)[:,None]**2 + 1e3
        
    tauinv = 1/tau
    dttauinv = dt/tau
    dttauinv2 = dttauinv**2
    
    doub_muW = doub_mat(muW)
    doub_SigW = doub_mat(SigW)
    
    mu = doub_muW@r + doub_vec(muH)
    Sig = doub_SigW@Cr + doub_vec(SigH)[:,None]
    
    NCdr0 = Cdr0.shape[1]
    if Nclc > NCdr0:
        Cdr[:,0,:NCdr0] = Cdr0
        Cdr[:,0,NCdr0:Nclc] = Cdr0[:,-1:]
        Cdr[:,:NCdr0,0] = Cdr0
        Cdr[:,NCdr0:Nclc,0] = Cdr0[:,-1:]
    else:
        Cdr[:,0,:Nclc] = Cdr0[:,:Nclc]
        Cdr[:,:Nclc,0] = Cdr0[:,:Nclc]
        
    Rphi = np.empty((Ntyp,),dtype=np.float32)
    Cphi = Cr - (doub_vec(tau)**2 - dt*doub_vec(tau))[:,None] * np.einsum('ij,kj->ki',d2_stencil(Tsav,dt),Cr)
    
    for i in range(Nint-1):
        if i > Nclc-1:
            Cdr[:,i+1,i-Nclc] = Cdr[:,i,i-Nclc]
            
        for j in range(max(0,i-Nclc),i+1):
            ij_idx = np.fmin(i-j,Nsav-1)
            
            Cdrij = Cdr[:,i,j]
            Sigdij = SigW@Cdrij
            
            kij = 0.5*(Sig[:Ntyp,ij_idx]+Sig[Ntyp:,ij_idx]-Sigdij)
            
            R_fn(mu[:Ntyp],mu[Ntyp:],Sig[:Ntyp,0],Sig[Ntyp:,0],kij,Rphi)
            
            Cdr[:,i+1,j+1] = Cdr[:,i,j+1]+Cdr[:,i+1,j]-Cdr[:,i,j] +\
                dttauinv*(-Cdr[:,i+1,j]-Cdr[:,i,j+1]+2*Cdr[:,i,j]) +\
                dttauinv2*(-Cdr[:,i,j]+Cphi[:Ntyp,ij_idx]+Cphi[Ntyp:,ij_idx]-2*Rphi)
                
            Cdr[:,i+1,j+1] = np.maximum(Cdr[:,i+1,j+1],dr**2)
            
            if np.any(np.abs(Cdr[:,i+1,j+1]) > 1e10) or np.any(np.isnan(Cdr[:,i+1,j+1])):
                print("system diverged when integrating Cdr")
                return Cdr,False
                
            Cdr[:,j+1,i+1] = Cdr[:,i+1,j+1]
            
        Ndiv = 20
        if (Ndiv*(i+1)) % (Nint-1) == 0:
            print("{:.2f}% completed".format((i+1)/(Nint-1)))
            
    Cdr_diag = each_diag(Cdr)
    
    return Cdr,\
        (np.max(Cdr_diag[:,-Nsav:],axis=1)-np.min(Cdr_diag[:,-Nsav:],axis=1))/\
            np.mean(Cdr_diag[:,-Nsav:],axis=1) < 1e-3

def sparse_dmft(tau,W,K,H,eH,M_fn,C_fn,Twrm,Tsav,dt,r0=None,Cr0=None):
    Ntyp = len(H)
    Nint = round((Twrm+Tsav)/dt)+1
    Nclc = round(1.5*Tsav/dt)+1
    Nsav = round(Tsav/dt)+1
    
    r = np.zeros((Ntyp,Nint),dtype=np.float32)
    Cr = np.zeros((Ntyp,Nint,Nint),dtype=np.float32)
    
    if r0 is None:
        r0 = 1e-8*np.ones((Ntyp),dtype=np.float32)
    if Cr0 is None:
        Cr0 = 1e2*np.ones((Ntyp,1),dtype=np.float32)
        
    tauinv = 1/tau
    dttauinv = dt/tau
    dttauinv2 = dttauinv**2
    
    muW = tau[:,None]*W*K
    SigW = tau[:,None]**2*W**2*K
    
    muH = tau*H
    SigH = (muH*eH)**2
    
    r[:,0] = r0
    
    NCr0 = Cr0.shape[1]
    if Nclc > NCr0:
        Cr[:,0,:NCr0] = Cr0
        Cr[:,0,NCr0:Nclc] = Cr0[:,-1:]
        Cr[:,:NCr0,0] = Cr0
        Cr[:,NCr0:Nclc,0] = Cr0[:,-1:]
    else:
        Cr[:,0,:Nclc] = Cr0[:,:Nclc]
        Cr[:,:Nclc,0] = Cr0[:,:Nclc]
        
    Mphi = np.empty((Ntyp,),dtype=np.float32)
    Cphi = np.empty((Ntyp,),dtype=np.float32)
    
    def drdt(ri,Sigii):
        mui = muW@ri + muH
        M_fn(mui,Sigii,Mphi)
        return tauinv*(-ri + Mphi)
    
    for i in range(Nint-1):
        Crii = Cr[:,i,i]
        Sigii = SigW@Crii + SigH
        
        k1 = drdt(r[:,i]          ,Sigii)
        k2 = drdt(r[:,i]+0.5*dt*k1,Sigii)
        k3 = drdt(r[:,i]+0.5*dt*k2,Sigii)
        k4 = drdt(r[:,i]+    dt*k3,Sigii)
        
        r[:,i+1] = r[:,i] + dt/6*(k1+2*k2+2*k3+k4)
        ri = r[:,i]
        mui = muW@ri + muH
        
        if np.any(np.abs(r[:,i+1]) > 1e10) or np.any(np.isnan(r[:,i+1])):
            print("system diverged when integrating r")
            return r,Cr,False

        if i > Nclc-1:
            Cr[:,i+1,i-Nclc] = Cr[:,i,i-Nclc]
            
        for j in range(max(0,i-Nclc),i+1):
            Crij = Cr[:,i,j]
            Sigij = SigW@Crij + SigH
            C_fn(mui,Sigii,Sigij,Cphi)
            Cr[:,i+1,j+1] = Cr[:,i,j+1]+Cr[:,i+1,j]-Cr[:,i,j] +\
                dttauinv*(-Cr[:,i+1,j]-Cr[:,i,j+1]+2*Cr[:,i,j]) + dttauinv2*(-Cr[:,i,j]+Cphi)
                
            Cr[:,i+1,j+1] = np.maximum(Cr[:,i+1,j+1],ri**2)
            
            if np.any(np.abs(Cr[:,i+1,j+1]) > 1e10) or np.any(np.isnan(Cr[:,i+1,j+1])):
                print("system diverged when integrating Cr")
                return r,Cr,False
                
            Cr[:,j+1,i+1] = Cr[:,i+1,j+1]
            
        Ndiv = 5
        if (Ndiv*(i+1)) % (Nint-1) == 0:
            print("{:.2f}% completed".format((i+1)/(Nint-1)))
            
    Cr_diag = each_diag(Cr)
    
    return r,Cr,\
        (np.max(Cr_diag[:,-Nsav:],axis=1)-np.min(Cr_diag[:,-Nsav:],axis=1))/\
            np.mean(Cr_diag[:,-Nsav:],axis=1) < 1e-3

def doub_sparse_dmft(tau,W,K,H,eH,M_fns,C_fns,Twrm,Tsav,dt,rb0=None,Crb0=None):
    Ntyp = len(H)
    
    doub_tau = doub_vec(tau)
    doub_W = doub_mat(W)
    doub_K = doub_vec(K)
    doub_H = doub_vec(H)
    
    def doub_M(mui,Sigii,out):
        M_fns[0](mui[:Ntyp],Sigii[:Ntyp],out[:Ntyp])
        M_fns[1](mui[Ntyp:],Sigii[Ntyp:],out[Ntyp:])
        
    def doub_C(mui,Sigii,Sigij,out):
        C_fns[0](mui[:Ntyp],Sigii[:Ntyp],Sigij[:Ntyp],out[:Ntyp])
        C_fns[1](mui[Ntyp:],Sigii[Ntyp:],Sigij[Ntyp:],out[Ntyp:])
        
    return sparse_dmft(doub_tau,doub_W,doub_K,doub_H,eH,doub_M,doub_C,Twrm,Tsav,dt,rb0,Crb0)

def diff_sparse_dmft(tau,W,K,H,eH,R_fn,Twrm,Tsav,dt,r,Cr,Cdr0=None):
    Ntyp = len(H)
    Nint = round((Twrm+Tsav)/dt)+1
    Nclc = round(1.5*Tsav/dt)+1
    Nsav = round(Tsav/dt)+1
    
    dr = r[Ntyp:] - r[:Ntyp]
    
    Cdr = np.zeros((Ntyp,Nint,Nint),dtype=np.float32)
    
    if Cdr0 is None:
        # Cdr0 = Cr[:Ntyp]
        Cdr0 = dr.astype(np.float32)[:,None]**2 + 1e3
        
    tauinv = 1/tau
    dttauinv = dt/tau
    dttauinv2 = dttauinv**2
    
    muW = tau[:,None]*W*K
    SigW = tau[:,None]**2*W**2*K
    
    muH = tau*H
    SigH = (muH*eH)**2
    
    doub_muW = doub_mat(muW)
    doub_SigW = doub_mat(SigW)
    
    mu = doub_muW@r + doub_vec(muH)
    Sig = doub_SigW@Cr + doub_vec(SigH)[:,None]
    
    NCdr0 = Cdr0.shape[1]
    if Nclc > NCdr0:
        Cdr[:,0,:NCdr0] = Cdr0
        Cdr[:,0,NCdr0:Nclc] = Cdr0[:,-1:]
        Cdr[:,:NCdr0,0] = Cdr0
        Cdr[:,NCdr0:Nclc,0] = Cdr0[:,-1:]
    else:
        Cdr[:,0,:Nclc] = Cdr0[:,:Nclc]
        Cdr[:,:Nclc,0] = Cdr0[:,:Nclc]
        
    Rphi = np.empty((Ntyp,),dtype=np.float32)
    Cphi = Cr - (doub_vec(tau)**2 - dt*doub_vec(tau))[:,None] * np.einsum('ij,kj->ki',d2_stencil(Tsav,dt),Cr)
    
    for i in range(Nint-1):
        if i > Nclc-1:
            Cdr[:,i+1,i-Nclc] = Cdr[:,i,i-Nclc]
            
        for j in range(max(0,i-Nclc),i+1):
            ij_idx = np.fmin(i-j,Nsav-1)
            
            Cdrij = Cdr[:,i,j]
            Sigdij = SigW@Cdrij
            
            kij = 0.5*(Sig[:Ntyp,ij_idx]+Sig[Ntyp:,ij_idx]-Sigdij)
            
            R_fn(mu[:Ntyp],mu[Ntyp:],Sig[:Ntyp,0],Sig[Ntyp:,0],kij,Rphi)
            
            Cdr[:,i+1,j+1] = Cdr[:,i,j+1]+Cdr[:,i+1,j]-Cdr[:,i,j] +\
                dttauinv*(-Cdr[:,i+1,j]-Cdr[:,i,j+1]+2*Cdr[:,i,j]) +\
                dttauinv2*(-Cdr[:,i,j]+Cphi[:Ntyp,ij_idx]+Cphi[Ntyp:,ij_idx]-2*Rphi)
                
            Cdr[:,i+1,j+1] = np.maximum(Cdr[:,i+1,j+1],dr**2)
            
            if np.any(np.abs(Cdr[:,i+1,j+1]) > 1e10) or np.any(np.isnan(Cdr[:,i+1,j+1])):
                print("system diverged when integrating Cdr")
                return Cdr,False
                
            Cdr[:,j+1,i+1] = Cdr[:,i+1,j+1]
            
        Ndiv = 20
        if (Ndiv*(i+1)) % (Nint-1) == 0:
            print("{:.2f}% completed".format((i+1)/(Nint-1)))
            
    Cdr_diag = each_diag(Cdr)
    
    return Cdr,\
        (np.max(Cdr_diag[:,-Nsav:],axis=1)-np.min(Cdr_diag[:,-Nsav:],axis=1))/\
            np.mean(Cdr_diag[:,-Nsav:],axis=1) < 1e-3
    
def get_solve_width(sa,L=180):
    widths = np.linspace(1,L*3/4,135)
    fbars = basesubwrapnorm(sa,widths,L)
    max_fbar = np.max(fbars)
    min_fbar = np.min(fbars)
    widths_vs_fbars_itp = interp1d(fbars,widths)
    def solve_widths(fbar):
        return widths_vs_fbars_itp(np.fmax(min_fbar,np.fmin(max_fbar,fbar)))
    return solve_widths
    
def get_2feat_solve_width(sa,dori=45,L=180):
    widths = np.linspace(1,L*3/4,135)
    fbars = (basesubwrapnorm(sa,widths,L) + basesubwrapnorm(sa+dori,widths,L)) /\
        (1 + basesubwrapnorm(dori,widths,L))
    max_fbar = np.max(fbars)
    min_fbar = np.min(fbars)
    widths_vs_fbars_itp = interp1d(fbars,widths)
    def solve_widths(fbar):
        return widths_vs_fbars_itp(np.fmax(min_fbar,np.fmin(max_fbar,fbar)))
    return solve_widths

def unstruct_fact(s,L=180):
    return (1/L-wrapnormdens(L/2,s,L))/(wrapnormdens(0,s,L)-wrapnormdens(L/2,s,L))

def struct_fact(x,sconv,sorig,L=180):
    return (wrapnormdens(x,sconv,L)-wrapnormdens(L/2,sorig,L))/\
        (wrapnormdens(0,sorig,L)-wrapnormdens(L/2,sorig,L))
    
def inv_overlap(xs,ss,L=180):
    overlap_mat = basesubwrapnorm(xs[None,:,None]-xs[None,None,:],ss[:,None,:],L)
    return np.linalg.inv(overlap_mat)

def sparse_ring_dmft(tau,W,K,Hb,Hp,eH,sW,sH,sa,M_fn,C_fn,Twrm,Tsav,dt,
                rb0=None,ra0=None,rp0=None,Crb0=None,Cra0=None,Crp0=None,Kb=None,L=180):
    if Kb is None:
        Kb = np.zeros_like(K)
        
    Ntyp = len(Hb)
    Nint = round((Twrm+Tsav)/dt)+1
    Nclc = round(1.5*Tsav/dt)+1
    Nsav = round(Tsav/dt)+1
    
    rb = np.zeros((Ntyp,Nint),dtype=np.float32)
    ra = np.zeros((Ntyp,Nint),dtype=np.float32)
    rp = np.zeros((Ntyp,Nint),dtype=np.float32)
    Crb = np.zeros((Ntyp,Nint,Nint),dtype=np.float32)
    Cra = np.zeros((Ntyp,Nint,Nint),dtype=np.float32)
    Crp = np.zeros((Ntyp,Nint,Nint),dtype=np.float32)
    
    if rb0 is None:
        rb0 = 1*np.ones((Ntyp),dtype=np.float32)
    if ra0 is None:
        ra0 = 2*np.ones((Ntyp),dtype=np.float32)
    if rp0 is None:
        rp0 = 5*np.ones((Ntyp),dtype=np.float32)
    if Crb0 is None:
        Crb0 = 1e2*np.ones((Ntyp,1),dtype=np.float32)
    if Cra0 is None:
        Cra0 = 4e2*np.ones((Ntyp,1),dtype=np.float32)
    if Crp0 is None:
        Crp0 = 25e2*np.ones((Ntyp,1),dtype=np.float32)
        
    tauinv = 1/tau
    dttauinv = dt/tau
    dttauinv2 = dttauinv**2
    
    solve_width = get_solve_width(sa,L)
    
    sW2 = sW**2
        
    muWb = tau[:,None]*W*Kb
    SigWb = tau[:,None]**2*W**2*Kb
    muW = tau[:,None]*W*K
    SigW = tau[:,None]**2*W**2*K
    
    muHb = tau*Hb
    SigHb = (muHb*eH)**2
    muHa = tau*(Hb+(Hp-Hb)*basesubwrapnorm(sa,sH,L))
    SigHa = (muHa*eH)**2
    muHp = tau*Hp
    SigHp = (muHp*eH)**2
    
    rb[:,0] = rb0
    ra[:,0] = ra0
    rp[:,0] = rp0
    
    NCr0 = Crb0.shape[1]
    if Nclc > NCr0:
        Crb[:,0,:NCr0] = Crb0
        Crb[:,0,NCr0:Nclc] = Crb0[:,-1:]
        Crb[:,:NCr0,0] = Crb0
        Crb[:,NCr0:Nclc,0] = Crb0[:,-1:]
        
        Cra[:,0,:NCr0] = Cra0
        Cra[:,0,NCr0:Nclc] = Cra0[:,-1:]
        Cra[:,:NCr0,0] = Cra0
        Cra[:,NCr0:Nclc,0] = Cra0[:,-1:]
        
        Crp[:,0,:NCr0] = Crp0
        Crp[:,0,NCr0:Nclc] = Crp0[:,-1:]
        Crp[:,:NCr0,0] = Crp0
        Crp[:,NCr0:Nclc,0] = Crp0[:,-1:]
    else:
        Crb[:,0,:Nclc] = Crb0[:,:Nclc]
        Crb[:,:Nclc,0] = Crb0[:,:Nclc]
        
        Cra[:,0,:Nclc] = Cra0[:,:Nclc]
        Cra[:,:Nclc,0] = Cra0[:,:Nclc]
        
        Crp[:,0,:Nclc] = Crp0[:,:Nclc]
        Crp[:,:Nclc,0] = Crp0[:,:Nclc]
        
    Mphib = np.empty((Ntyp,),dtype=np.float32)
    Mphia = np.empty((Ntyp,),dtype=np.float32)
    Mphip = np.empty((Ntyp,),dtype=np.float32)
    Cphib = np.empty((Ntyp,),dtype=np.float32)
    Cphia = np.empty((Ntyp,),dtype=np.float32)
    Cphip = np.empty((Ntyp,),dtype=np.float32)
    
    def drdt(rbi,rai,rpi,Sigbii,Sigaii,Sigpii):
        sri = solve_width((rai-rbi)/(rpi-rbi))
        sWri = np.sqrt(sW2+sri**2)
        rpmbi = rpi - rbi
        mubi = (muW+muWb)@rbi + (unstruct_fact(sri,L)*muWb)@rpmbi + muHb
        muai = mubi + (struct_fact(sa,sWri,sri,L)*muW)@rpmbi + muHa-muHb
        mupi = mubi + (struct_fact(0,sWri,sri,L)*muW)@rpmbi + muHp-muHb
        mubi = mubi + (struct_fact(L/2,sWri,sri,L)*muW)@rpmbi
        M_fn(mubi,Sigbii,Mphib)
        M_fn(muai,Sigaii,Mphia)
        M_fn(mupi,Sigpii,Mphip)
        return tauinv*(-rbi + Mphib), tauinv*(-rai + Mphia),  tauinv*(-rpi + Mphip)
    
    for i in range(Nint-1):
        Crbii = Crb[:,i,i]
        Craii = Cra[:,i,i]
        Crpii = Crp[:,i,i]
        sCrii = solve_width((Craii-Crbii)/(Crpii-Crbii))
        sWCrii = np.sqrt(sW2+sCrii**2)
        Crpmbii = Crpii - Crbii
        Sigbii = (SigW+SigWb)@Crbii + (unstruct_fact(sCrii,L)*SigWb)@Crpmbii + SigHb
        Sigaii = Sigbii + (struct_fact(sa,sWCrii,sCrii,L)*SigW)@Crpmbii + SigHa-SigHb
        Sigpii = Sigbii + (struct_fact(0,sWCrii,sCrii,L)*SigW)@Crpmbii + SigHp-SigHb
        Sigbii = Sigbii + (struct_fact(L/2,sWCrii,sCrii,L)*SigW)@Crpmbii
            
        kb1,ka1,kp1 = drdt(rb[:,i]           ,ra[:,i]           ,rp[:,i]           ,Sigbii,Sigaii,Sigpii)
        kb2,ka2,kp2 = drdt(rb[:,i]+0.5*dt*kb1,ra[:,i]+0.5*dt*kb1,rp[:,i]+0.5*dt*kb1,Sigbii,Sigaii,Sigpii)
        kb3,ka3,kp3 = drdt(rb[:,i]+0.5*dt*kb2,ra[:,i]+0.5*dt*kb2,rp[:,i]+0.5*dt*kb2,Sigbii,Sigaii,Sigpii)
        kb4,ka4,kp4 = drdt(rb[:,i]+    dt*kb2,ra[:,i]+    dt*kb2,rp[:,i]+    dt*kb2,Sigbii,Sigaii,Sigpii)
        
        rb[:,i+1] = rb[:,i] + dt/6*(kb1+2*kb2+2*kb3+kb4)
        ra[:,i+1] = ra[:,i] + dt/6*(ka1+2*ka2+2*ka3+ka4)
        rp[:,i+1] = rp[:,i] + dt/6*(kp1+2*kp2+2*kp3+kp4)
        rbi = rb[:,i]
        rai = ra[:,i]
        rpi = rp[:,i]
        sri = solve_width((rai-rbi)/(rpi-rbi))
        sWri = np.sqrt(sW2+sri**2)
        rpmbi = rpi - rbi
        mubi = (muW+muWb)@rbi + (unstruct_fact(sri,L)*muWb)@rpmbi + muHb
        muai = mubi + (struct_fact(sa,sWri,sri,L)*muW)@rpmbi + muHa-muHb
        mupi = mubi + (struct_fact(0,sWri,sri,L)*muW)@rpmbi + muHp-muHb
        mubi = mubi + (struct_fact(L/2,sWri,sri,L)*muW)@rpmbi
        
        if np.any(np.abs(rb[:,i+1]) > 1e10) or np.any(np.isnan(rb[:,i+1])):
            print(mubi,muai,mupi,sri)
            print(Sigbii,Sigaii,Sigpii,sCrii)
            print("system diverged when integrating rb")
            return rb,ra,rp,Crb,Cra,Crp,False,False,False
        if np.any(np.abs(ra[:,i+1]) > 1e10) or np.any(np.isnan(ra[:,i+1])):
            print(mubi,muai,mupi,sri)
            print(Sigbii,Sigaii,Sigpii,sCrii)
            print("system diverged when integrating ra")
            return rb,ra,rp,Crb,Cra,Crp,False,False,False
        if np.any(np.abs(rp[:,i+1]) > 1e10) or np.any(np.isnan(rp[:,i+1])):
            print(mubi,muai,mupi,sri)
            print(Sigbii,Sigaii,Sigpii,sCrii)
            print("system diverged when integrating rp")
            return rb,ra,rp,Crb,Cra,Crp,False,False,False

        if i > Nclc-1:
            Crb[:,i+1,i-Nclc] = Crb[:,i,i-Nclc]
            Cra[:,i+1,i-Nclc] = Cra[:,i,i-Nclc]
            Crp[:,i+1,i-Nclc] = Crp[:,i,i-Nclc]
            
        for j in range(max(0,i-Nclc),i+1):
            Crbij = Crb[:,i,j]
            Craij = Cra[:,i,j]
            Crpij = Crp[:,i,j]
            sCrij = solve_width((Craij-Crbij)/(Crpij-Crbij))
            sWCrij = np.sqrt(sW2+sCrij**2)
            Crpmbij = Crpij - Crbij
            Sigbij = (SigW+SigWb)@Crbij + (unstruct_fact(sCrij,L)*SigWb)@Crpmbij + SigHb
            Sigaij = Sigbij + (struct_fact(sa,sWCrij,sCrij,L)*SigW)@Crpmbij + SigHa-SigHb
            Sigpij = Sigbij + (struct_fact(0,sWCrij,sCrij,L)*SigW)@Crpmbij + SigHp-SigHb
            Sigbij = Sigbij + (struct_fact(L/2,sWCrij,sCrij,L)*SigW)@Crpmbij
            C_fn(mubi,Sigbii,Sigbij,Cphib)
            C_fn(muai,Sigaii,Sigaij,Cphia)
            C_fn(mupi,Sigpii,Sigpij,Cphip)
            Crb[:,i+1,j+1] = Crb[:,i,j+1]+Crb[:,i+1,j]-Crb[:,i,j] +\
                dttauinv*(-Crb[:,i+1,j]-Crb[:,i,j+1]+2*Crb[:,i,j]) + dttauinv2*(-Crb[:,i,j]+Cphib)
            Cra[:,i+1,j+1] = Cra[:,i,j+1]+Cra[:,i+1,j]-Cra[:,i,j] +\
                dttauinv*(-Cra[:,i+1,j]-Cra[:,i,j+1]+2*Cra[:,i,j]) + dttauinv2*(-Cra[:,i,j]+Cphia)
            Crp[:,i+1,j+1] = Crp[:,i,j+1]+Crp[:,i+1,j]-Crp[:,i,j] +\
                dttauinv*(-Crp[:,i+1,j]-Crp[:,i,j+1]+2*Crp[:,i,j]) + dttauinv2*(-Crp[:,i,j]+Cphip)
                
            Crb[:,i+1,j+1] = np.maximum(Crb[:,i+1,j+1],rbi**2)
            Cra[:,i+1,j+1] = np.maximum(Cra[:,i+1,j+1],rai**2)
            Crp[:,i+1,j+1] = np.maximum(Crp[:,i+1,j+1],rpi**2)
            
            if np.any(np.abs(Crb[:,i+1,j+1]) > 1e10) or np.any(np.isnan(Crb[:,i+1,j+1])):
                print(mubi,muai,mupi,sri)
                print(Sigbii,Sigaii,Sigpii,sCrii)
                print(Sigbij,Sigaij,Sigpij,sCrij)
                print("system diverged when integrating Crb")
                return rb,ra,rp,Crb,Cra,Crp,False,False,False
            if np.any(np.abs(Cra[:,i+1,j+1]) > 1e10) or np.any(np.isnan(Cra[:,i+1,j+1])):
                print(mubi,muai,mupi,sri)
                print(Sigbii,Sigaii,Sigpii,sCrii)
                print(Sigbij,Sigaij,Sigpij,sCrij)
                print("system diverged when integrating Cra")
                return rb,ra,rp,Crb,Cra,Crp,False,False,False
            if np.any(np.abs(Crp[:,i+1,j+1]) > 1e10) or np.any(np.isnan(Crp[:,i+1,j+1])):
                print(mubi,muai,mupi,sri)
                print(Sigbii,Sigaii,Sigpii,sCrii)
                print(Sigbij,Sigaij,Sigpij,sCrij)
                print("system diverged when integrating Crp")
                return rb,ra,rp,Crb,Cra,Crp,False,False,False
                
            Crb[:,j+1,i+1] = Crb[:,i+1,j+1]
            Cra[:,j+1,i+1] = Cra[:,i+1,j+1]
            Crp[:,j+1,i+1] = Crp[:,i+1,j+1]
            
        Ndiv = 5
        if (Ndiv*(i+1)) % (Nint-1) == 0:
            print("{:.2f}% completed".format((i+1)/(Nint-1)))
            
    Crb_diag = each_diag(Crb)
    Cra_diag = each_diag(Cra)
    Crp_diag = each_diag(Crp)
    
    return rb,ra,rp,Crb,Cra,Crp,\
        (np.max(Crb_diag[:,-Nsav:],axis=1)-np.min(Crb_diag[:,-Nsav:],axis=1))/\
            np.mean(Crb_diag[:,-Nsav:],axis=1) < 1e-3,\
        (np.max(Cra_diag[:,-Nsav:],axis=1)-np.min(Cra_diag[:,-Nsav:],axis=1))/\
            np.mean(Cra_diag[:,-Nsav:],axis=1) < 1e-3,\
        (np.max(Crp_diag[:,-Nsav:],axis=1)-np.min(Crp_diag[:,-Nsav:],axis=1))/\
            np.mean(Crp_diag[:,-Nsav:],axis=1) < 1e-3

def sparse_2feat_ring_dmft(tau,W,K,Hb,Hp,eH,sW,sH,sa,M_fn,C_fn,Twrm,Tsav,dt,
                rb0=None,ra0=None,rp0=None,Crb0=None,Cra0=None,Crp0=None,Kb=None,dori=45,L=180):
    if Kb is None:
        Kb = np.zeros_like(K)
        
    Ntyp = len(Hb)
    Nint = round((Twrm+Tsav)/dt)+1
    Nclc = round(1.5*Tsav/dt)+1
    Nsav = round(Tsav/dt)+1
    
    rb = np.zeros((Ntyp,Nint),dtype=np.float32)
    ra = np.zeros((Ntyp,Nint),dtype=np.float32)
    rp = np.zeros((Ntyp,Nint),dtype=np.float32)
    Crb = np.zeros((Ntyp,Nint,Nint),dtype=np.float32)
    Cra = np.zeros((Ntyp,Nint,Nint),dtype=np.float32)
    Crp = np.zeros((Ntyp,Nint,Nint),dtype=np.float32)
    
    if rb0 is None:
        rb0 = 1*np.ones((Ntyp),dtype=np.float32)
    if ra0 is None:
        ra0 = 2*np.ones((Ntyp),dtype=np.float32)
    if rp0 is None:
        rp0 = 5*np.ones((Ntyp),dtype=np.float32)
    if Crb0 is None:
        Crb0 = 1e2*np.ones((Ntyp,1),dtype=np.float32)
    if Cra0 is None:
        Cra0 = 4e2*np.ones((Ntyp,1),dtype=np.float32)
    if Crp0 is None:
        Crp0 = 25e2*np.ones((Ntyp,1),dtype=np.float32)
        
    tauinv = 1/tau
    dttauinv = dt/tau
    dttauinv2 = dttauinv**2
    
    xpeaks = np.array([0,-dori])
    solve_width = get_2feat_solve_width(sa,dori,L)
    
    sW2 = sW**2
        
    muWb = tau[:,None]*W*Kb
    SigWb = tau[:,None]**2*W**2*Kb
    muW = tau[:,None]*W*K
    SigW = tau[:,None]**2*W**2*K
    
    muHb = tau*Hb
    SigHb = (muHb*eH)**2
    muHa = tau*(Hb+(Hp-Hb)*basesubwrapnorm(sa,sH,L))
    SigHa = (muHa*eH)**2
    muHp = tau*Hp
    SigHp = (muHp*eH)**2
    
    rb[:,0] = rb0
    ra[:,0] = ra0
    rp[:,0] = rp0
    
    NCr0 = Crb0.shape[1]
    if Nclc > NCr0:
        Crb[:,0,:NCr0] = Crb0
        Crb[:,0,NCr0:Nclc] = Crb0[:,-1:]
        Crb[:,:NCr0,0] = Crb0
        Crb[:,NCr0:Nclc,0] = Crb0[:,-1:]
        
        Cra[:,0,:NCr0] = Cra0
        Cra[:,0,NCr0:Nclc] = Cra0[:,-1:]
        Cra[:,:NCr0,0] = Cra0
        Cra[:,NCr0:Nclc,0] = Cra0[:,-1:]
        
        Crp[:,0,:NCr0] = Crp0
        Crp[:,0,NCr0:Nclc] = Crp0[:,-1:]
        Crp[:,:NCr0,0] = Crp0
        Crp[:,NCr0:Nclc,0] = Crp0[:,-1:]
    else:
        Crb[:,0,:Nclc] = Crb0[:,:Nclc]
        Crb[:,:Nclc,0] = Crb0[:,:Nclc]
        
        Cra[:,0,:Nclc] = Cra0[:,:Nclc]
        Cra[:,:Nclc,0] = Cra0[:,:Nclc]
        
        Crp[:,0,:Nclc] = Crp0[:,:Nclc]
        Crp[:,:Nclc,0] = Crp0[:,:Nclc]
        
    Mphib = np.empty((Ntyp,),dtype=np.float32)
    Mphia = np.empty((Ntyp,),dtype=np.float32)
    Mphip = np.empty((Ntyp,),dtype=np.float32)
    Cphib = np.empty((Ntyp,),dtype=np.float32)
    Cphia = np.empty((Ntyp,),dtype=np.float32)
    Cphip = np.empty((Ntyp,),dtype=np.float32)
    
    rOinv = np.empty((Ntyp,),dtype=np.float32)
    CrOinv = np.empty((Ntyp,),dtype=np.float32)
    
    def drdt(rbi,rai,rpi,Sigbii,Sigaii,Sigpii):
        sri = solve_width((rai-rbi)/(rpi-rbi))
        rOinv[:] = np.sum(inv_overlap(xpeaks,sri[:,None])[:,:,0],-1)
        sWri = np.sqrt(sW2+sri**2)
        rpmbi = (rpi - rbi)*rOinv
        mubi = (muW+muWb)@rbi + (unstruct_fact(sri,L)*muWb)@rpmbi + muHb
        muai = mubi + ((struct_fact(sa,sWri,sri,L)+struct_fact(sa+dori,sWri,sri,L))*muW)@rpmbi + muHa-muHb
        mupi = mubi + ((struct_fact(0,sWri,sri,L)+struct_fact(dori,sWri,sri,L))*muW)@rpmbi + muHp-muHb
        mubi = mubi + (2*struct_fact(L/2,sWri,sri,L)*muW)@rpmbi
        M_fn(mubi,Sigbii,Mphib)
        M_fn(muai,Sigaii,Mphia)
        M_fn(mupi,Sigpii,Mphip)
        return tauinv*(-rbi + Mphib), tauinv*(-rai + Mphia),  tauinv*(-rpi + Mphip)
    
    for i in range(Nint-1):
        Crbii = Crb[:,i,i]
        Craii = Cra[:,i,i]
        Crpii = Crp[:,i,i]
        sCrii = solve_width((Craii-Crbii)/(Crpii-Crbii))
        CrOinv[:] = np.sum(inv_overlap(xpeaks,sCrii[:,None])[:,:,0],-1)
        sWCrii = np.sqrt(sW2+sCrii**2)
        Crpmbii = (Crpii - Crbii)*CrOinv
        Sigbii = (SigW+SigWb)@Crbii + (unstruct_fact(sCrii,L)*SigWb)@Crpmbii + SigHb
        Sigaii = Sigbii + ((struct_fact(sa,sWCrii,sCrii,L)+\
            struct_fact(sa+dori,sWCrii,sCrii,L))*SigW)@Crpmbii + SigHa-SigHb
        Sigpii = Sigbii + ((struct_fact(0,sWCrii,sCrii,L)+\
            struct_fact(dori,sWCrii,sCrii,L))*SigW)@Crpmbii + SigHp-SigHb
        Sigbii = Sigbii + (2*struct_fact(L/2,sWCrii,sCrii,L)*SigW)@Crpmbii
            
        kb1,ka1,kp1 = drdt(rb[:,i]           ,ra[:,i]           ,rp[:,i]           ,Sigbii,Sigaii,Sigpii)
        kb2,ka2,kp2 = drdt(rb[:,i]+0.5*dt*kb1,ra[:,i]+0.5*dt*kb1,rp[:,i]+0.5*dt*kb1,Sigbii,Sigaii,Sigpii)
        kb3,ka3,kp3 = drdt(rb[:,i]+0.5*dt*kb2,ra[:,i]+0.5*dt*kb2,rp[:,i]+0.5*dt*kb2,Sigbii,Sigaii,Sigpii)
        kb4,ka4,kp4 = drdt(rb[:,i]+    dt*kb2,ra[:,i]+    dt*kb2,rp[:,i]+    dt*kb2,Sigbii,Sigaii,Sigpii)
        
        rb[:,i+1] = rb[:,i] + dt/6*(kb1+2*kb2+2*kb3+kb4)
        ra[:,i+1] = ra[:,i] + dt/6*(ka1+2*ka2+2*ka3+ka4)
        rp[:,i+1] = rp[:,i] + dt/6*(kp1+2*kp2+2*kp3+kp4)
        rbi = rb[:,i]
        rai = ra[:,i]
        rpi = rp[:,i]
        sri = solve_width((rai-rbi)/(rpi-rbi))
        rOinv[:] = np.sum(inv_overlap(xpeaks,sri[:,None])[:,:,0],-1)
        sWri = np.sqrt(sW2+sri**2)
        rpmbi = (rpi - rbi)*rOinv
        mubi = (muW+muWb)@rbi + (unstruct_fact(sri,L)*muWb)@rpmbi + muHb
        muai = mubi + ((struct_fact(sa,sWri,sri,L)+struct_fact(sa+dori,sWri,sri,L))*muW)@rpmbi + muHa-muHb
        mupi = mubi + ((struct_fact(0,sWri,sri,L)+struct_fact(dori,sWri,sri,L))*muW)@rpmbi + muHp-muHb
        mubi = mubi + (2*struct_fact(L/2,sWri,sri,L)*muW)@rpmbi
        
        if np.any(np.abs(rb[:,i+1]) > 1e10) or np.any(np.isnan(rb[:,i+1])):
            print(mubi,muai,mupi,sri)
            print(Sigbii,Sigaii,Sigpii,sCrii)
            print("system diverged when integrating rb")
            return rb,ra,rp,Crb,Cra,Crp,False,False,False
        if np.any(np.abs(ra[:,i+1]) > 1e10) or np.any(np.isnan(ra[:,i+1])):
            print(mubi,muai,mupi,sri)
            print(Sigbii,Sigaii,Sigpii,sCrii)
            print("system diverged when integrating ra")
            return rb,ra,rp,Crb,Cra,Crp,False,False,False
        if np.any(np.abs(rp[:,i+1]) > 1e10) or np.any(np.isnan(rp[:,i+1])):
            print(mubi,muai,mupi,sri)
            print(Sigbii,Sigaii,Sigpii,sCrii)
            print("system diverged when integrating rp")
            return rb,ra,rp,Crb,Cra,Crp,False,False,False

        if i > Nclc-1:
            Crb[:,i+1,i-Nclc] = Crb[:,i,i-Nclc]
            Cra[:,i+1,i-Nclc] = Cra[:,i,i-Nclc]
            Crp[:,i+1,i-Nclc] = Crp[:,i,i-Nclc]
            
        for j in range(max(0,i-Nclc),i+1):
            Crbij = Crb[:,i,j]
            Craij = Cra[:,i,j]
            Crpij = Crp[:,i,j]
            sCrij = solve_width((Craij-Crbij)/(Crpij-Crbij))
            CrOinv[:] = np.sum(inv_overlap(xpeaks,sCrij[:,None])[:,:,0],-1)
            sWCrij = np.sqrt(sW2+sCrij**2)
            Crpmbij = (Crpij - Crbij)*CrOinv
            Sigbij = (SigW+SigWb)@Crbij + (unstruct_fact(sCrij,L)*SigWb)@Crpmbij + SigHb
            Sigaij = Sigbij + ((struct_fact(sa,sWCrij,sCrij,L)+\
                                struct_fact(sa+dori,sWCrij,sCrij,L))*SigW)@Crpmbij + SigHa-SigHb
            Sigpij = Sigbij + ((struct_fact(0,sWCrij,sCrij,L)+\
                                struct_fact(dori,sWCrij,sCrij,L))*SigW)@Crpmbij + SigHp-SigHb
            Sigbij = Sigbij + (2*struct_fact(L/2,sWCrij,sCrij,L)*SigW)@Crpmbij
            C_fn(mubi,Sigbii,Sigbij,Cphib)
            C_fn(muai,Sigaii,Sigaij,Cphia)
            C_fn(mupi,Sigpii,Sigpij,Cphip)
            Crb[:,i+1,j+1] = Crb[:,i,j+1]+Crb[:,i+1,j]-Crb[:,i,j] +\
                dttauinv*(-Crb[:,i+1,j]-Crb[:,i,j+1]+2*Crb[:,i,j]) + dttauinv2*(-Crb[:,i,j]+Cphib)
            Cra[:,i+1,j+1] = Cra[:,i,j+1]+Cra[:,i+1,j]-Cra[:,i,j] +\
                dttauinv*(-Cra[:,i+1,j]-Cra[:,i,j+1]+2*Cra[:,i,j]) + dttauinv2*(-Cra[:,i,j]+Cphia)
            Crp[:,i+1,j+1] = Crp[:,i,j+1]+Crp[:,i+1,j]-Crp[:,i,j] +\
                dttauinv*(-Crp[:,i+1,j]-Crp[:,i,j+1]+2*Crp[:,i,j]) + dttauinv2*(-Crp[:,i,j]+Cphip)
                
            Crb[:,i+1,j+1] = np.maximum(Crb[:,i+1,j+1],rbi**2)
            Cra[:,i+1,j+1] = np.maximum(Cra[:,i+1,j+1],rai**2)
            Crp[:,i+1,j+1] = np.maximum(Crp[:,i+1,j+1],rpi**2)
            
            if np.any(np.abs(Crb[:,i+1,j+1]) > 1e10) or np.any(np.isnan(Crb[:,i+1,j+1])):
                print(mubi,muai,mupi,sri)
                print(Sigbii,Sigaii,Sigpii,sCrii)
                print(Sigbij,Sigaij,Sigpij,sCrij)
                print("system diverged when integrating Crb")
                return rb,ra,rp,Crb,Cra,Crp,False,False,False
            if np.any(np.abs(Cra[:,i+1,j+1]) > 1e10) or np.any(np.isnan(Cra[:,i+1,j+1])):
                print(mubi,muai,mupi,sri)
                print(Sigbii,Sigaii,Sigpii,sCrii)
                print(Sigbij,Sigaij,Sigpij,sCrij)
                print("system diverged when integrating Cra")
                return rb,ra,rp,Crb,Cra,Crp,False,False,False
            if np.any(np.abs(Crp[:,i+1,j+1]) > 1e10) or np.any(np.isnan(Crp[:,i+1,j+1])):
                print(mubi,muai,mupi,sri)
                print(Sigbii,Sigaii,Sigpii,sCrii)
                print(Sigbij,Sigaij,Sigpij,sCrij)
                print("system diverged when integrating Crp")
                return rb,ra,rp,Crb,Cra,Crp,False,False,False
                
            Crb[:,j+1,i+1] = Crb[:,i+1,j+1]
            Cra[:,j+1,i+1] = Cra[:,i+1,j+1]
            Crp[:,j+1,i+1] = Crp[:,i+1,j+1]
            
        Ndiv = 5
        if (Ndiv*(i+1)) % (Nint-1) == 0:
            print("{:.2f}% completed".format((i+1)/(Nint-1)))
            
    Crb_diag = each_diag(Crb)
    Cra_diag = each_diag(Cra)
    Crp_diag = each_diag(Crp)
    
    return rb,ra,rp,Crb,Cra,Crp,\
        (np.max(Crb_diag[:,-Nsav:],axis=1)-np.min(Crb_diag[:,-Nsav:],axis=1))/\
            np.mean(Crb_diag[:,-Nsav:],axis=1) < 1e-3,\
        (np.max(Cra_diag[:,-Nsav:],axis=1)-np.min(Cra_diag[:,-Nsav:],axis=1))/\
            np.mean(Cra_diag[:,-Nsav:],axis=1) < 1e-3,\
        (np.max(Crp_diag[:,-Nsav:],axis=1)-np.min(Crp_diag[:,-Nsav:],axis=1))/\
            np.mean(Crp_diag[:,-Nsav:],axis=1) < 1e-3

def doub_sparse_ring_dmft(tau,W,K,Hb,Hp,eH,sW,sH,sa,M_fns,C_fns,Twrm,Tsav,dt,
                     rb0=None,ra0=None,rp0=None,Crb0=None,Cra0=None,Crp0=None,Kb=None,L=180):
    if Kb is None:
        doub_Kb = None
    else:
        doub_Kb = doub_vec(Kb)
        
    Ntyp = len(Hb)
    
    doub_tau = doub_vec(tau)
    doub_W = doub_mat(W)
    doub_K = doub_vec(K)
    doub_Hb = doub_vec(Hb)
    doub_Hp = doub_vec(Hp)
    doub_sW = doub_mat(sW)
    doub_sH = doub_vec(sH)
    
    def doub_M(mui,Sigii,out):
        M_fns[0](mui[:Ntyp],Sigii[:Ntyp],out[:Ntyp])
        M_fns[1](mui[Ntyp:],Sigii[Ntyp:],out[Ntyp:])
        
    def doub_C(mui,Sigii,Sigij,out):
        C_fns[0](mui[:Ntyp],Sigii[:Ntyp],Sigij[:Ntyp],out[:Ntyp])
        C_fns[1](mui[Ntyp:],Sigii[Ntyp:],Sigij[Ntyp:],out[Ntyp:])
        
    return sparse_ring_dmft(doub_tau,doub_W,doub_K,doub_Hb,doub_Hp,eH,doub_sW,doub_sH,sa,doub_M,doub_C,Twrm,Tsav,dt,
                      rb0,ra0,rp0,Crb0,Cra0,Crp0,Kb=doub_Kb,L=L)

def doub_sparse_2feat_ring_dmft(tau,W,K,Hb,Hp,eH,sW,sH,sa,M_fns,C_fns,Twrm,Tsav,dt,
                     rb0=None,ra0=None,rp0=None,Crb0=None,Cra0=None,Crp0=None,Kb=None,dori=45,L=180):
    if Kb is None:
        doub_Kb = None
    else:
        doub_Kb = doub_vec(Kb)
        
    Ntyp = len(Hb)
    
    doub_tau = doub_vec(tau)
    doub_W = doub_mat(W)
    doub_K = doub_vec(K)
    doub_Hb = doub_vec(Hb)
    doub_Hp = doub_vec(Hp)
    doub_sW = doub_mat(sW)
    doub_sH = doub_vec(sH)
    
    def doub_M(mui,Sigii,out):
        M_fns[0](mui[:Ntyp],Sigii[:Ntyp],out[:Ntyp])
        M_fns[1](mui[Ntyp:],Sigii[Ntyp:],out[Ntyp:])
        
    def doub_C(mui,Sigii,Sigij,out):
        C_fns[0](mui[:Ntyp],Sigii[:Ntyp],Sigij[:Ntyp],out[:Ntyp])
        C_fns[1](mui[Ntyp:],Sigii[Ntyp:],Sigij[Ntyp:],out[Ntyp:])
        
    return sparse_2feat_ring_dmft(doub_tau,doub_W,doub_K,doub_Hb,doub_Hp,eH,doub_sW,doub_sH,sa,doub_M,doub_C,
                                  Twrm,Tsav,dt,rb0,ra0,rp0,Crb0,Cra0,Crp0,Kb=doub_Kb,dori=dori,L=L)

def diff_sparse_ring_dmft(tau,W,K,Hb,Hp,eH,sW,sH,sa,R_fn,Twrm,Tsav,dt,rb,ra,rp,Crb,Cra,Crp,
                     Cdrb0=None,Cdra0=None,Cdrp0=None,Kb=None,L=180):
    if Kb is None:
        Kb = np.zeros_like(K)
        
    Ntyp = len(Hb)
    Nint = round((Twrm+Tsav)/dt)+1
    Nclc = round(1.5*Tsav/dt)+1
    Nsav = round(Tsav/dt)+1
    
    drb = rb[Ntyp:] - rb[:Ntyp]
    dra = ra[Ntyp:] - ra[:Ntyp]
    drp = rp[Ntyp:] - rp[:Ntyp]
    
    Cdrb = np.zeros((Ntyp,Nint,Nint),dtype=np.float32)
    Cdra = np.zeros((Ntyp,Nint,Nint),dtype=np.float32)
    Cdrp = np.zeros((Ntyp,Nint,Nint),dtype=np.float32)
    
    if Cdrb0 is None:
        # Cdrb0 = Crb[:Ntyp]
        Cdrb0 = (rb[Ntyp:] - rb[:Ntyp]).astype(np.float32)[:,None]**2 + 1e3
    if Cdra0 is None:
        # Cdra0 = Cra[:Ntyp]
        Cdra0 = (ra[Ntyp:] - ra[:Ntyp]).astype(np.float32)[:,None]**2 + 2e3
    if Cdrp0 is None:
        # Cdrp0 = Crp[:Ntyp]
        Cdrp0 = (rp[Ntyp:] - rp[:Ntyp]).astype(np.float32)[:,None]**2 + 5e3
        
    tauinv = 1/tau
    dttauinv = dt/tau
    dttauinv2 = dttauinv**2
    
    sW2 = sW**2
    
    solve_width = get_solve_width(sa,L)
    
    muW = tau[:,None]*W*K
    SigW = tau[:,None]**2*W**2*K
    muWb = tau[:,None]*W*Kb
    SigWb = tau[:,None]**2*W**2*Kb
    
    muHb = tau*Hb
    SigHb = (muHb*eH)**2
    muHa = tau*(Hb+(Hp-Hb)*basesubwrapnorm(sa,sH,L))
    SigHa = (muHa*eH)**2
    muHp = tau*Hp
    SigHp = (muHp*eH)**2
    
    doub_muW = doub_mat(muW)
    doub_SigW = doub_mat(SigW)
    doub_muWb = doub_mat(muWb)
    doub_SigWb = doub_mat(SigWb)
    
    sr = solve_width((ra-rb)/(rp-rb))
    sWr = np.sqrt(doub_mat(sW2)+sr**2)
    rpmb = rp - rb
    sCr = solve_width((Cra-Crb)/(Crp-Crb))
    sWCr = np.sqrt(doub_mat(sW2)[:,:,None]+sCr[None,:,:]**2)
    Crpmb = Crp - Crb
    mub = (doub_muW+doub_muWb)@rb + (unstruct_fact(sr,L)*doub_muWb)@rpmb + doub_vec(muHb)
    mua = mub + (struct_fact(sa,sWr,sr,L)*doub_muW)@rpmb + doub_vec(muHa-muHb)
    mup = mub + (struct_fact(0,sWr,sr,L)*doub_muW)@rpmb + doub_vec(muHp-muHb)
    mub = mub + (struct_fact(L/2,sWr,sr,L)*doub_muW)@rpmb
    Sigb = (doub_SigW+doub_SigWb)@Crb + each_matmul(unstruct_fact(sCr,L)*doub_SigWb[:,:,None],Crpmb) +\
        doub_vec(SigHb)[:,None]
    Siga = Sigb + each_matmul(struct_fact(sa,sWCr,sCr,L)*doub_SigW[:,:,None],Crpmb) + doub_vec(SigHa-SigHb)[:,None]
    Sigp = Sigb + each_matmul(struct_fact(0,sWCr,sCr,L)*doub_SigW[:,:,None],Crpmb) + doub_vec(SigHp-SigHb)[:,None]
    Sigb = Sigb + each_matmul(struct_fact(L/2,sWCr,sCr,L)*doub_SigW[:,:,None],Crpmb)
    
    NCdr0 = Cdrb0.shape[1]
    if Nclc > NCdr0:
        Cdrb[:,0,:NCdr0] = Cdrb0
        Cdrb[:,0,NCdr0:Nclc] = Cdrb0[:,-1:]
        Cdrb[:,:NCdr0,0] = Cdrb0
        Cdrb[:,NCdr0:Nclc,0] = Cdrb0[:,-1:]
        
        Cdra[:,0,:NCdr0] = Cdra0
        Cdra[:,0,NCdr0:Nclc] = Cdra0[:,-1:]
        Cdra[:,:NCdr0,0] = Cdra0
        Cdra[:,NCdr0:Nclc,0] = Cdra0[:,-1:]
        
        Cdrp[:,0,:NCdr0] = Cdrp0
        Cdrp[:,0,NCdr0:Nclc] = Cdrp0[:,-1:]
        Cdrp[:,:NCdr0,0] = Cdrp0
        Cdrp[:,NCdr0:Nclc,0] = Cdrp0[:,-1:]
    else:
        Cdrb[:,0,:Nclc] = Cdrb0[:,:Nclc]
        Cdrb[:,:Nclc,0] = Cdrb0[:,:Nclc]
        
        Cdra[:,0,:Nclc] = Cdra0[:,:Nclc]
        Cdra[:,:Nclc,0] = Cdra0[:,:Nclc]
        
        Cdrp[:,0,:Nclc] = Cdrp0[:,:Nclc]
        Cdrp[:,:Nclc,0] = Cdrp0[:,:Nclc]
        
    Rphib = np.empty((Ntyp,),dtype=np.float32)
    Rphia = np.empty((Ntyp,),dtype=np.float32)
    Rphip = np.empty((Ntyp,),dtype=np.float32)
    d2_mat = d2_stencil(Tsav,dt)
    Cphib = Crb - (doub_vec(tau)**2 - dt*doub_vec(tau))[:,None] * np.einsum('ij,kj->ki',d2_mat,Crb)
    Cphia = Cra - (doub_vec(tau)**2 - dt*doub_vec(tau))[:,None] * np.einsum('ij,kj->ki',d2_mat,Cra)
    Cphip = Crp - (doub_vec(tau)**2 - dt*doub_vec(tau))[:,None] * np.einsum('ij,kj->ki',d2_mat,Crp)
    
    for i in range(Nint-1):
        if i > Nclc-1:
            Cdrb[:,i+1,i-Nclc] = Cdrb[:,i,i-Nclc]
            Cdra[:,i+1,i-Nclc] = Cdra[:,i,i-Nclc]
            Cdrp[:,i+1,i-Nclc] = Cdrp[:,i,i-Nclc]
            
        for j in range(max(0,i-Nclc),i+1):
            ij_idx = np.fmin(i-j,Nsav-1)
            
            Cdrbij = Cdrb[:,i,j]
            Cdraij = Cdra[:,i,j]
            Cdrpij = Cdrp[:,i,j]
            sCdrij = solve_width((Cdraij-Cdrbij)/(Cdrpij-Cdrbij))
            sWCdrij = np.sqrt(sW2+sCdrij**2)
            Cdrpmbij = Cdrpij - Cdrbij
            Sigdbij = (SigW+SigWb)@Cdrbij + (unstruct_fact(sCdrij,L)*SigWb)@Cdrpmbij
            Sigdaij = Sigdbij + (struct_fact(sa,sWCdrij,sCdrij,L)*SigW)@Cdrpmbij
            Sigdpij = Sigdbij + (struct_fact(0,sWCdrij,sCdrij,L)*SigW)@Cdrpmbij
            Sigdbij = Sigdbij + (struct_fact(L/2,sWCdrij,sCdrij,L)*SigW)@Cdrpmbij
            
            kbij = 0.5*(Sigb[:Ntyp,ij_idx]+Sigb[Ntyp:,ij_idx]-Sigdbij)
            kaij = 0.5*(Siga[:Ntyp,ij_idx]+Siga[Ntyp:,ij_idx]-Sigdaij)
            kpij = 0.5*(Sigp[:Ntyp,ij_idx]+Sigp[Ntyp:,ij_idx]-Sigdpij)
            
            R_fn(mub[:Ntyp],mub[Ntyp:],Sigb[:Ntyp,0],Sigb[Ntyp:,0],kbij,Rphib)
            R_fn(mua[:Ntyp],mua[Ntyp:],Siga[:Ntyp,0],Siga[Ntyp:,0],kaij,Rphia)
            R_fn(mup[:Ntyp],mup[Ntyp:],Sigp[:Ntyp,0],Sigp[Ntyp:,0],kpij,Rphip)
            
            Cdrb[:,i+1,j+1] = Cdrb[:,i,j+1]+Cdrb[:,i+1,j]-Cdrb[:,i,j] +\
                dttauinv*(-Cdrb[:,i+1,j]-Cdrb[:,i,j+1]+2*Cdrb[:,i,j]) +\
                dttauinv2*(-Cdrb[:,i,j]+Cphib[:Ntyp,ij_idx]+Cphib[Ntyp:,ij_idx]-2*Rphib)
            Cdra[:,i+1,j+1] = Cdra[:,i,j+1]+Cdra[:,i+1,j]-Cdra[:,i,j] +\
                dttauinv*(-Cdra[:,i+1,j]-Cdra[:,i,j+1]+2*Cdra[:,i,j]) +\
                dttauinv2*(-Cdra[:,i,j]+Cphia[:Ntyp,ij_idx]+Cphia[Ntyp:,ij_idx]-2*Rphia)
            Cdrp[:,i+1,j+1] = Cdrp[:,i,j+1]+Cdrp[:,i+1,j]-Cdrp[:,i,j] +\
                dttauinv*(-Cdrp[:,i+1,j]-Cdrp[:,i,j+1]+2*Cdrp[:,i,j]) +\
                dttauinv2*(-Cdrp[:,i,j]+Cphip[:Ntyp,ij_idx]+Cphip[Ntyp:,ij_idx]-2*Rphip)
                
            Cdrb[:,i+1,j+1] = np.maximum(Cdrb[:,i+1,j+1],drb**2)
            Cdra[:,i+1,j+1] = np.maximum(Cdra[:,i+1,j+1],dra**2)
            Cdrp[:,i+1,j+1] = np.maximum(Cdrp[:,i+1,j+1],drp**2)
            
            if np.any(np.abs(Cdrb[:,i+1,j+1]) > 1e10) or np.any(np.isnan(Cdrb[:,i+1,j+1])):
                print(Sigdbij,Sigdaij,Sigdpij,sCdrij)
                print("system diverged when integrating Cdrb")
                return Cdrb,Cdra,Cdrp,False,False,False
            if np.any(np.abs(Cdra[:,i+1,j+1]) > 1e10) or np.any(np.isnan(Cdra[:,i+1,j+1])):
                print(Sigdbij,Sigdaij,Sigdpij,sCdrij)
                print("system diverged when integrating Cdra")
                return Cdrb,Cdra,Cdrp,False,False,False
            if np.any(np.abs(Cdrp[:,i+1,j+1]) > 1e10) or np.any(np.isnan(Cdrp[:,i+1,j+1])):
                print(Sigdbij,Sigdaij,Sigdpij,sCdrij)
                print("system diverged when integrating Cdrp")
                return Cdrb,Cdra,Cdrp,False,False,False
                
            Cdrb[:,j+1,i+1] = Cdrb[:,i+1,j+1]
            Cdra[:,j+1,i+1] = Cdra[:,i+1,j+1]
            Cdrp[:,j+1,i+1] = Cdrp[:,i+1,j+1]
            
        Ndiv = 20
        if (Ndiv*(i+1)) % (Nint-1) == 0:
            print("{:.2f}% completed".format((i+1)/(Nint-1)))
            
    Cdrb_diag = each_diag(Cdrb)
    Cdra_diag = each_diag(Cdra)
    Cdrp_diag = each_diag(Cdrp)
    
    return Cdrb,Cdra,Cdrp,\
        (np.max(Cdrb_diag[:,-Nsav:],axis=1)-np.min(Cdrb_diag[:,-Nsav:],axis=1))/\
            np.mean(Cdrb_diag[:,-Nsav:],axis=1) < 1e-3,\
        (np.max(Cdra_diag[:,-Nsav:],axis=1)-np.min(Cdra_diag[:,-Nsav:],axis=1))/\
            np.mean(Cdra_diag[:,-Nsav:],axis=1) < 1e-3,\
        (np.max(Cdrp_diag[:,-Nsav:],axis=1)-np.min(Cdrp_diag[:,-Nsav:],axis=1))/\
            np.mean(Cdrp_diag[:,-Nsav:],axis=1) < 1e-3

def sparse_full_ring_dmft(tau,W,K,Hb,Hp,eH,sW,sH,M_fn,C_fn,Twrm,Tsav,dt,
                          rs0=None,Crs0=None,Kb=None,L=180,Nori=20):
    if Kb is None:
        Kb = np.zeros_like(K)
        
    Ntyp = len(Hb)
    Nint = round((Twrm+Tsav)/dt)+1
    Nclc = round(1.5*Tsav/dt)+1
    Nsav = round(Tsav/dt)+1
    
    rs = np.zeros((Nori,Ntyp,Nint),dtype=np.float32)
    Crs = np.zeros((Nori,Ntyp,Nint,Nint),dtype=np.float32)
    
    oris = np.arange(Nori)/Nori * L
    
    if rs0 is None:
        rs0 = (1+4*basesubwrapnorm(oris,15))[:,None]*np.ones((Ntyp),dtype=np.float32)[None,:]
    if Crs0 is None:
        Crs0 = (1e2+24e2*basesubwrapnorm(oris,15))[:,None,None]*np.ones((Ntyp,1),dtype=np.float32)[None,:,:]
        
    tauinv = 1/tau
    dttauinv = dt/tau
    dttauinv2 = dttauinv**2
    
    muWb = tau[:,None]*W*Kb
    SigWb = tau[:,None]**2*W**2*Kb
    muW = tau[:,None]*W*K
    SigW = tau[:,None]**2*W**2*K
    
    doris = oris[:,None] - oris[None,:]
    kerns = wrapnormdens(doris[:,None,:,None],sW[None,:,None,:],L)
    
    muWs = (muWb[None,:,None,:] + muW[None,:,None,:]*2*np.pi*kerns) / Nori
    SigWs = (SigWb[None,:,None,:] + SigW[None,:,None,:]*2*np.pi*kerns) / Nori
    
    muHs = tau*(Hb[None,:]+(Hp-Hb)[None,:]*basesubwrapnorm(oris[:,None],sH[None,:],L))
    SigHs = (muHs*eH)**2
    
    rs[:,:,0] = rs0
    
    NCr0 = Crs0.shape[2]
    if Nclc > NCr0:
        Crs[:,:,0,:NCr0] = Crs0
        Crs[:,:,0,NCr0:Nclc] = Crs0[:,:,-1:]
        Crs[:,:,:NCr0,0] = Crs0
        Crs[:,:,NCr0:Nclc,0] = Crs0[:,:,-1:]
    else:
        Crs[:,:,0,:Nclc] = Crs0[:,:,:Nclc]
        Crs[:,:,:Nclc,0] = Crs0[:,:,:Nclc]
        
    Mphis = np.empty((Nori,Ntyp),dtype=np.float32)
    Cphis = np.empty((Nori,Ntyp),dtype=np.float32)
    
    def drdt(rsi,Sigsii):
        musi = np.einsum('ijkl,kl->ij',muWs,rsi) + muHs
        for ori_idx in range(Nori):
            M_fn(musi[ori_idx],Sigsii[ori_idx],Mphis[ori_idx])
        return tauinv*(-rsi + Mphis)
    
    for i in range(Nint-1):
        Crsii = Crs[:,:,i,i]
        Sigsii = np.einsum('ijkl,kl->ij',SigWs,Crsii) + SigHs
            
        kb1 = drdt(rs[:,:,i]           ,Sigsii)
        kb2 = drdt(rs[:,:,i]+0.5*dt*kb1,Sigsii)
        kb3 = drdt(rs[:,:,i]+0.5*dt*kb2,Sigsii)
        kb4 = drdt(rs[:,:,i]+    dt*kb2,Sigsii)
        
        rs[:,:,i+1] = rs[:,:,i] + dt/6*(kb1+2*kb2+2*kb3+kb4)
        rsi = rs[:,:,i]
        musi = np.einsum('ijkl,kl->ij',muWs,rsi) + muHs
        
        if np.any(np.abs(rs[:,:,i+1]) > 1e10) or np.any(np.isnan(rs[:,:,i+1])):
            print(musi)
            print(Sigsii)
            print("system diverged when integrating rb")
            return rs,Crs,False

        if i > Nclc-1:
            Crs[:,:,i+1,i-Nclc] = Crs[:,:,i,i-Nclc]
            
        for j in range(max(0,i-Nclc),i+1):
            Crsij = Crs[:,:,i,j]
            Sigsij = np.einsum('ijkl,kl->ij',SigWs,Crsij) + SigHs
            for ori_idx in range(Nori):
                C_fn(musi[ori_idx],Sigsii[ori_idx],Sigsij[ori_idx],Cphis[ori_idx])
            Crs[:,:,i+1,j+1] = Crs[:,:,i,j+1]+Crs[:,:,i+1,j]-Crs[:,:,i,j] +\
                dttauinv*(-Crs[:,:,i+1,j]-Crs[:,:,i,j+1]+2*Crs[:,:,i,j]) + dttauinv2*(-Crs[:,:,i,j]+Cphis)
                
            Crs[:,:,i+1,j+1] = np.maximum(Crs[:,:,i+1,j+1],rsi**2)
            
            if np.any(np.abs(Crs[:,:,i+1,j+1]) > 1e10) or np.any(np.isnan(Crs[:,:,i+1,j+1])):
                print(musi)
                print(Sigsii)
                print(Sigsij)
                print("system diverged when integrating Crb")
                return rs,Crs,False
                
            Crs[:,:,j+1,i+1] = Crs[:,:,i+1,j+1]
            
        Ndiv = 5
        if (Ndiv*(i+1)) % (Nint-1) == 0:
            print("{:.2f}% completed".format((i+1)/(Nint-1)))
            
    Crs_diag = each_diag(Crs)
    
    return rs,Crs,\
        (np.max(Crs_diag[:,:,-Nsav:],axis=-1)-np.min(Crs_diag[:,:,-Nsav:],axis=-1))/\
            np.mean(Crs_diag[:,:,-Nsav:],axis=-1) < 1e-3

def doub_sparse_full_ring_dmft(tau,W,K,Hb,Hp,eH,sW,sH,M_fns,C_fns,Twrm,Tsav,dt,
                               rs0=None,Crs0=None,Kb=None,L=180,Nori=20):
    if Kb is None:
        doub_Kb = None
    else:
        doub_Kb = doub_vec(Kb)
        
    Ntyp = len(Hb)
    
    doub_tau = doub_vec(tau)
    doub_W = doub_mat(W)
    doub_K = doub_vec(K)
    doub_Hb = doub_vec(Hb)
    doub_Hp = doub_vec(Hp)
    doub_sW = np.block([[sW,sW],[sW,sW]])
    doub_sH = doub_vec(sH)
    
    def doub_M(mui,Sigii,out):
        M_fns[0](mui[:Ntyp],Sigii[:Ntyp],out[:Ntyp])
        M_fns[1](mui[Ntyp:],Sigii[Ntyp:],out[Ntyp:])
        
    def doub_C(mui,Sigii,Sigij,out):
        C_fns[0](mui[:Ntyp],Sigii[:Ntyp],Sigij[:Ntyp],out[:Ntyp])
        C_fns[1](mui[Ntyp:],Sigii[Ntyp:],Sigij[Ntyp:],out[Ntyp:])
        
    return sparse_full_ring_dmft(doub_tau,doub_W,doub_K,doub_Hb,doub_Hp,eH,doub_sW,doub_sH,doub_M,doub_C,Twrm,Tsav,dt,
                                 rs0,Crs0,Kb=doub_Kb,L=L,Nori=Nori)

def diff_sparse_full_ring_dmft(tau,W,K,Hb,Hp,eH,sW,sH,R_fn,Twrm,Tsav,dt,rs,Crs,
                               Cdrs0=None,Kb=None,L=180,Nori=20):
    if Kb is None:
        Kb = np.zeros_like(K)
        
    Ntyp = len(Hb)
    Nint = round((Twrm+Tsav)/dt)+1
    Nclc = round(1.5*Tsav/dt)+1
    Nsav = round(Tsav/dt)+1
    
    drs = rs[:,Ntyp:] - rs[:,:Ntyp]
    
    Cdrs = np.zeros((Nori,Ntyp,Nint,Nint),dtype=np.float32)
    
    oris = np.arange(Nori)/Nori * L
    
    if Cdrs0 is None:
        # Cdrs0 = Crs[:,:Ntyp]
        Cdrs0 = (rs[:,Ntyp:] - rs[:,:Ntyp]).astype(np.float32)[:,:,None]**2 +\
            (1e3+4e3*basesubwrapnorm(oris,15))[:,None,None]
        
    tauinv = 1/tau
    dttauinv = dt/tau
    dttauinv2 = dttauinv**2
    
    muWb = tau[:,None]*W*Kb
    SigWb = tau[:,None]**2*W**2*Kb
    muW = tau[:,None]*W*K
    SigW = tau[:,None]**2*W**2*K
    
    doris = oris[:,None] - oris[None,:]
    kerns = wrapnormdens(doris[:,None,:,None],sW[None,:,None,:],L)
    
    muWs = (muWb[None,:,None,:] + muW[None,:,None,:]*2*np.pi*kerns) / Nori
    SigWs = (SigWb[None,:,None,:] + SigW[None,:,None,:]*2*np.pi*kerns) / Nori
    
    muHs = tau*(Hb[None,:]+(Hp-Hb)[None,:]*basesubwrapnorm(oris[:,None],sH[None,:],L))
    SigHs = (muHs*eH)**2
    
    doub_tau = doub_vec(tau)
    doub_muW = doub_mat(muW)
    doub_SigW = doub_mat(SigW)
    doub_muWb = doub_mat(muWb)
    doub_SigWb = doub_mat(SigWb)
    doub_sW = np.block([[sW,sW],[sW,sW]])
    doub_Hb = doub_vec(Hb)
    doub_Hp = doub_vec(Hp)
    doub_sH = doub_vec(sH)
    
    doub_kerns = wrapnormdens(doris[:,None,:,None],doub_sW[None,:,None,:],L)

    doub_muWs = (doub_muWb[None,:,None,:] + doub_muW[None,:,None,:]*2*np.pi*doub_kerns) / Nori
    doub_SigWs = (doub_SigWb[None,:,None,:] + doub_SigW[None,:,None,:]*2*np.pi*doub_kerns) / Nori

    doub_muHs = doub_tau*(doub_Hb[None,:]+(doub_Hp-doub_Hb)[None,:]*basesubwrapnorm(oris[:,None],doub_sH[None,:],L))
    doub_SigHs = (doub_muHs*eH)**2
    
    mus = np.einsum('ijkl,kl->ij',doub_muWs,rs) + doub_muHs
    Sigs = np.einsum('ijkl,klm->ijm',doub_SigWs,Crs) + doub_SigHs[:,:,None]
    
    NCdr0 = Cdrs0.shape[1]
    if Nclc > NCdr0:
        Cdrs[:,:,0,:NCdr0] = Cdrs0
        Cdrs[:,:,0,NCdr0:Nclc] = Cdrs0[:,:,-1:]
        Cdrs[:,:,:NCdr0,0] = Cdrs0
        Cdrs[:,:,NCdr0:Nclc,0] = Cdrs0[:,:,-1:]
    else:
        Cdrs[:,:,0,:Nclc] = Cdrs0[:,:,:Nclc]
        Cdrs[:,:,:Nclc,0] = Cdrs0[:,:,:Nclc]
        
    Rphis = np.empty((Nori,Ntyp),dtype=np.float32)
    Cphis = Crs - (doub_vec(tau)**2 - dt*doub_vec(tau))[None,:,None] * np.einsum('ij,klj->kli',d2_stencil(Tsav,dt),Crs)
    
    for i in range(Nint-1):
        if i > Nclc-1:
            Cdrs[:,:,i+1,i-Nclc] = Cdrs[:,:,i,i-Nclc]
            
        for j in range(max(0,i-Nclc),i+1):
            ij_idx = np.fmin(i-j,Nsav-1)
            
            Cdrsij = Cdrs[:,:,i,j]
            Sigdsij = np.einsum('ijkl,kl->ij',SigWs,Cdrsij) + SigHs
            
            ksij = 0.5*(Sigs[:,:Ntyp,ij_idx]+Sigs[:,Ntyp:,ij_idx]-Sigdsij)
            
            for ori_idx in range(Nori):
                R_fn(mus[ori_idx,:Ntyp],mus[ori_idx,Ntyp:],Sigs[ori_idx,:Ntyp,0],Sigs[ori_idx,Ntyp:,0],
                     ksij[ori_idx],Rphis[ori_idx])
            
            Cdrs[:,:,i+1,j+1] = Cdrs[:,:,i,j+1]+Cdrs[:,:,i+1,j]-Cdrs[:,:,i,j] +\
                dttauinv*(-Cdrs[:,:,i+1,j]-Cdrs[:,:,i,j+1]+2*Cdrs[:,:,i,j]) +\
                dttauinv2*(-Cdrs[:,:,i,j]+Cphis[:,:Ntyp,ij_idx]+Cphis[:,Ntyp:,ij_idx]-2*Rphis)
                
            Cdrs[:,:,i+1,j+1] = np.maximum(Cdrs[:,:,i+1,j+1],drs**2)
            
            if np.any(np.abs(Cdrs[:,:,i+1,j+1]) > 1e10) or np.any(np.isnan(Cdrs[:,:,i+1,j+1])):
                print(Sigdsij)
                print("system diverged when integrating Cdrb")
                return Cdrs,False
                
            Cdrs[:,:,j+1,i+1] = Cdrs[:,:,i+1,j+1]
            
        Ndiv = 20
        if (Ndiv*(i+1)) % (Nint-1) == 0:
            print("{:.2f}% completed".format((i+1)/(Nint-1)))
            
    Cdrs_diag = each_diag(Cdrs)
    
    return Cdrs,\
        (np.max(Cdrs_diag[:,:,-Nsav:],axis=-1)-np.min(Cdrs_diag[:,:,-Nsav:],axis=-1))/\
            np.mean(Cdrs_diag[:,:,-Nsav:],axis=-1) < 1e-3

def run_first_stage_dmft(prms,rX,CVh,res_dir,rc,Twrm,Tsav,dt,which='both',return_full=False):
    Nsav = round(Tsav/dt)+1
    
    K = prms['K']
    J = prms['J']
    beta = prms['beta']
    gE = prms['gE']
    gI = prms['gI']
    hE = prms['hE']
    hI = prms['hI']
    
    tau = np.array([rc.tE,rc.tI],dtype=np.float32)
    W = J*np.array([[1,-gE],[1./beta,-gI/beta]],dtype=np.float32)
    Ks = np.array([K,K/4],dtype=np.float32)
    H = rX*K*J*np.array([hE,hI/beta],dtype=np.float32)
    eH = CVh
    
    muH = tau*H
    SigH = (muH*eH)**2
    
    FE,FI,ME,MI,CE,CI = base_itp_moments(res_dir)
    FL,ML,CL = opto_itp_moments(res_dir,prms['L'],prms['CVL'])
    
    def base_M(mui,Sigii,out):
        out[0] = ME(mui[0],Sigii[0])[0]
        out[1] = MI(mui[1],Sigii[1])[0]
        
    def base_C(mui,Sigii,Sigij,out):
        out[0] = CE(mui[0],Sigii[0],Sigij[0])[0]
        out[1] = CI(mui[1],Sigii[1],Sigij[1])[0]
    
    def opto_M(mui,Sigii,out):
        out[0] = ML(mui[0],Sigii[0])[0]
        out[1] = MI(mui[1],Sigii[1])[0]
        
    def opto_C(mui,Sigii,Sigij,out):
        out[0] = CL(mui[0],Sigii[0],Sigij[0])[0]
        out[1] = CI(mui[1],Sigii[1],Sigij[1])[0]
    
    start = time.process_time()
    
    if which=='base':
        full_r,full_Cr,conv = sparse_dmft(tau,W,Ks,H,eH,base_M,base_C,Twrm,Tsav,dt)
    elif which=='opto':
        full_r,full_Cr,conv = sparse_dmft(tau,W,Ks,H,eH,opto_M,opto_C,Twrm,Tsav,dt)
    elif which=='both':
        full_r,full_Cr,conv = doub_sparse_dmft(tau,W,Ks,H,eH,[base_M,opto_M],[base_C,opto_C],Twrm,Tsav,dt)
    else:
        raise NotImplementedError('Only implemented options for \'which\' keyword are: \'base\', \'opto\', and \'both\'')
        
    print('integrating first stage took',time.process_time() - start,'s')

    # extract predicted moments after long time evolution
    r = full_r[:,-1]
    Cr = full_Cr[:,-1,-1:-Nsav-1:-1]
    
    muW = tau[:,None]*W*Ks
    SigW = tau[:,None]**2*W**2*Ks
    
    if which in ('base','opto'):
        mu = muW@r + muH
        Sig = SigW@Cr + SigH[:,None]
    elif which=='both':
        doub_muW = doub_mat(muW)
        doub_SigW = doub_mat(SigW)

        mu = doub_muW@r + doub_vec(muH)
        Sig = doub_SigW@Cr + doub_vec(SigH)[:,None]
    else:
        raise NotImplementedError('Only implemented options for \'which\' keyword are: \'base\', \'opto\', and \'both\'')
    
    res_dict = {}
    
    res_dict['r'] = r
    res_dict['Cr'] = Cr
    
    res_dict['mu'] = mu
    res_dict['Sig'] = Sig
    
    res_dict['conv'] = conv
    
    if return_full:
        res_dict['full_r'] = full_r
        res_dict['full_Cr'] = full_Cr
    
    return res_dict

def run_second_stage_dmft(first_res_dict,prms,rX,CVh,res_dir,rc,Twrm,Tsav,dt,return_full=False):
    Nsav = round(Tsav/dt)+1
    
    K = prms['K']
    J = prms['J']
    beta = prms['beta']
    gE = prms['gE']
    gI = prms['gI']
    hE = prms['hE']
    hI = prms['hI']
    
    tau = np.array([rc.tE,rc.tI],dtype=np.float32)
    W = J*np.array([[1,-gE],[1./beta,-gI/beta]],dtype=np.float32)
    Ks = np.array([K,K/4],dtype=np.float32)
    H = rX*K*J*np.array([hE,hI/beta],dtype=np.float32)
    eH = CVh
    
    muH = tau*H
    SigH = (muH*eH)**2
    
    FE,FI,ME,MI,CE,CI = base_itp_moments(res_dir)
    FL,ML,CL = opto_itp_moments(res_dir,prms['L'],prms['CVL'])

    def diff_R(mu1i,mu2i,Sig1ii,Sig2ii,kij,out):
        out[0] = R_simp(ME,ML,mu1i[0],mu2i[0],Sig1ii[0],Sig2ii[0],kij[0])
        out[1] = R_simp(MI,MI,mu1i[1],mu2i[1],Sig1ii[1],Sig2ii[1],kij[1])

    r = first_res_dict['r']
    Cr = first_res_dict['Cr']
    
    muW = tau[:,None]*W*Ks
    SigW = tau[:,None]**2*W**2*Ks
    
    doub_muW = doub_mat(muW)
    doub_SigW = doub_mat(SigW)
    
    mu = doub_muW@r + doub_vec(muH)
    Sig = doub_SigW@Cr + doub_vec(SigH)[:,None]

    start = time.process_time()

    full_Cdr,convd = diff_sparse_dmft(tau,W,Ks,H,eH,diff_R,Twrm,Tsav,dt,r,Cr)

    print('integrating second stage took',time.process_time() - start,'s')

    # extract predicted moments after long time evolution
    dr = r[:2] - r[2:]
    Cdr = full_Cdr[:,-1,-1:-Nsav-1:-1]

    dmu = mu[:2] - mu[2:]

    Sigd = SigW@Cdr
    
    res_dict = {}
    
    res_dict['dr'] = dr
    res_dict['Cdr'] = Cdr
    
    res_dict['dmu'] = dmu
    res_dict['Sigd'] = Sigd

    res_dict['convd'] = convd
    
    if return_full:
        res_dict['full_Cdr'] = full_Cdr
    
    return res_dict

def run_two_stage_dmft(prms,rX,CVh,res_dir,rc,Twrm,Tsav,dt,return_full=False):
    Nsav = round(Tsav/dt)+1
    
    K = prms['K']
    J = prms['J']
    beta = prms['beta']
    gE = prms['gE']
    gI = prms['gI']
    hE = prms['hE']
    hI = prms['hI']
    
    tau = np.array([rc.tE,rc.tI],dtype=np.float32)
    W = J*np.array([[1,-gE],[1./beta,-gI/beta]],dtype=np.float32)
    Ks = np.array([K,K/4],dtype=np.float32)
    H = rX*K*J*np.array([hE,hI/beta],dtype=np.float32)
    eH = CVh
    
    muH = tau*H
    SigH = (muH*eH)**2
    
    FE,FI,ME,MI,CE,CI = base_itp_moments(res_dir)
    FL,ML,CL = opto_itp_moments(res_dir,prms['L'],prms['CVL'])
    
    def base_M(mui,Sigii,out):
        out[0] = ME(mui[0],Sigii[0])[0]
        out[1] = MI(mui[1],Sigii[1])[0]
    
    def base_C(mui,Sigii,Sigij,out):
        out[0] = CE(mui[0],Sigii[0],Sigij[0])[0]
        out[1] = CI(mui[1],Sigii[1],Sigij[1])[0]
    
    def opto_M(mui,Sigii,out):
        out[0] = ML(mui[0],Sigii[0])[0]
        out[1] = MI(mui[1],Sigii[1])[0]
    
    def opto_C(mui,Sigii,Sigij,out):
        out[0] = CL(mui[0],Sigii[0],Sigij[0])[0]
        out[1] = CI(mui[1],Sigii[1],Sigij[1])[0]

    def diff_R(mu1i,mu2i,Sig1ii,Sig2ii,kij,out):
        out[0] = R_simp(ME,ML,mu1i[0],mu2i[0],Sig1ii[0],Sig2ii[0],kij[0])
        out[1] = R_simp(MI,MI,mu1i[1],mu2i[1],Sig1ii[1],Sig2ii[1],kij[1])
    
    start = time.process_time()
    
    full_r,full_Cr,conv = doub_sparse_dmft(tau,W,Ks,H,eH,[base_M,opto_M],[base_C,opto_C],Twrm,Tsav,dt)

    print('integrating first stage took',time.process_time() - start,'s')

    # extract predicted moments after long time evolution
    r = full_r[:,-1]
    Cr = full_Cr[:,-1,-1:-Nsav-1:-1]
    
    muW = tau[:,None]*W*Ks
    SigW = tau[:,None]**2*W**2*Ks
    
    doub_muW = doub_mat(muW)
    doub_SigW = doub_mat(SigW)
    
    mu = doub_muW@r + doub_vec(muH)
    Sig = doub_SigW@Cr + doub_vec(SigH)[:,None]

    start = time.process_time()

    full_Cdr,convd = diff_sparse_dmft(tau,W,Ks,H,eH,diff_R,Twrm,Tsav,dt,r,Cr)

    print('integrating second stage took',time.process_time() - start,'s')

    # extract predicted moments after long time evolution
    dr = r[:2] - r[2:]
    Cdr = full_Cdr[:,-1,-1:-Nsav-1:-1]

    dmu = mu[:2] - mu[2:]

    Sigd = SigW@Cdr
    
    res_dict = {}
    
    res_dict['r'] = r
    res_dict['Cr'] = Cr
    res_dict['dr'] = dr
    res_dict['Cdr'] = Cdr
    
    res_dict['mu'] = mu
    res_dict['Sig'] = Sig
    res_dict['dmu'] = dmu
    res_dict['Sigd'] = Sigd

    res_dict['conv'] = conv
    res_dict['convd'] = convd
    
    if return_full:
        res_dict['full_r'] = full_r
        res_dict['full_Cr'] = full_Cr
        res_dict['full_Cdr'] = full_Cdr
    
    return res_dict

def run_first_stage_ring_dmft(prms,rX,cA,CVh,res_dir,rc,Twrm,Tsav,dt,sa=15,L=180,which='both',return_full=False):    
    Nsav = round(Tsav/dt)+1
    
    K = prms['K']
    SoriE = prms['SoriE']
    SoriI = prms['SoriI']
    SoriF = prms['SoriF']
    J = prms['J']
    beta = prms['beta']
    gE = prms['gE']
    gI = prms['gI']
    hE = prms['hE']
    hI = prms['hI']
    basefrac = prms.get('basefrac',0)
    baseinp = prms.get('baseinp',0)
    baseprob = prms.get('baseprob',0)
    
    tau = np.array([rc.tE,rc.tI],dtype=np.float32)
    W = J*np.array([[1,-gE],[1./beta,-gI/beta]],dtype=np.float32)
    Ks =  (1-basefrac)*(1-baseprob)   *np.array([K,K/4],dtype=np.float32)
    Kbs =((1-basefrac)*(1-baseprob)-1)*np.array([K,K/4],dtype=np.float32)
    Hb = rX*(1+((1-basefrac)*(1-baseinp)-1)*cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    Hp = rX*(1+                             cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    eH = CVh
    sW = np.array([[SoriE,SoriI],[SoriE,SoriI]],dtype=np.float32)
    sH = np.array([SoriF,SoriF],dtype=np.float32)
    
    sW2 = sW**2
    
    solve_width = get_solve_width(sa,L)
    
    muHb = tau*Hb
    SigHb = (muHb*eH)**2
    muHa = tau*(Hb+(Hp-Hb)*basesubwrapnorm(sa,sH,L))
    SigHa = (muHa*eH)**2
    muHp = tau*Hp
    SigHp = (muHp*eH)**2
    
    FE,FI,ME,MI,CE,CI = base_itp_moments(res_dir)
    FL,ML,CL = opto_itp_moments(res_dir,prms['L'],prms['CVL'])
    
    def base_M(mui,Sigii,out):
        out[0] = ME(mui[0],Sigii[0])[0]
        out[1] = MI(mui[1],Sigii[1])[0]
        
    def base_C(mui,Sigii,Sigij,out):
        out[0] = CE(mui[0],Sigii[0],Sigij[0])[0]
        out[1] = CI(mui[1],Sigii[1],Sigij[1])[0]
    
    def opto_M(mui,Sigii,out):
        out[0] = ML(mui[0],Sigii[0])[0]
        out[1] = MI(mui[1],Sigii[1])[0]
        
    def opto_C(mui,Sigii,Sigij,out):
        out[0] = CL(mui[0],Sigii[0],Sigij[0])[0]
        out[1] = CI(mui[1],Sigii[1],Sigij[1])[0]
    
    start = time.process_time()
    
    if which=='base':
        full_rb,full_ra,full_rp,full_Crb,full_Cra,full_Crp,\
            convb,conva,convp = sparse_ring_dmft(tau,W,Ks,Hb,Hp,eH,sW,sH,sa,base_M,base_C,Twrm,Tsav,dt,Kb=Kbs,L=L)
    elif which=='opto':
        full_rb,full_ra,full_rp,full_Crb,full_Cra,full_Crp,\
            convb,conva,convp = sparse_ring_dmft(tau,W,Ks,Hb,Hp,eH,sW,sH,sa,opto_M,opto_C,Twrm,Tsav,dt,Kb=Kbs,L=L)
    elif which=='both':
        full_rb,full_ra,full_rp,full_Crb,full_Cra,full_Crp,\
            convb,conva,convp = doub_sparse_ring_dmft(tau,W,Ks,Hb,Hp,eH,sW,sH,sa,[base_M,opto_M],[base_C,opto_C],
                                                      Twrm,Tsav,dt,Kb=Kbs,L=L)
    else:
        raise NotImplementedError('Only implemented options for \'which\' keyword are: \'base\', \'opto\', and \'both\'')
        
    print('integrating first stage took',time.process_time() - start,'s')

    # extract predicted moments after long time evolution
    rb = full_rb[:,-1]
    ra = full_ra[:,-1]
    rp = full_rp[:,-1]
    Crb = full_Crb[:,-1,-1:-Nsav-1:-1]
    Cra = full_Cra[:,-1,-1:-Nsav-1:-1]
    Crp = full_Crp[:,-1,-1:-Nsav-1:-1]
    
    muW = tau[:,None]*W*Ks
    SigW = tau[:,None]**2*W**2*Ks
    muWb = tau[:,None]*W*Kbs
    SigWb = tau[:,None]**2*W**2*Kbs
    
    sr = solve_width((ra-rb)/(rp-rb))
    sCr = solve_width((Cra-Crb)/(Crp-Crb))
    
    if which in ('base','opto'):
        sWr = np.sqrt(sW2+sr**2)
        rpmb = rp-rb
        sWCr = np.sqrt(sW2[:,:,None]+sCr[None,:,:]**2)
        Crpmb = Crp-Crb
        mub = (muW+muWb)@rb + (unstruct_fact(sr,L)*muWb)@rpmb + muHb
        mua = mub + (struct_fact(sa,sWr,sr,L)*muW)@rpmb + muHa-muHb
        mup = mub + (struct_fact(0,sWr,sr,L)*muW)@rpmb + muHp-muHb
        mub = mub + (struct_fact(L/2,sWr,sr,L)*muW)@rpmb
        Sigb = (SigW+SigWb)@Crb + each_matmul(unstruct_fact(sCr,L)*SigWb[:,:,None],Crpmb) + (SigHb)[:,None]
        Siga = Sigb + each_matmul(struct_fact(sa,sWCr,sCr,L)*SigW[:,:,None],Crpmb) + (SigHa-SigHb)[:,None]
        Sigp = Sigb + each_matmul(struct_fact(0,sWCr,sCr,L)*SigW[:,:,None],Crpmb) + (SigHp-SigHb)[:,None]
        Sigb = Sigb + each_matmul(struct_fact(L/2,sWCr,sCr,L)*SigW[:,:,None],Crpmb)
    elif which=='both':
        doub_muW = doub_mat(muW)
        doub_SigW = doub_mat(SigW)
        doub_muWb = doub_mat(muWb)
        doub_SigWb = doub_mat(SigWb)
        
        sWr = np.sqrt(doub_mat(sW2)+sr**2)
        rpmb = rp-rb
        sWCr = np.sqrt(doub_mat(sW2)[:,:,None]+sCr[None,:,:]**2)
        Crpmb = Crp-Crb
        mub = (doub_muW+doub_muWb)@rb + (unstruct_fact(sr,L)*doub_muWb)@rpmb + doub_vec(muHb)
        mua = mub + (struct_fact(sa,sWr,sr,L)*doub_muW)@rpmb + doub_vec(muHa-muHb)
        mup = mub + (struct_fact(0,sWr,sr,L)*doub_muW)@rpmb + doub_vec(muHp-muHb)
        mub = mub + (struct_fact(L/2,sWr,sr,L)*doub_muW)@rpmb
        Sigb = (doub_SigW+doub_SigWb)@Crb + each_matmul(unstruct_fact(sCr,L)*doub_SigWb[:,:,None],Crpmb) +\
            doub_vec(SigHb)[:,None]
        Siga = Sigb + each_matmul(struct_fact(sa,sWCr,sCr,L)*doub_SigW[:,:,None],Crpmb) + doub_vec(SigHa-SigHb)[:,None]
        Sigp = Sigb + each_matmul(struct_fact(0,sWCr,sCr,L)*doub_SigW[:,:,None],Crpmb) + doub_vec(SigHp-SigHb)[:,None]
        Sigb = Sigb + each_matmul(struct_fact(L/2,sWCr,sCr,L)*doub_SigW[:,:,None],Crpmb)
    else:
        raise NotImplementedError('Only implemented options for \'which\' keyword are: \'base\', \'opto\', and \'both\'')
    
    res_dict = {}
    
    res_dict['rb'] = rb
    res_dict['ra'] = ra
    res_dict['rp'] = rp
    res_dict['sr'] = sr
    res_dict['Crb'] = Crb
    res_dict['Cra'] = Cra
    res_dict['Crp'] = Crp
    res_dict['sCr'] = sCr
    
    res_dict['mub'] = mub
    res_dict['mua'] = mua
    res_dict['mup'] = mup
    res_dict['Sigb'] = Sigb
    res_dict['Siga'] = Siga
    res_dict['Sigp'] = Sigp
    
    res_dict['convb'] = convb
    res_dict['conva'] = conva
    res_dict['convp'] = convp
    
    if return_full:
        res_dict['full_rb'] = full_rb
        res_dict['full_ra'] = full_ra
        res_dict['full_rp'] = full_rp
        res_dict['full_Crb'] = full_Crb
        res_dict['full_Cra'] = full_Cra
        res_dict['full_Crp'] = full_Crp
    
    return res_dict

def run_decoupled_two_site_dmft(prms,rX,cA,CVh,res_dir,rc,Twrm,Tsav,dt,L=180,
                                struct_dict=None,which='both',return_full=False):
    Nsav = round(Tsav/dt)+1
    
    K = prms['K']
    SoriE = prms['SoriE']
    SoriI = prms['SoriI']
    SoriF = prms['SoriF']
    J = prms['J']
    beta = prms['beta']
    gE = prms['gE']
    gI = prms['gI']
    hE = prms['hE']
    hI = prms['hI']
    basefrac = prms.get('basefrac',0)
    baseinp = prms.get('baseinp',0)
    baseprob = prms.get('baseprob',0)
    
    tau = np.array([rc.tE,rc.tI],dtype=np.float32)
    W = J*np.array([[1,-gE],[1./beta,-gI/beta]],dtype=np.float32)
    Ks =  (1-basefrac)*(1-baseprob)   *np.array([K,K/4],dtype=np.float32)
    Kbs =((1-basefrac)*(1-baseprob)-1)*np.array([K,K/4],dtype=np.float32)
    Hb = rX*(1+((1-basefrac)*(1-baseinp)-1)*cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    Hp = rX*(1+                             cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    eH = CVh
    sW = np.array([[SoriE,SoriI],[SoriE,SoriI]],dtype=np.float32)
    
    sW2 = sW**2
    
    muW = tau[:,None]*W*Ks
    SigW = tau[:,None]**2*W**2*Ks
    muWb = tau[:,None]*W*Kbs
    SigWb = tau[:,None]**2*W**2*Kbs
    
    sr = struct_dict['sr']
    sCr = struct_dict['sCr'][:,-1]
    
    sWr = np.sqrt(sW2+sr**2)
    sWCr = np.sqrt(sW2+sCr**2)
    
    muWbb = (1 - struct_fact(180/2,sWr,sr,180)) * muW + (1 - unstruct_fact(sr,L)) * muWb
    muWbp = struct_fact(180/2,sWr,sr,180) * muW + unstruct_fact(sr,L) * muWb
    muWpb = (1 - struct_fact(0,sWr,sr,180)) * muW + (1 - unstruct_fact(sr,L)) * muWb
    muWpp = struct_fact(0,sWr,sr,180) * muW + unstruct_fact(sr,L) * muWb

    SigWbb = (1 - struct_fact(180/2,sWCr,sCr,180)) * SigW + (1 - unstruct_fact(sCr,L)) * SigWb
    SigWbp = struct_fact(180/2,sWCr,sCr,180) * SigW + unstruct_fact(sCr,L) * SigWb
    SigWpb = (1 - struct_fact(0,sWCr,sCr,180)) * SigW + (1 - unstruct_fact(sCr,L)) * SigWb
    SigWpp = struct_fact(0,sWCr,sCr,180) * SigW + unstruct_fact(sCr,L) * SigWb
    
    muHb = tau*Hb + muWbp@struct_dict.get('rp',0)
    SigHb = ((tau*Hb*eH)**2)[:,None] + SigWbp@struct_dict.get('Crp',0)
    muHp = tau*Hp + muWpb@struct_dict.get('rb',0)
    SigHp = ((tau*Hp*eH)**2)[:,None] + SigWpb@struct_dict.get('Crb',0)
    
    Norig = SigHb.shape[1]
    if Norig!=Nsav:
        temp = SigHb.copy()
        SigHb = np.zeros((2,Nsav))
        SigHb[:,:Norig] = temp
        SigHb[:,Norig:] = temp[:,-1]
        
        temp = SigHp.copy()
        SigHp = np.zeros((2,Nsav))
        SigHp[:,:Norig] = temp
        SigHp[:,Norig:] = temp[:,-1]
    
    FE,FI,ME,MI,CE,CI = base_itp_moments(res_dir)
    FL,ML,CL = opto_itp_moments(res_dir,prms['L'],prms['CVL'])
    
    def base_M(mui,Sigii,out):
        out[0] = ME(mui[0],Sigii[0])[0]
        out[1] = MI(mui[1],Sigii[1])[0]
        
    def base_C(mui,Sigii,Sigij,out):
        out[0] = CE(mui[0],Sigii[0],Sigij[0])[0]
        out[1] = CI(mui[1],Sigii[1],Sigij[1])[0]
    
    def opto_M(mui,Sigii,out):
        out[0] = ML(mui[0],Sigii[0])[0]
        out[1] = MI(mui[1],Sigii[1])[0]
        
    def opto_C(mui,Sigii,Sigij,out):
        out[0] = CL(mui[0],Sigii[0],Sigij[0])[0]
        out[1] = CI(mui[1],Sigii[1],Sigij[1])[0]
    
    start = time.process_time()
    
    if which=='base':
        full_rb,full_Crb,convb = gauss_dmft(tau,muWbb,SigWbb,muHb,SigHb,base_M,base_C,Twrm,Tsav,dt)
        full_rp,full_Crp,convp = gauss_dmft(tau,muWpp,SigWpp,muHp,SigHp,base_M,base_C,Twrm,Tsav,dt)
    elif which=='opto':
        full_rb,full_Crb,convb = gauss_dmft(tau,muWbb,SigWbb,muHb,SigHb,opto_M,opto_C,Twrm,Tsav,dt)
        full_rp,full_Crp,convp = gauss_dmft(tau,muWpp,SigWpp,muHp,SigHp,opto_M,opto_C,Twrm,Tsav,dt)
    else:
        raise NotImplementedError('Only implemented options for \'which\' keyword are: \'base\' and \'opto\'')
        
    print('integrating first stage took',time.process_time() - start,'s')

    # extract predicted moments after long time evolution
    rb = full_rb[:,-1]
    rp = full_rp[:,-1]
    Crb = full_Crb[:,-1,-1:-Nsav-1:-1]
    Crp = full_Crp[:,-1,-1:-Nsav-1:-1]
    
    if which in ('base','opto'):
        mub = muWbb@rb + muHb
        Sigb = SigWbb@Crb + SigHb
        mup = muWpp@rp + muHp
        Sigp = SigWpp@Crp + SigHp
    else:
        raise NotImplementedError('Only implemented options for \'which\' keyword are: \'base\', \'opto\', and \'both\'')
    
    res_dict = {}
    
    res_dict['rb'] = rb
    res_dict['rp'] = rp
    res_dict['sr'] = sr
    res_dict['Crb'] = Crb
    res_dict['Crp'] = Crp
    res_dict['sCr'] = sCr
    
    res_dict['mub'] = mub
    res_dict['mup'] = mup
    res_dict['Sigb'] = Sigb
    res_dict['Sigp'] = Sigp
    
    res_dict['convb'] = convb
    res_dict['convp'] = convp
    
    if return_full:
        res_dict['full_rb'] = full_rb
        res_dict['full_rp'] = full_rp
        res_dict['full_Crb'] = full_Crb
        res_dict['full_Crp'] = full_Crp
    
    return res_dict

def run_decoupled_ring_dmft(prms,rX,cA,CVh,res_dir,rc,Twrm,Tsav,dt,L=180,Nori=20,
                                struct_dict=None,which='both',return_full=False):
    Nsav = round(Tsav/dt)+1
    
    K = prms['K']
    SoriE = prms['SoriE']
    SoriI = prms['SoriI']
    SoriF = prms['SoriF']
    J = prms['J']
    beta = prms['beta']
    gE = prms['gE']
    gI = prms['gI']
    hE = prms['hE']
    hI = prms['hI']
    basefrac = prms.get('basefrac',0)
    baseinp = prms.get('baseinp',0)
    baseprob = prms.get('baseprob',0)
    
    tau = np.array([rc.tE,rc.tI],dtype=np.float32)
    W = J*np.array([[1,-gE],[1./beta,-gI/beta]],dtype=np.float32)
    Ks =  (1-basefrac)*(1-baseprob)   *np.array([K,K/4],dtype=np.float32)
    Kbs =((1-basefrac)*(1-baseprob)-1)*np.array([K,K/4],dtype=np.float32)
    Hb = rX*(1+((1-basefrac)*(1-baseinp)-1)*cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    Hp = rX*(1+                             cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    eH = CVh
    sW = np.array([[SoriE,SoriI],[SoriE,SoriI]],dtype=np.float32)
    sH = np.array([SoriF,SoriF],dtype=np.float32)
    
    sW2 = sW**2
    
    muW = tau[:,None]*W*Ks
    SigW = tau[:,None]**2*W**2*Ks
    muWb = tau[:,None]*W*Kbs
    SigWb = tau[:,None]**2*W**2*Kbs
    
    sr = struct_dict['sr']
    sCr = struct_dict['sCr'][:,-1]
    
    sWr = np.sqrt(sW2+sr**2)
    sWCr = np.sqrt(sW2+sCr**2)
    
    muWs = np.zeros((Nori,Nori,2,2))
    SigWs = np.zeros((Nori,Nori,2,2))
    
    oris = np.arange(Nori)/Nori * L
    doris = oris[:,None] - oris[None,:]
    kerns = wrapnormdens(doris[:,:,None,None],sW[None,None,:,:],L)
    
    muWs = (muWb[None,None,:,:] + muW[None,None,:,:]*2*np.pi*kerns) / Nori
    SigWs = (SigWb[None,None,:,:] + SigW[None,None,:,:]*2*np.pi*kerns) / Nori
    
    muWxs = muWs.copy()
    SigWxs = SigWs.copy()
    for i in range(Nori):
        muWxs[i,i] = 0
        SigWxs[i,i] = 0
    
    rs = struct_dict['rb'][None,:] + (struct_dict['rp']-struct_dict['rb'])[None,:]*\
        basesubwrapnorm(oris[:,None],sr[None,:])
    Crs = struct_dict['Crb'][None,:,:] + (struct_dict['Crp']-struct_dict['Crb'])[None,:,:]*\
        basesubwrapnorm(oris[:,None,None],sCr[None,:,None])
    
    muHs = tau[None,:]*(Hb[None,:] + (Hp-Hb)[None,:]*basesubwrapnorm(oris[:,None],sH[None,:])) +\
        np.einsum('ijkl,jl->ik',muWxs,rs)
    SigHs = ((tau[None,:]*(Hb[None,:] + (Hp-Hb)[None,:]*basesubwrapnorm(oris[:,None],sH[None,:]))*eH)**2)[:,:,None] +\
        np.einsum('ijkl,jlm->ikm',SigWxs,Crs)
    
    FE,FI,ME,MI,CE,CI = base_itp_moments(res_dir)
    FL,ML,CL = opto_itp_moments(res_dir,prms['L'],prms['CVL'])
    
    def base_M(mui,Sigii,out):
        out[0] = ME(mui[0],Sigii[0])[0]
        out[1] = MI(mui[1],Sigii[1])[0]
        
    def base_C(mui,Sigii,Sigij,out):
        out[0] = CE(mui[0],Sigii[0],Sigij[0])[0]
        out[1] = CI(mui[1],Sigii[1],Sigij[1])[0]
    
    def opto_M(mui,Sigii,out):
        out[0] = ML(mui[0],Sigii[0])[0]
        out[1] = MI(mui[1],Sigii[1])[0]
        
    def opto_C(mui,Sigii,Sigij,out):
        out[0] = CL(mui[0],Sigii[0],Sigij[0])[0]
        out[1] = CI(mui[1],Sigii[1],Sigij[1])[0]
    
    start = time.process_time()
    
    full_rs = [None]*20
    full_Crs = [None]*20
    convs = [None]*20
    
    if which=='base':
        for i in range(Nori):
            full_rs[i],full_Crs[i],convs[i] = gauss_dmft(tau,muWs[i,i],SigWs[i,i],muHs[i],SigHs[i],
                                                         base_M,base_C,Twrm,Tsav,dt)
    elif which=='opto':
        for i in range(Nori):
            full_rs[i],full_Crs[i],convs[i] = gauss_dmft(tau,muWs[i,i],SigWs[i,i],muHs[i],SigHs[i],
                                                         opto_M,opto_C,Twrm,Tsav,dt)
    else:
        raise NotImplementedError('Only implemented options for \'which\' keyword are: \'base\' and \'opto\'')
        
    print('integrating first stage took',time.process_time() - start,'s')

    # extract predicted moments after long time evolution
    for i in range(Nori):
        rs[i] = full_rs[i][:,-1]
        Crs[i] = full_Crs[i][:,-1,-1:-Nsav-1:-1]
    
    if which in ('base','opto'):
        mus = np.zeros_like(rs)
        Sigs = np.zeros_like(Crs)
        for i in range(Nori):
            mus[i] = muWs[i,i]@rs[i] + muHs[i]
            Sigs[i] = SigWs[i,i]@Crs[i] + SigHs[i]
    else:
        raise NotImplementedError('Only implemented options for \'which\' keyword are: \'base\', \'opto\', and \'both\'')
    
    res_dict = {}
    
    res_dict['rs'] = rs
    res_dict['sr'] = sr
    res_dict['Crs'] = Crs
    res_dict['sCr'] = sCr
    
    res_dict['mus'] = mus
    res_dict['Sigs'] = Sigs
    
    res_dict['convs'] = convs
    
    if return_full:
        res_dict['full_rs'] = full_rs
        res_dict['full_Crs'] = full_Crs
    
    return res_dict

def run_second_stage_ring_dmft(first_res_dict,prms,rX,cA,CVh,res_dir,rc,Twrm,Tsav,dt,sa=15,L=180,return_full=False):
    Nsav = round(Tsav/dt)+1
    
    K = prms['K']
    SoriE = prms['SoriE']
    SoriI = prms['SoriI']
    SoriF = prms['SoriF']
    J = prms['J']
    beta = prms['beta']
    gE = prms['gE']
    gI = prms['gI']
    hE = prms['hE']
    hI = prms['hI']
    basefrac = prms.get('basefrac',0)
    baseinp = prms.get('baseinp',0)
    baseprob = prms.get('baseprob',0)
    
    tau = np.array([rc.tE,rc.tI],dtype=np.float32)
    W = J*np.array([[1,-gE],[1./beta,-gI/beta]],dtype=np.float32)
    Ks =  (1-basefrac)*(1-baseprob)   *np.array([K,K/4],dtype=np.float32)
    Kbs =((1-basefrac)*(1-baseprob)-1)*np.array([K,K/4],dtype=np.float32)
    Hb = rX*(1+((1-basefrac)*(1-baseinp)-1)*cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    Hp = rX*(1+                             cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    eH = CVh
    sW = np.array([[SoriE,SoriI],[SoriE,SoriI]],dtype=np.float32)
    sH = np.array([SoriF,SoriF],dtype=np.float32)
    
    sW2 = sW**2
    
    solve_width = get_solve_width(sa,L)
    
    muHb = tau*Hb
    SigHb = (muHb*eH)**2
    muHa = tau*(Hb+(Hp-Hb)*basesubwrapnorm(sa,sH,L))
    SigHa = (muHa*eH)**2
    muHp = tau*Hp
    SigHp = (muHp*eH)**2
    
    FE,FI,ME,MI,CE,CI = base_itp_moments(res_dir)
    FL,ML,CL = opto_itp_moments(res_dir,prms['L'],prms['CVL'])

    def diff_R(mu1i,mu2i,Sig1ii,Sig2ii,kij,out):
        out[0] = R_simp(ME,ML,mu1i[0],mu2i[0],Sig1ii[0],Sig2ii[0],kij[0])
        out[1] = R_simp(MI,MI,mu1i[1],mu2i[1],Sig1ii[1],Sig2ii[1],kij[1])

    rb = first_res_dict['rb']
    ra = first_res_dict['ra']
    rp = first_res_dict['rp']
    Crb = first_res_dict['Crb']
    Cra = first_res_dict['Cra']
    Crp = first_res_dict['Crp']
    
    muW = tau[:,None]*W*Ks
    SigW = tau[:,None]**2*W**2*Ks
    muWb = tau[:,None]*W*Kbs
    SigWb = tau[:,None]**2*W**2*Kbs
    
    doub_muW = doub_mat(muW)
    doub_SigW = doub_mat(SigW)
    doub_muWb = doub_mat(muWb)
    doub_SigWb = doub_mat(SigWb)
    
    sr = solve_width((ra-rb)/(rp-rb))
    sWr = np.sqrt(doub_mat(sW2)+sr**2)
    rpmb = rp-rb
    sCr = solve_width((Cra-Crb)/(Crp-Crb))
    sWCr = np.sqrt(doub_mat(sW2)[:,:,None]+sCr[None,:,:]**2)
    Crpmb = Crp-Crb
    
    mub = (doub_muW+doub_muWb)@rb + (unstruct_fact(sr,L)*doub_muWb)@rpmb + doub_vec(muHb)
    mua = mub + (struct_fact(sa,sWr,sr,L)*doub_muW)@rpmb + doub_vec(muHa-muHb)
    mup = mub + (struct_fact(0,sWr,sr,L)*doub_muW)@rpmb + doub_vec(muHp-muHb)
    mub = mub + (struct_fact(L/2,sWr,sr,L)*doub_muW)@rpmb
    Sigb = (doub_SigW+doub_SigWb)@Crb + each_matmul(unstruct_fact(sCr,L)*doub_SigWb[:,:,None],Crpmb) +\
        doub_vec(SigHb)[:,None]
    Siga = Sigb + each_matmul(struct_fact(sa,sWCr,sCr,L)*doub_SigW[:,:,None],Crpmb) + doub_vec(SigHa-SigHb)[:,None]
    Sigp = Sigb + each_matmul(struct_fact(0,sWCr,sCr,L)*doub_SigW[:,:,None],Crpmb) + doub_vec(SigHp-SigHb)[:,None]
    Sigb = Sigb + each_matmul(struct_fact(L/2,sWCr,sCr,L)*doub_SigW[:,:,None],Crpmb)

    start = time.process_time()

    full_Cdrb,full_Cdra,full_Cdrp,\
        convdb,convda,convdp = diff_sparse_ring_dmft(tau,W,Ks,Hb,Hp,eH,sW,sH,sa,diff_R,Twrm,Tsav,dt,
                                                rb,ra,rp,Crb,Cra,Crp,Kb=Kbs,L=L)

    print('integrating second stage took',time.process_time() - start,'s')

    # extract predicted moments after long time evolution
    drb = rb[2:] - rb[:2]
    dra = ra[2:] - ra[:2]
    drp = rp[2:] - rp[:2]
    Cdrb = full_Cdrb[:,-1,-1:-Nsav-1:-1]
    Cdra = full_Cdra[:,-1,-1:-Nsav-1:-1]
    Cdrp = full_Cdrp[:,-1,-1:-Nsav-1:-1]

    dmub = mub[2:] - mub[:2]
    dmua = mua[2:] - mua[:2]
    dmup = mup[2:] - mup[:2]

    sCdr = solve_width((Cdra-Cdrb)/(Cdrp-Cdrb))
    sWCdr = np.sqrt(sW2[:,:,None]+sCdr[None,:,:]**2)
    Cdrpmb = Cdrp - Cdrb
    Sigdb = (SigW+SigWb)@Cdrb + each_matmul(unstruct_fact(sCdr,L)*SigWb[:,:,None],Cdrpmb)
    Sigda = Sigdb + each_matmul(struct_fact(sa,sWCdr,sCdr,L)*SigW[:,:,None],Cdrpmb)
    Sigdp = Sigdb + each_matmul(struct_fact(0,sWCdr,sCdr,L)*SigW[:,:,None],Cdrpmb)
    Sigdb = Sigdb + each_matmul(struct_fact(L/2,sWCdr,sCdr,L)*SigW[:,:,None],Cdrpmb)
    
    res_dict = {}
    
    res_dict['drb'] = drb
    res_dict['dra'] = dra
    res_dict['drp'] = drp
    res_dict['Cdrb'] = Cdrb
    res_dict['Cdra'] = Cdra
    res_dict['Cdrp'] = Cdrp
    res_dict['sCdr'] = sCdr
    
    res_dict['dmub'] = dmub
    res_dict['dmua'] = dmua
    res_dict['dmup'] = dmup
    res_dict['Sigdb'] = Sigdb
    res_dict['Sigda'] = Sigda
    res_dict['Sigdp'] = Sigdp

    res_dict['convdb'] = convdb
    res_dict['convda'] = convda
    res_dict['convdp'] = convdp
    
    if return_full:
        res_dict['full_Cdrb'] = full_Cdrb
        res_dict['full_Cdra'] = full_Cdra
        res_dict['full_Cdrp'] = full_Cdrp
    
    return res_dict

def run_two_stage_ring_dmft(prms,rX,cA,CVh,res_dir,rc,Twrm,Tsav,dt,sa=15,L=180,return_full=False):
    Nsav = round(Tsav/dt)+1
    
    K = prms['K']
    SoriE = prms['SoriE']
    SoriI = prms['SoriI']
    SoriF = prms['SoriF']
    J = prms['J']
    beta = prms['beta']
    gE = prms['gE']
    gI = prms['gI']
    hE = prms['hE']
    hI = prms['hI']
    basefrac = prms.get('basefrac',0)
    baseinp = prms.get('baseinp',0)
    baseprob = prms.get('baseprob',0)
    
    tau = np.array([rc.tE,rc.tI],dtype=np.float32)
    W = J*np.array([[1,-gE],[1./beta,-gI/beta]],dtype=np.float32)
    Ks =  (1-basefrac)*(1-baseprob)   *np.array([K,K/4],dtype=np.float32)
    Kbs =((1-basefrac)*(1-baseprob)-1)*np.array([K,K/4],dtype=np.float32)
    Hb = rX*(1+((1-basefrac)*(1-baseinp)-1)*cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    Hp = rX*(1+                             cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    eH = CVh
    sW = np.array([[SoriE,SoriI],[SoriE,SoriI]],dtype=np.float32)
    sH = np.array([SoriF,SoriF],dtype=np.float32)
    
    sW2 = sW**2
    
    solve_width = get_solve_width(sa,L)
    
    muHb = tau*Hb
    SigHb = (muHb*eH)**2
    muHa = tau*(Hb+(Hp-Hb)*basesubwrapnorm(sa,sH,L))
    SigHa = (muHa*eH)**2
    muHp = tau*Hp
    SigHp = (muHp*eH)**2
    
    FE,FI,ME,MI,CE,CI = base_itp_moments(res_dir)
    FL,ML,CL = opto_itp_moments(res_dir,prms['L'],prms['CVL'])
    
    def base_M(mui,Sigii,out):
        out[0] = ME(mui[0],Sigii[0])[0]
        out[1] = MI(mui[1],Sigii[1])[0]
    
    def base_C(mui,Sigii,Sigij,out):
        out[0] = CE(mui[0],Sigii[0],Sigij[0])[0]
        out[1] = CI(mui[1],Sigii[1],Sigij[1])[0]
    
    def opto_M(mui,Sigii,out):
        out[0] = ML(mui[0],Sigii[0])[0]
        out[1] = MI(mui[1],Sigii[1])[0]
    
    def opto_C(mui,Sigii,Sigij,out):
        out[0] = CL(mui[0],Sigii[0],Sigij[0])[0]
        out[1] = CI(mui[1],Sigii[1],Sigij[1])[0]

    def diff_R(mu1i,mu2i,Sig1ii,Sig2ii,kij,out):
        out[0] = R_simp(ME,ML,mu1i[0],mu2i[0],Sig1ii[0],Sig2ii[0],kij[0])
        out[1] = R_simp(MI,MI,mu1i[1],mu2i[1],Sig1ii[1],Sig2ii[1],kij[1])
    
    start = time.process_time()
    
    full_rb,full_ra,full_rp,full_Crb,full_Cra,full_Crp,\
        convb,conva,convp = doub_sparse_ring_dmft(tau,W,Ks,Hb,Hp,eH,sW,sH,sa,[base_M,opto_M],[base_C,opto_C],
                                                  Twrm,Tsav,dt,Kb=Kbs,L=L)

    print('integrating first stage took',time.process_time() - start,'s')

    # extract predicted moments after long time evolution
    rb = full_rb[:,-1]
    ra = full_ra[:,-1]
    rp = full_rp[:,-1]
    Crb = full_Crb[:,-1,-1:-Nsav-1:-1]
    Cra = full_Cra[:,-1,-1:-Nsav-1:-1]
    Crp = full_Crp[:,-1,-1:-Nsav-1:-1]
    
    muW = tau[:,None]*W*Ks
    SigW = tau[:,None]**2*W**2*Ks
    muWb = tau[:,None]*W*Kbs
    SigWb = tau[:,None]**2*W**2*Kbs
    
    doub_muW = doub_mat(muW)
    doub_SigW = doub_mat(SigW)
    doub_muWb = doub_mat(muWb)
    doub_SigWb = doub_mat(SigWb)
    
    sr = solve_width((ra-rb)/(rp-rb))
    sWr = np.sqrt(doub_mat(sW2)+sr**2)
    rpmb = rp-rb
    sCr = solve_width((Cra-Crb)/(Crp-Crb))
    sWCr = np.sqrt(doub_mat(sW2)[:,:,None]+sCr[None,:,:]**2)
    Crpmb = Crp-Crb
    
    mub = (doub_muW+doub_muWb)@rb + (unstruct_fact(sr,L)*doub_muWb)@rpmb + doub_vec(muHb)
    mua = mub + (struct_fact(sa,sWr,sr,L)*doub_muW)@rpmb + doub_vec(muHa-muHb)
    mup = mub + (struct_fact(0,sWr,sr,L)*doub_muW)@rpmb + doub_vec(muHp-muHb)
    mub = mub + (struct_fact(L/2,sWr,sr,L)*doub_muW)@rpmb
    Sigb = (doub_SigW+doub_SigWb)@Crb + each_matmul(unstruct_fact(sCr,L)*doub_SigWb[:,:,None],Crpmb) +\
        doub_vec(SigHb)[:,None]
    Siga = Sigb + each_matmul(struct_fact(sa,sWCr,sCr,L)*doub_SigW[:,:,None],Crpmb) + doub_vec(SigHa-SigHb)[:,None]
    Sigp = Sigb + each_matmul(struct_fact(0,sWCr,sCr,L)*doub_SigW[:,:,None],Crpmb) + doub_vec(SigHp-SigHb)[:,None]
    Sigb = Sigb + each_matmul(struct_fact(L/2,sWCr,sCr,L)*doub_SigW[:,:,None],Crpmb)

    start = time.process_time()

    full_Cdrb,full_Cdra,full_Cdrp,\
        convdb,convda,convdp = diff_sparse_ring_dmft(tau,W,Ks,Hb,Hp,eH,sW,sH,sa,diff_R,Twrm,Tsav,dt,
                                                rb,ra,rp,Crb,Cra,Crp,Kb=Kbs,L=L)

    print('integrating second stage took',time.process_time() - start,'s')

    # extract predicted moments after long time evolution
    drb = rb[2:] - rb[:2]
    dra = ra[2:] - ra[:2]
    drp = rp[2:] - rp[:2]
    Cdrb = full_Cdrb[:,-1,-1:-Nsav-1:-1]
    Cdra = full_Cdra[:,-1,-1:-Nsav-1:-1]
    Cdrp = full_Cdrp[:,-1,-1:-Nsav-1:-1]

    dmub = mub[2:] - mub[:2]
    dmua = mua[2:] - mua[:2]
    dmup = mup[2:] - mup[:2]

    sCdr = solve_width((Cdra-Cdrb)/(Cdrp-Cdrb))
    sWCdr = np.sqrt(sW2[:,:,None]+sCdr[None,:,:]**2)
    Cdrpmb = Cdrp - Cdrb
    Sigdb = (SigW+SigWb)@Cdrb + each_matmul(unstruct_fact(sCdr,L)*SigWb[:,:,None],Cdrpmb)
    Sigda = Sigdb + each_matmul(struct_fact(sa,sWCdr,sCdr,L)*SigW[:,:,None],Cdrpmb)
    Sigdp = Sigdb + each_matmul(struct_fact(0,sWCdr,sCdr,L)*SigW[:,:,None],Cdrpmb)
    Sigdb = Sigdb + each_matmul(struct_fact(L/2,sWCdr,sCdr,L)*SigW[:,:,None],Cdrpmb)
    
    res_dict = {}
    
    res_dict['rb'] = rb
    res_dict['ra'] = ra
    res_dict['rp'] = rp
    res_dict['sr'] = sr
    res_dict['Crb'] = Crb
    res_dict['Cra'] = Cra
    res_dict['Crp'] = Crp
    res_dict['sCr'] = sCr
    res_dict['drb'] = drb
    res_dict['dra'] = dra
    res_dict['drp'] = drp
    res_dict['Cdrb'] = Cdrb
    res_dict['Cdra'] = Cdra
    res_dict['Cdrp'] = Cdrp
    res_dict['sCdr'] = sCdr
    
    res_dict['mub'] = mub
    res_dict['mua'] = mua
    res_dict['mup'] = mup
    res_dict['Sigb'] = Sigb
    res_dict['Siga'] = Siga
    res_dict['Sigp'] = Sigp
    res_dict['dmub'] = dmub
    res_dict['dmua'] = dmua
    res_dict['dmup'] = dmup
    res_dict['Sigdb'] = Sigdb
    res_dict['Sigda'] = Sigda
    res_dict['Sigdp'] = Sigdp

    res_dict['convb'] = convb
    res_dict['conva'] = conva
    res_dict['convp'] = convp
    res_dict['convdb'] = convdb
    res_dict['convda'] = convda
    res_dict['convdp'] = convdp
    
    if return_full:
        res_dict['full_rb'] = full_rb
        res_dict['full_ra'] = full_ra
        res_dict['full_rp'] = full_rp
        res_dict['full_Crb'] = full_Crb
        res_dict['full_Cra'] = full_Cra
        res_dict['full_Crp'] = full_Crp
        res_dict['full_Cdrb'] = full_Cdrb
        res_dict['full_Cdra'] = full_Cdra
        res_dict['full_Cdrp'] = full_Cdrp
    
    return res_dict

def run_2feat_ring_dmft(prms,rX,cA,CVh,res_dir,rc,Twrm,Tsav,dt,sa=15,dori=45,L=180,which='both',return_full=False):    
    Nsav = round(Tsav/dt)+1
    
    K = prms['K']
    SoriE = prms['SoriE']
    SoriI = prms['SoriI']
    SoriF = prms['SoriF']
    J = prms['J']
    beta = prms['beta']
    gE = prms['gE']
    gI = prms['gI']
    hE = prms['hE']
    hI = prms['hI']
    basefrac = prms.get('basefrac',0)
    baseinp = prms.get('baseinp',0)
    baseprob = prms.get('baseprob',0)
    
    tau = np.array([rc.tE,rc.tI],dtype=np.float32)
    W = J*np.array([[1,-gE],[1./beta,-gI/beta]],dtype=np.float32)
    Ks =  (1-basefrac)*(1-baseprob)   *np.array([K,K/4],dtype=np.float32)
    Kbs =((1-basefrac)*(1-baseprob)-1)*np.array([K,K/4],dtype=np.float32)
    Hb = rX*(1+((1-basefrac)*(1-baseinp)-1)*cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    Hp = rX*(1+                             cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    eH = CVh
    sW = np.array([[SoriE,SoriI],[SoriE,SoriI]],dtype=np.float32)
    sH = np.array([SoriF,SoriF],dtype=np.float32)
    
    sW2 = sW**2
    
    xpeaks = np.array([0,-dori])
    solve_width = get_2feat_solve_width(sa,dori,L)
    
    muHb = tau*Hb
    SigHb = (muHb*eH)**2
    muHa = tau*(Hb+(Hp-Hb)*basesubwrapnorm(sa,sH,L))
    SigHa = (muHa*eH)**2
    muHp = tau*Hp
    SigHp = (muHp*eH)**2
    
    FE,FI,ME,MI,CE,CI = base_itp_moments(res_dir)
    FL,ML,CL = opto_itp_moments(res_dir,prms['L'],prms['CVL'])
    
    def base_M(mui,Sigii,out):
        out[0] = ME(mui[0],Sigii[0])[0]
        out[1] = MI(mui[1],Sigii[1])[0]
        
    def base_C(mui,Sigii,Sigij,out):
        out[0] = CE(mui[0],Sigii[0],Sigij[0])[0]
        out[1] = CI(mui[1],Sigii[1],Sigij[1])[0]
    
    def opto_M(mui,Sigii,out):
        out[0] = ML(mui[0],Sigii[0])[0]
        out[1] = MI(mui[1],Sigii[1])[0]
        
    def opto_C(mui,Sigii,Sigij,out):
        out[0] = CL(mui[0],Sigii[0],Sigij[0])[0]
        out[1] = CI(mui[1],Sigii[1],Sigij[1])[0]
    
    start = time.process_time()
    
    if which=='base':
        full_rb,full_ra,full_rp,full_Crb,full_Cra,full_Crp,\
            convb,conva,convp = sparse_2feat_ring_dmft(tau,W,Ks,Hb,Hp,eH,sW,sH,sa,base_M,base_C,Twrm,Tsav,dt,Kb=Kbs,L=L)
    elif which=='opto':
        full_rb,full_ra,full_rp,full_Crb,full_Cra,full_Crp,\
            convb,conva,convp = sparse_2feat_ring_dmft(tau,W,Ks,Hb,Hp,eH,sW,sH,sa,opto_M,opto_C,Twrm,Tsav,dt,Kb=Kbs,L=L)
    elif which=='both':
        full_rb,full_ra,full_rp,full_Crb,full_Cra,full_Crp,\
            convb,conva,convp = doub_sparse_2feat_ring_dmft(tau,W,Ks,Hb,Hp,eH,sW,sH,sa,[base_M,opto_M],[base_C,opto_C],
                                                      Twrm,Tsav,dt,Kb=Kbs,L=L)
    else:
        raise NotImplementedError('Only implemented options for \'which\' keyword are: \'base\', \'opto\', and \'both\'')
        
    print('integrating first stage took',time.process_time() - start,'s')

    # extract predicted moments after long time evolution
    rb = full_rb[:,-1]
    ra = full_ra[:,-1]
    rp = full_rp[:,-1]
    Crb = full_Crb[:,-1,-1:-Nsav-1:-1]
    Cra = full_Cra[:,-1,-1:-Nsav-1:-1]
    Crp = full_Crp[:,-1,-1:-Nsav-1:-1]
    
    muW = tau[:,None]*W*Ks
    SigW = tau[:,None]**2*W**2*Ks
    muWb = tau[:,None]*W*Kbs
    SigWb = tau[:,None]**2*W**2*Kbs
    
    sr = solve_width((ra-rb)/(rp-rb))
    sCr = solve_width((Cra-Crb)/(Crp-Crb))
    
    if which in ('base','opto'):
        sWr = np.sqrt(sW2+sr**2)
        rOinv = np.sum(inv_overlap(xpeaks,sr[:,None])[:,:,0],-1)
        rpmb = (rp - rb)*rOinv
        sWCr = np.sqrt(sW2[:,:,None]+sCr[None,:,:]**2)
        CrOinv = np.sum(inv_overlap(xpeaks,sCr.flatten()[:,None])[:,:,0],-1).reshape(-1,Nsav)
        Crpmb = (Crp - Crb)*CrOinv
        mub = (muW+muWb)@rb + (unstruct_fact(sr,L)*muWb)@rpmb + muHb
        mua = mub + ((struct_fact(sa,sWr,sr,L)+struct_fact(sa+dori,sWr,sr,L))*muW)@rpmb + muHa-muHb
        mup = mub + ((struct_fact(0,sWr,sr,L)+struct_fact(dori,sWr,sr,L))*muW)@rpmb + muHp-muHb
        mub = mub + (2*struct_fact(L/2,sWr,sr,L)*muW)@rpmb
        Sigb = (SigW+SigWb)@Crb + each_matmul(unstruct_fact(sCr,L)*SigWb[:,:,None],Crpmb) + (SigHb)[:,None]
        Siga = Sigb + each_matmul((struct_fact(sa,sWCr,sCr,L)+\
                                   struct_fact(sa+dori,sWCr,sCr,L))*SigW[:,:,None],Crpmb) + (SigHa-SigHb)[:,None]
        Sigp = Sigb + each_matmul((struct_fact(0,sWCr,sCr,L)+\
                                   struct_fact(dori,sWCr,sCr,L))*SigW[:,:,None],Crpmb) + (SigHp-SigHb)[:,None]
        Sigb = Sigb + each_matmul(2*struct_fact(L/2,sWCr,sCr,L)*SigW[:,:,None],Crpmb)
    elif which=='both':
        doub_muW = doub_mat(muW)
        doub_SigW = doub_mat(SigW)
        doub_muWb = doub_mat(muWb)
        doub_SigWb = doub_mat(SigWb)
        
        sWr = np.sqrt(doub_mat(sW2)+sr**2)
        rOinv = np.sum(inv_overlap(xpeaks,sr[:,None])[:,:,0],-1)
        rpmb = (rp - rb)*rOinv
        sWCr = np.sqrt(doub_mat(sW2)[:,:,None]+sCr[None,:,:]**2)
        CrOinv = np.sum(inv_overlap(xpeaks,sCr.flatten()[:,None])[:,:,0],-1).reshape(-1,Nsav)
        Crpmb = (Crp - Crb)*CrOinv
        mub = (doub_muW+doub_muWb)@rb + (unstruct_fact(sr,L)*doub_muWb)@rpmb + doub_vec(muHb)
        mua = mub + ((struct_fact(sa,sWr,sr,L)+struct_fact(sa+dori,sWr,sr,L))*doub_muW)@rpmb + doub_vec(muHa-muHb)
        mup = mub + ((struct_fact(0,sWr,sr,L)+struct_fact(dori,sWr,sr,L))*doub_muW)@rpmb + doub_vec(muHp-muHb)
        mub = mub + (2*struct_fact(L/2,sWr,sr,L)*doub_muW)@rpmb
        Sigb = (doub_SigW+doub_SigWb)@Crb + each_matmul(unstruct_fact(sCr,L)*doub_SigWb[:,:,None],Crpmb) +\
            doub_vec(SigHb)[:,None]
        Siga = Sigb + each_matmul((struct_fact(sa,sWCr,sCr,L)+\
                                   struct_fact(sa+dori,sWCr,sCr,L))*doub_SigW[:,:,None],Crpmb) +\
                                       doub_vec(SigHa-SigHb)[:,None]
        Sigp = Sigb + each_matmul((struct_fact(0,sWCr,sCr,L)+\
                                   struct_fact(dori,sWCr,sCr,L))*doub_SigW[:,:,None],Crpmb) +\
                                       doub_vec(SigHp-SigHb)[:,None]
        Sigb = Sigb + each_matmul(2*struct_fact(L/2,sWCr,sCr,L)*doub_SigW[:,:,None],Crpmb)
    else:
        raise NotImplementedError('Only implemented options for \'which\' keyword are: \'base\', \'opto\', and \'both\'')
    
    res_dict = {}
    
    res_dict['rb'] = rb
    res_dict['ra'] = ra
    res_dict['rp'] = rp
    res_dict['sr'] = sr
    res_dict['Crb'] = Crb
    res_dict['Cra'] = Cra
    res_dict['Crp'] = Crp
    res_dict['sCr'] = sCr
    
    res_dict['mub'] = mub
    res_dict['mua'] = mua
    res_dict['mup'] = mup
    res_dict['Sigb'] = Sigb
    res_dict['Siga'] = Siga
    res_dict['Sigp'] = Sigp
    
    res_dict['convb'] = convb
    res_dict['conva'] = conva
    res_dict['convp'] = convp
    
    if return_full:
        res_dict['full_rb'] = full_rb
        res_dict['full_ra'] = full_ra
        res_dict['full_rp'] = full_rp
        res_dict['full_Crb'] = full_Crb
        res_dict['full_Cra'] = full_Cra
        res_dict['full_Crp'] = full_Crp
    
    return res_dict

def run_first_stage_full_ring_dmft(prms,rX,cA,CVh,res_dir,rc,Twrm,Tsav,dt,L=180,Nori=20,
                                   which='both',return_full=False):    
    Nsav = round(Tsav/dt)+1
    
    K = prms['K']
    SoriE = prms['SoriE']
    SoriI = prms['SoriI']
    SoriF = prms['SoriF']
    J = prms['J']
    beta = prms['beta']
    gE = prms['gE']
    gI = prms['gI']
    hE = prms['hE']
    hI = prms['hI']
    basefrac = prms.get('basefrac',0)
    baseinp = prms.get('baseinp',0)
    baseprob = prms.get('baseprob',0)
    
    tau = np.array([rc.tE,rc.tI],dtype=np.float32)
    W = J*np.array([[1,-gE],[1./beta,-gI/beta]],dtype=np.float32)
    Ks =  (1-basefrac)*(1-baseprob)   *np.array([K,K/4],dtype=np.float32)
    Kbs =((1-basefrac)*(1-baseprob)-1)*np.array([K,K/4],dtype=np.float32)
    Hb = rX*(1+((1-basefrac)*(1-baseinp)-1)*cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    Hp = rX*(1+                             cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    eH = CVh
    sW = np.array([[SoriE,SoriI],[SoriE,SoriI]],dtype=np.float32)
    sH = np.array([SoriF,SoriF],dtype=np.float32)
    
    oris = np.arange(Nori)/Nori * L
    doris = oris[:,None] - oris[None,:]
    kerns = wrapnormdens(doris[:,None,:,None],sW[None,:,None,:],L)
    
    muHs = tau*(Hb[None,:]+(Hp-Hb)[None,:]*basesubwrapnorm(oris[:,None],sH[None,:],L))
    SigHs = (muHs*eH)**2
    
    FE,FI,ME,MI,CE,CI = base_itp_moments(res_dir)
    FL,ML,CL = opto_itp_moments(res_dir,prms['L'],prms['CVL'])
    
    def base_M(mui,Sigii,out):
        out[0] = ME(mui[0],Sigii[0])[0]
        out[1] = MI(mui[1],Sigii[1])[0]
        
    def base_C(mui,Sigii,Sigij,out):
        out[0] = CE(mui[0],Sigii[0],Sigij[0])[0]
        out[1] = CI(mui[1],Sigii[1],Sigij[1])[0]
    
    def opto_M(mui,Sigii,out):
        out[0] = ML(mui[0],Sigii[0])[0]
        out[1] = MI(mui[1],Sigii[1])[0]
        
    def opto_C(mui,Sigii,Sigij,out):
        out[0] = CL(mui[0],Sigii[0],Sigij[0])[0]
        out[1] = CI(mui[1],Sigii[1],Sigij[1])[0]
    
    start = time.process_time()
    
    if which=='base':
        full_rs,full_Crs,convs = sparse_full_ring_dmft(tau,W,Ks,Hb,Hp,eH,sW,sH,base_M,base_C,
                                                       Twrm,Tsav,dt,Kb=Kbs,L=L,Nori=Nori)
    elif which=='opto':
        full_rs,full_Crs,convs = sparse_full_ring_dmft(tau,W,Ks,Hb,Hp,eH,sW,sH,opto_M,opto_C,
                                                       Twrm,Tsav,dt,Kb=Kbs,L=L,Nori=Nori)
    elif which=='both':
        full_rs,full_Crs,convs = doub_sparse_full_ring_dmft(tau,W,Ks,Hb,Hp,eH,sW,sH,[base_M,opto_M],[base_C,opto_C],
                                                            Twrm,Tsav,dt,Kb=Kbs,L=L,Nori=Nori)
    else:
        raise NotImplementedError('Only implemented options for \'which\' keyword are: \'base\', \'opto\', and \'both\'')
        
    print('integrating first stage took',time.process_time() - start,'s')

    # extract predicted moments after long time evolution
    rs = full_rs[:,:,-1]
    Crs = full_Crs[:,:,-1,-1:-Nsav-1:-1]
    
    muW = tau[:,None]*W*Ks
    SigW = tau[:,None]**2*W**2*Ks
    muWb = tau[:,None]*W*Kbs
    SigWb = tau[:,None]**2*W**2*Kbs
    
    muWs = (muWb[None,:,None,:] + muW[None,:,None,:]*2*np.pi*kerns) / Nori
    SigWs = (SigWb[None,:,None,:] + SigW[None,:,None,:]*2*np.pi*kerns) / Nori
    
    if which in ('base','opto'):
        mus = np.einsum('ijkl,kl->ij',muWs,rs) + muHs
        Sigs = np.einsum('ijkl,klm->ijm',SigWs,Crs) + SigHs[:,:,None]
    elif which=='both':
        doub_tau = doub_vec(tau)
        doub_muW = doub_mat(muW)
        doub_SigW = doub_mat(SigW)
        doub_muWb = doub_mat(muWb)
        doub_SigWb = doub_mat(SigWb)
        doub_sW = np.block([[sW,sW],[sW,sW]])
        doub_Hb = doub_vec(Hb)
        doub_Hp = doub_vec(Hp)
        doub_sH = doub_vec(sH)

        doub_kerns = wrapnormdens(doris[:,None,:,None],doub_sW[None,:,None,:],L)

        doub_muWs = (doub_muWb[None,:,None,:] + doub_muW[None,:,None,:]*2*np.pi*doub_kerns) / Nori
        doub_SigWs = (doub_SigWb[None,:,None,:] + doub_SigW[None,:,None,:]*2*np.pi*doub_kerns) / Nori

        doub_muHs = doub_tau*(doub_Hb[None,:]+(doub_Hp-doub_Hb)[None,:]*basesubwrapnorm(oris[:,None],doub_sH[None,:],L))
        doub_SigHs = (doub_muHs*eH)**2
        
        mus = np.einsum('ijkl,kl->ij',doub_muWs,rs) + doub_muHs
        Sigs = np.einsum('ijkl,klm->ijm',doub_SigWs,Crs) + doub_SigHs[:,:,None]
    else:
        raise NotImplementedError('Only implemented options for \'which\' keyword are: \'base\', \'opto\', and \'both\'')
    
    res_dict = {}
    
    res_dict['rs'] = rs
    res_dict['Crs'] = Crs
    
    res_dict['mus'] = mus
    res_dict['Sigs'] = Sigs
    
    res_dict['convs'] = convs
    
    if return_full:
        res_dict['full_rs'] = full_rs
        res_dict['full_Crs'] = full_Crs
    
    return res_dict

def run_second_stage_full_ring_dmft(first_res_dict,prms,rX,cA,CVh,res_dir,rc,Twrm,Tsav,dt,
                                    L=180,Nori=20,return_full=False):
    Nsav = round(Tsav/dt)+1
    
    K = prms['K']
    SoriE = prms['SoriE']
    SoriI = prms['SoriI']
    SoriF = prms['SoriF']
    J = prms['J']
    beta = prms['beta']
    gE = prms['gE']
    gI = prms['gI']
    hE = prms['hE']
    hI = prms['hI']
    basefrac = prms.get('basefrac',0)
    baseinp = prms.get('baseinp',0)
    baseprob = prms.get('baseprob',0)
    
    tau = np.array([rc.tE,rc.tI],dtype=np.float32)
    W = J*np.array([[1,-gE],[1./beta,-gI/beta]],dtype=np.float32)
    Ks =  (1-basefrac)*(1-baseprob)   *np.array([K,K/4],dtype=np.float32)
    Kbs =((1-basefrac)*(1-baseprob)-1)*np.array([K,K/4],dtype=np.float32)
    Hb = rX*(1+((1-basefrac)*(1-baseinp)-1)*cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    Hp = rX*(1+                             cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    eH = CVh
    sW = np.array([[SoriE,SoriI],[SoriE,SoriI]],dtype=np.float32)
    sH = np.array([SoriF,SoriF],dtype=np.float32)
    
    oris = np.arange(Nori)/Nori * L
    doris = oris[:,None] - oris[None,:]
    kerns = wrapnormdens(doris[:,None,:,None],sW[None,:,None,:],L)
    
    FE,FI,ME,MI,CE,CI = base_itp_moments(res_dir)
    FL,ML,CL = opto_itp_moments(res_dir,prms['L'],prms['CVL'])

    def diff_R(mu1i,mu2i,Sig1ii,Sig2ii,kij,out):
        out[0] = R_simp(ME,ML,mu1i[0],mu2i[0],Sig1ii[0],Sig2ii[0],kij[0])
        out[1] = R_simp(MI,MI,mu1i[1],mu2i[1],Sig1ii[1],Sig2ii[1],kij[1])

    rs = first_res_dict['rs']
    Crs = first_res_dict['Crs']
    
    muW = tau[:,None]*W*Ks
    SigW = tau[:,None]**2*W**2*Ks
    muWb = tau[:,None]*W*Kbs
    SigWb = tau[:,None]**2*W**2*Kbs
    
    SigWs = (SigWb[None,:,None,:] + SigW[None,:,None,:]*2*np.pi*kerns) / Nori
    
    doub_tau = doub_vec(tau)
    doub_muW = doub_mat(muW)
    doub_SigW = doub_mat(SigW)
    doub_muWb = doub_mat(muWb)
    doub_SigWb = doub_mat(SigWb)
    doub_sW = np.block([[sW,sW],[sW,sW]])
    doub_Hb = doub_vec(Hb)
    doub_Hp = doub_vec(Hp)
    doub_sH = doub_vec(sH)
    
    doub_kerns = wrapnormdens(doris[:,None,:,None],doub_sW[None,:,None,:],L)

    doub_muWs = (doub_muWb[None,:,None,:] + doub_muW[None,:,None,:]*2*np.pi*doub_kerns) / Nori
    doub_SigWs = (doub_SigWb[None,:,None,:] + doub_SigW[None,:,None,:]*2*np.pi*doub_kerns) / Nori

    doub_muHs = doub_tau*(doub_Hb[None,:]+(doub_Hp-doub_Hb)[None,:]*basesubwrapnorm(oris[:,None],doub_sH[None,:],L))
    doub_SigHs = (doub_muHs*eH)**2
    
    mus = np.einsum('ijkl,kl->ij',doub_muWs,rs) + doub_muHs
    Sigs = np.einsum('ijkl,klm->ijm',doub_SigWs,Crs) + doub_SigHs[:,:,None]

    start = time.process_time()

    full_Cdrs,convds = diff_sparse_full_ring_dmft(tau,W,Ks,Hb,Hp,eH,sW,sH,diff_R,Twrm,Tsav,dt,
                                                  rs,Crs,Kb=Kbs,L=L,Nori=Nori)

    print('integrating second stage took',time.process_time() - start,'s')

    # extract predicted moments after long time evolution
    drs = rs[:,2:] - rs[:,:2]
    Cdrs = full_Cdrs[:,:,-1,-1:-Nsav-1:-1]

    dmus = mus[:,2:] - mus[:,:2]

    Sigds = np.einsum('ijkl,klm->ijm',SigWs,Cdrs)
    
    res_dict = {}
    
    res_dict['drs'] = drs
    res_dict['Cdrs'] = Cdrs
    
    res_dict['dmus'] = dmus
    res_dict['Sigds'] = Sigds

    res_dict['convds'] = convds
    
    if return_full:
        res_dict['full_Cdrs'] = full_Cdrs
    
    return res_dict

def run_two_stage_full_ring_dmft(prms,rX,cA,CVh,res_dir,rc,Twrm,Tsav,dt,L=180,Nori=20,
                                 which='both',return_full=False):    
    Nsav = round(Tsav/dt)+1
    
    K = prms['K']
    SoriE = prms['SoriE']
    SoriI = prms['SoriI']
    SoriF = prms['SoriF']
    J = prms['J']
    beta = prms['beta']
    gE = prms['gE']
    gI = prms['gI']
    hE = prms['hE']
    hI = prms['hI']
    basefrac = prms.get('basefrac',0)
    baseinp = prms.get('baseinp',0)
    baseprob = prms.get('baseprob',0)
    
    tau = np.array([rc.tE,rc.tI],dtype=np.float32)
    W = J*np.array([[1,-gE],[1./beta,-gI/beta]],dtype=np.float32)
    Ks =  (1-basefrac)*(1-baseprob)   *np.array([K,K/4],dtype=np.float32)
    Kbs =((1-basefrac)*(1-baseprob)-1)*np.array([K,K/4],dtype=np.float32)
    Hb = rX*(1+((1-basefrac)*(1-baseinp)-1)*cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    Hp = rX*(1+                             cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    eH = CVh
    sW = np.array([[SoriE,SoriI],[SoriE,SoriI]],dtype=np.float32)
    sH = np.array([SoriF,SoriF],dtype=np.float32)
    
    oris = np.arange(Nori)/Nori * L
    doris = oris[:,None] - oris[None,:]
    kerns = wrapnormdens(doris[:,None,:,None],sW[None,:,None,:],L)
    
    muHs = tau*(Hb[None,:]+(Hp-Hb)[None,:]*basesubwrapnorm(oris[:,None],sH[None,:],L))
    SigHs = (muHs*eH)**2
    
    FE,FI,ME,MI,CE,CI = base_itp_moments(res_dir)
    FL,ML,CL = opto_itp_moments(res_dir,prms['L'],prms['CVL'])
    
    def base_M(mui,Sigii,out):
        out[0] = ME(mui[0],Sigii[0])[0]
        out[1] = MI(mui[1],Sigii[1])[0]
        
    def base_C(mui,Sigii,Sigij,out):
        out[0] = CE(mui[0],Sigii[0],Sigij[0])[0]
        out[1] = CI(mui[1],Sigii[1],Sigij[1])[0]
    
    def opto_M(mui,Sigii,out):
        out[0] = ML(mui[0],Sigii[0])[0]
        out[1] = MI(mui[1],Sigii[1])[0]
        
    def opto_C(mui,Sigii,Sigij,out):
        out[0] = CL(mui[0],Sigii[0],Sigij[0])[0]
        out[1] = CI(mui[1],Sigii[1],Sigij[1])[0]

    def diff_R(mu1i,mu2i,Sig1ii,Sig2ii,kij,out):
        out[0] = R_simp(ME,ML,mu1i[0],mu2i[0],Sig1ii[0],Sig2ii[0],kij[0])
        out[1] = R_simp(MI,MI,mu1i[1],mu2i[1],Sig1ii[1],Sig2ii[1],kij[1])
    
    start = time.process_time()
    
    full_rs,full_Crs,convs = doub_sparse_full_ring_dmft(tau,W,Ks,Hb,Hp,eH,sW,sH,[base_M,opto_M],[base_C,opto_C],
                                                        Twrm,Tsav,dt,Kb=Kbs,L=L,Nori=Nori)
        
    print('integrating first stage took',time.process_time() - start,'s')

    # extract predicted moments after long time evolution
    rs = full_rs[:,:,-1]
    Crs = full_Crs[:,:,-1,-1:-Nsav-1:-1]
    
    muW = tau[:,None]*W*Ks
    SigW = tau[:,None]**2*W**2*Ks
    muWb = tau[:,None]*W*Kbs
    SigWb = tau[:,None]**2*W**2*Kbs
    
    SigWs = (SigWb[None,:,None,:] + SigW[None,:,None,:]*2*np.pi*kerns) / Nori
    
    doub_tau = doub_vec(tau)
    doub_muW = doub_mat(muW)
    doub_SigW = doub_mat(SigW)
    doub_muWb = doub_mat(muWb)
    doub_SigWb = doub_mat(SigWb)
    doub_sW = np.block([[sW,sW],[sW,sW]])
    doub_Hb = doub_vec(Hb)
    doub_Hp = doub_vec(Hp)
    doub_sH = doub_vec(sH)
    
    doub_kerns = wrapnormdens(doris[:,None,:,None],doub_sW[None,:,None,:],L)

    doub_muWs = (doub_muWb[None,:,None,:] + doub_muW[None,:,None,:]*2*np.pi*doub_kerns) / Nori
    doub_SigWs = (doub_SigWb[None,:,None,:] + doub_SigW[None,:,None,:]*2*np.pi*doub_kerns) / Nori

    doub_muHs = doub_tau*(doub_Hb[None,:]+(doub_Hp-doub_Hb)[None,:]*basesubwrapnorm(oris[:,None],doub_sH[None,:],L))
    doub_SigHs = (doub_muHs*eH)**2
    
    mus = np.einsum('ijkl,kl->ij',doub_muWs,rs) + doub_muHs
    Sigs = np.einsum('ijkl,klm->ijm',doub_SigWs,Crs) + doub_SigHs[:,:,None]

    start = time.process_time()

    full_Cdrs,convds = diff_sparse_full_ring_dmft(tau,W,Ks,Hb,Hp,eH,sW,sH,diff_R,Twrm,Tsav,dt,
                                                  rs,Crs,Kb=Kbs,L=L,Nori=Nori)

    print('integrating second stage took',time.process_time() - start,'s')

    # extract predicted moments after long time evolution
    drs = rs[:,2:] - rs[:,:2]
    Cdrs = full_Cdrs[:,:,-1,-1:-Nsav-1:-1]

    dmus = mus[:,2:] - mus[:,:2]

    Sigds = np.einsum('ijkl,klm->ijm',SigWs,Cdrs)
    
    res_dict = {}
    
    res_dict['rs'] = rs
    res_dict['Crs'] = Crs
    res_dict['drs'] = drs
    res_dict['Cdrs'] = Cdrs
    
    res_dict['mus'] = mus
    res_dict['Sigs'] = Sigs
    res_dict['dmus'] = dmus
    res_dict['Sigds'] = Sigds

    res_dict['convs'] = convs
    res_dict['convds'] = convds
    
    if return_full:
        res_dict['full_rs'] = full_rs
        res_dict['full_Crs'] = full_Crs
        res_dict['full_Cdrs'] = full_Cdrs
    
    return res_dict

def lin_resp_mats(tau,muW,SigW,dmuH,dSigH,M_fn,C_fn,Tsav,dt,mu,Sig):
    Ntyp = len(dmuH)
    Nsav = round(Tsav/dt)+1
    
    def Md_fn(mu,Sig,out):
        outr = np.zeros_like(out)
        outl = np.zeros_like(out)
        M_fn(mu+dmu,Sig,outr)
        M_fn(mu-dmu,Sig,outl)
        out[:] = (outr-outl)/(2*dmu)
    
    def Md2_fn(mu,Sig,out):
        outr = np.zeros_like(out)
        outl = np.zeros_like(out)
        M_fn(mu,Sig+dmu**2,outr)
        M_fn(mu,Sig-dmu**2,outl)
        out[:] = (outr-outl)/dmu**2
    
    def Rd_fn(mu,Sig,Cov,out):
        outr = np.zeros_like(out)
        outl = np.zeros_like(out)
        C_fn(mu+dmu,Sig,Cov,outr)
        C_fn(mu-dmu,Sig,Cov,outl)
        out[:] = (outr-outl)/(4*dmu)
    
    def Rd2_fn(mu,Sig,Cov,out):
        outr = np.zeros_like(out)
        outl = np.zeros_like(out)
        C_fn(mu,np.fmax(Cov,Sig+dmu**2),Cov,outr)
        C_fn(mu,np.fmax(Cov,Sig-dmu**2),Cov,outl)
        out[:] = (outr-outl)/(np.fmax(Cov,Sig+dmu**2)-np.fmax(Cov,Sig-dmu**2))
    
    def Cd_fn(mu,Sig,Cov,out):
        outr = np.zeros_like(out)
        outl = np.zeros_like(out)
        C_fn(mu,Sig,np.fmin(Sig,Cov+dmu**2),outr)
        C_fn(mu,Sig,np.fmin(Sig,Cov-dmu**2),outl)
        out[:] = (outr-outl)/(np.fmin(Sig,Cov+dmu**2)-np.fmin(Sig,Cov-dmu**2))
    
    Mdphi = np.empty((Ntyp,),dtype=np.float32)
    Md2phi = np.empty((Ntyp,),dtype=np.float32)
    Rdphi = np.empty((Ntyp,Nsav),dtype=np.float32)
    Rd2phi = np.empty((Ntyp,Nsav),dtype=np.float32)
    Cdphi = np.empty((Ntyp,Nsav),dtype=np.float32)
    
    Rd2phi = smooth_func(Rd2phi,dt)
    Cdphi = smooth_func(Cdphi,dt)
    
    Md_fn(mu,Sig[:,0],Mdphi)
    Md2_fn(mu,Sig[:,0],Md2phi)
    for i in range(Nsav):
        Rd_fn(mu,Sig[:,0],Sig[:,i],Rdphi[:,i])
        Rd2_fn(mu,Sig[:,0],Sig[:,i],Rd2phi[:,i])
        Cd_fn(mu,Sig[:,0],Sig[:,i],Cdphi[:,i])
        
    d2_mat = d2_stencil(Tsav,dt)
    del_vec = np.concatenate(([1],np.zeros(Nsav-1)))
    
    res_dict = {}
    
    res_dict['d2_mat'] = d2_mat
    res_dict['Mdphi'] = Mdphi
    res_dict['Md2phi'] = Md2phi
    res_dict['Rdphi'] = Rdphi
    res_dict['Rd2phi'] = Rd2phi
    res_dict['Cdphi'] = Cdphi
    
    res_dict['A'] = np.eye(Ntyp) - Mdphi[:,None] * muW
    res_dict['B'] = -0.5 * Md2phi[:,None,None] * SigW[:,:,None] * del_vec[None,None,:]
    res_dict['C'] = -2 * Rdphi[:,:,None] * muW[:,None,:]
    res_dict['D'] = np.eye(Ntyp)[:,None,:,None]*np.eye(Nsav)[None,:,None,:] +\
        - (np.diag(tau**2)[:,None,:,None] - dt*np.diag(tau)[:,None,:,None]) * d2_mat[None,:,None,:] +\
        - Cdphi[:,:,None,None] * SigW[:,None,:,None] * np.eye(Nsav)[None,:,None,:] +\
        - Rd2phi[:,:,None,None] * SigW[:,None,:,None] * del_vec[None,None,None,:]
    res_dict['D0dis'] = np.eye(Ntyp)[:,None,:,None]*np.eye(Nsav)[None,:,None,:] +\
        - (np.diag(tau**2)[:,None,:,None] - dt*np.diag(tau)[:,None,:,None]) * d2_mat[None,:,None,:]
    
    if dSigH.ndim==1:
        res_dict['E'] = Mdphi * dmuH + 0.5 * Md2phi * dSigH
        res_dict['F'] = 2 * Rdphi * dmuH[:,None] + (Cdphi + Rd2phi) * dSigH[:,None]
    else:
        res_dict['E'] = Mdphi * dmuH + 0.5 * Md2phi * dSigH[:,0]
        res_dict['F'] = 2 * Rdphi * dmuH[:,None] + Cdphi * dSigH + Rd2phi * dSigH[:,0]
        
    return res_dict