import os
import pickle
import numpy as np
from scipy.linalg import toeplitz
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import quad,simpson
from scipy.special import erf
import time

sr2pi = np.sqrt(2*np.pi)
sr8 = np.sqrt(8)

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
        return np.einsum('ijj->ij',A)
    else:
        out = np.zeros(np.array(A.shape[:2])-[0,k])
        for i in range(len(A)):
            out[i] = np.diag(A[i],k)
        return out

def each_matmul(A,B):
    return np.einsum('ijk,jk->ik',A,B)

def grid_stat(stat,A,Tstat,dt):
    Ntyp = A.shape[0]
    Nsav = A.shape[1]
    Nstat = round(Tstat/dt)+1
    A_ext = np.zeros((Ntyp,Nstat))
    if Nsav < Nstat:
        A_ext[:,:Nsav] = A
        A_ext[:,Nsav:] = A[:,-1:]
    else:
        A_ext = A[:,:Nstat]
    A_mat = np.array([toeplitz(A_ext[typ_idx]) for typ_idx in range(Ntyp)])
    return stat(A_mat,axis=(1,2))

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
        
    Mphi = np.empty((Ntyp),dtype=np.float32)
    Cphi = np.empty((Ntyp),dtype=np.float32)
    
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
    
    Cdr = np.zeros((Ntyp,Nint,Nint),dtype=np.float32)
    
    if Cdr0 is None:
        # Cdr0 = Cr[:Ntyp]
        Cdr0 = (r[Ntyp:] - r[:Ntyp]).astype(np.float32)[:,None]**2 + 1e3
        
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
        
    Rphi = np.empty((Ntyp),dtype=np.float32)
    
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
                dttauinv2*(-Cdr[:,i,j]+Cr[:Ntyp,ij_idx]+Cr[Ntyp:,ij_idx]-2*Rphi)
            
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

def sparse_ring_dmft(tau,W,K,Hb,Hp,eH,sW,sH,sa,M_fn,C_fn,Twrm,Tsav,dt,
                rb0=None,ra0=None,rp0=None,Crb0=None,Cra0=None,Crp0=None,Kb=None):
    if Kb is None:
        Kb = np.zeros_like(K)
    S = 180
        
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
    
    sa2 = sa**2
    sW2 = sW**2
    sH2 = sH**2
        
    muWb = tau[:,None]*W*Kb
    SigWb = tau[:,None]**2*W**2*Kb
    muW = tau[:,None]*W*K
    SigW = tau[:,None]**2*W**2*K
    
    muHb = tau*Hb
    SigHb = (muHb*eH)**2
    muHa = tau*(Hb+(Hp-Hb)*np.exp(-0.5*sa2/sH2))
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
        
    Mphib = np.empty((Ntyp),dtype=np.float32)
    Mphia = np.empty((Ntyp),dtype=np.float32)
    Mphip = np.empty((Ntyp),dtype=np.float32)
    Cphib = np.empty((Ntyp),dtype=np.float32)
    Cphia = np.empty((Ntyp),dtype=np.float32)
    Cphip = np.empty((Ntyp),dtype=np.float32)
    
    def drdt(rbi,rai,rpi,Sigbii,Sigaii,Sigpii):
        sr2i = np.fmax(1e-1,0.5*sa2/np.log(np.fmax(np.abs((rpi-rbi)/(rai-rbi)),1+1e-4)))
        sWr2i = sW2+sr2i
        sri = np.sqrt(sr2i)
        srdivsWri = np.sqrt(sr2i/sWr2i)
        rpmbi = rpi - rbi
        mubi = (muW+muWb)@rbi + (sri*sr2pi/S*erf(S/(sr8*sri))*muWb)@rpmbi + muHb
        muai = mubi + (srdivsWri*np.exp(-0.5*sa2/sWr2i)*muW)@rpmbi + muHa-muHb
        mupi = mubi + (srdivsWri*muW)@rpmbi + muHp-muHb
        M_fn(mubi,Sigbii,Mphib)
        M_fn(muai,Sigaii,Mphia)
        M_fn(mupi,Sigpii,Mphip)
        return tauinv*(-rbi + Mphib), tauinv*(-rai + Mphia),  tauinv*(-rpi + Mphip)
    
    for i in range(Nint-1):
        Crbii = Crb[:,i,i]
        Craii = Cra[:,i,i]
        Crpii = Crp[:,i,i]
        sCr2ii = np.fmax(1e-1,0.5*sa2/np.log(np.fmax(np.abs((Crpii-Crbii)/(Craii-Crbii)),1+1e-4)))
        sWCr2ii = sW2+sCr2ii
        sCrii = np.sqrt(sCr2ii)
        sCrdivsWCr2ii = np.sqrt(sCr2ii/sWCr2ii)
        Crpmbii = Crpii - Crbii
        Sigbii = (SigW+SigWb)@Crbii + (sCrii*sr2pi/S*erf(S/(sr8*sCrii))*SigWb)@Crpmbii + SigHb
        Sigaii = Sigbii + (sCrdivsWCr2ii*np.exp(-0.5*sa2/sWCr2ii)*SigW)@Crpmbii + SigHa-SigHb
        Sigpii = Sigbii + (sCrdivsWCr2ii*SigW)@Crpmbii + SigHp-SigHb
            
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
        sr2i = np.fmax(1e-1,0.5*sa2/np.log(np.fmax(np.abs((rpi-rbi)/(rai-rbi)),1+1e-4)))
        sWr2i = sW2+sr2i
        sri = np.sqrt(sr2i)
        srdivsWri = np.sqrt(sr2i/sWr2i)
        rpmbi = rpi - rbi
        mubi = (muW+muWb)@rbi + (sri*sr2pi/S*erf(S/(sr8*sri))*muWb)@rpmbi + muHb
        muai = mubi + (srdivsWri*np.exp(-0.5*sa2/sWr2i)*muW)@rpmbi + muHa-muHb
        mupi = mubi + (srdivsWri*muW)@rpmbi + muHp-muHb
        
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
            sCr2ij = np.fmax(1e-1,0.5*sa2/np.log(np.fmax(np.abs((Crpij-Crbij)/(Craij-Crbij)),1+1e-4)))
            sWCr2ij = sW2+sCr2ij
            sCrij = np.sqrt(sCr2ij)
            sCrdivsWCr2ij = np.sqrt(sCr2ij/sWCr2ij)
            Crpmbij = Crpij - Crbij
            Sigbij = (SigW+SigWb)@Crbij + (sCrij*sr2pi/S*erf(S/(sr8*sCrij))*SigWb)@Crpmbij + SigHb
            Sigaij = Sigbij + (sCrdivsWCr2ij*np.exp(-0.5*sa2/sWCr2ij)*SigW)@Crpmbij + SigHa-SigHb
            Sigpij = Sigbij + (sCrdivsWCr2ij*SigW)@Crpmbij + SigHp-SigHb
            C_fn(mubi,Sigbii,Sigbij,Cphib)
            C_fn(muai,Sigaii,Sigaij,Cphia)
            C_fn(mupi,Sigpii,Sigpij,Cphip)
            Crb[:,i+1,j+1] = Crb[:,i,j+1]+Crb[:,i+1,j]-Crb[:,i,j] +\
                dttauinv*(-Crb[:,i+1,j]-Crb[:,i,j+1]+2*Crb[:,i,j]) + dttauinv2*(-Crb[:,i,j]+Cphib)
            Cra[:,i+1,j+1] = Cra[:,i,j+1]+Cra[:,i+1,j]-Cra[:,i,j] +\
                dttauinv*(-Cra[:,i+1,j]-Cra[:,i,j+1]+2*Cra[:,i,j]) + dttauinv2*(-Cra[:,i,j]+Cphia)
            Crp[:,i+1,j+1] = Crp[:,i,j+1]+Crp[:,i+1,j]-Crp[:,i,j] +\
                dttauinv*(-Crp[:,i+1,j]-Crp[:,i,j+1]+2*Crp[:,i,j]) + dttauinv2*(-Crp[:,i,j]+Cphip)
            
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
                     rb0=None,ra0=None,rp0=None,Crb0=None,Cra0=None,Crp0=None,Kb=None):
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
                      rb0,ra0,rp0,Crb0,Cra0,Crp0,Kb=doub_Kb)

def diff_sparse_ring_dmft(tau,W,K,Hb,Hp,eH,sW,sH,sa,R_fn,Twrm,Tsav,dt,rb,ra,rp,Crb,Cra,Crp,
                     Cdrb0=None,Cdra0=None,Cdrp0=None,Kb=None):
    if Kb is None:
        Kb = np.zeros_like(K)
    S = 180
        
    Ntyp = len(Hb)
    Nint = round((Twrm+Tsav)/dt)+1
    Nclc = round(1.5*Tsav/dt)+1
    Nsav = round(Tsav/dt)+1
    
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
    
    sa2 = sa**2
    sW2 = sW**2
    sH2 = sH**2
    
    muW = tau[:,None]*W*K
    SigW = tau[:,None]**2*W**2*K
    muWb = tau[:,None]*W*Kb
    SigWb = tau[:,None]**2*W**2*Kb
    
    muHb = tau*Hb
    SigHb = (muHb*eH)**2
    muHa = tau*(Hb+(Hp-Hb)*np.exp(-0.5*sa2/sH2))
    SigHa = (muHa*eH)**2
    muHp = tau*Hp
    SigHp = (muHp*eH)**2
    
    doub_muW = doub_mat(muW)
    doub_SigW = doub_mat(SigW)[:,:,None]
    doub_muWb = doub_mat(muWb)
    doub_SigWb = doub_mat(SigWb)[:,:,None]
    
    sr2 = np.fmax(1e-1,0.5*sa2/np.log(np.fmax(np.abs((rp-rb)/(ra-rb)),1+1e-4)))
    sWr2 = doub_mat(sW2)+sr2
    sr = np.sqrt(sr2)
    srdivsWr = np.sqrt(sr2/sWr2)
    rpmb = rp - rb
    sCr2 = np.fmax(1e-1,0.5*sa2/np.log(np.fmax(np.abs((Crp-Crb)/(Cra-Crb)),1+1e-4)))
    sWCr2 = doub_mat(sW2)[:,:,None]+sCr2[None,:,:]
    sCr = np.sqrt(sCr2)
    sCrdivsWCr2 = np.sqrt(sCr2/sWCr2)
    Crpmb = Crp - Crb
    mub = (doub_muW+doub_muWb)@rb + (sr*sr2pi/S*erf(S/(sr8*sr))*doub_muWb)@rpmb + doub_vec(muHb)
    mua = mub + (srdivsWr*np.exp(-0.5*sa2/sWr2)*doub_muW)@rpmb + doub_vec(muHa-muHb)
    mup = mub + (srdivsWr*doub_muW)@rpmb + doub_vec(muHp-muHb)
    Sigb = each_matmul(doub_SigW+doub_SigWb,Crb) + each_matmul(sCr*sr2pi/S*erf(S/(sr8*sCr))*doub_SigWb,Crpmb) +\
        doub_vec(SigHb)[:,None]
    Siga = Sigb + each_matmul(sCrdivsWCr2*np.exp(-0.5*sa2/sWCr2)*doub_SigW,Crpmb) + doub_vec(SigHa-SigHb)[:,None]
    Sigp = Sigb + each_matmul(sCrdivsWCr2*doub_SigW,Crpmb) + doub_vec(SigHp-SigHb)[:,None]
    
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
        
    Rphib = np.empty((Ntyp),dtype=np.float32)
    Rphia = np.empty((Ntyp),dtype=np.float32)
    Rphip = np.empty((Ntyp),dtype=np.float32)
    
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
            sCdr2ij = np.fmax(1e-1,0.5*sa2/np.log(np.fmax(np.abs((Cdrpij-Cdrbij)/(Cdraij-Cdrbij)),1+1e-4)))
            sWCdr2ij = sW2+sCdr2ij
            sCdrij = np.sqrt(sCdr2ij)
            sCdrdivsWCdr2ij = np.sqrt(sCdr2ij/sWCdr2ij)
            Cdrpmbij = Cdrpij - Cdrbij
            Sigdbij = (SigW+SigWb)@Cdrbij + (sCdrij*sr2pi/S*erf(S/(sr8*sCdrij))*SigWb)@Cdrpmbij
            Sigdaij = Sigdbij + (sCdrdivsWCdr2ij*np.exp(-0.5*sa2/sWCdr2ij)*SigW)@Cdrpmbij
            Sigdpij = Sigdbij + (sCdrdivsWCdr2ij*SigW)@Cdrpmbij
            
            kbij = 0.5*(Sigb[:Ntyp,ij_idx]+Sigb[Ntyp:,ij_idx]-Sigdbij)
            kaij = 0.5*(Siga[:Ntyp,ij_idx]+Siga[Ntyp:,ij_idx]-Sigdaij)
            kpij = 0.5*(Sigp[:Ntyp,ij_idx]+Sigp[Ntyp:,ij_idx]-Sigdpij)
            
            R_fn(mub[:Ntyp],mub[Ntyp:],Sigb[:Ntyp,0],Sigb[Ntyp:,0],kbij,Rphib)
            R_fn(mua[:Ntyp],mua[Ntyp:],Siga[:Ntyp,0],Siga[Ntyp:,0],kaij,Rphia)
            R_fn(mup[:Ntyp],mup[Ntyp:],Sigp[:Ntyp,0],Sigp[Ntyp:,0],kpij,Rphip)
            
            Cdrb[:,i+1,j+1] = Cdrb[:,i,j+1]+Cdrb[:,i+1,j]-Cdrb[:,i,j] +\
                dttauinv*(-Cdrb[:,i+1,j]-Cdrb[:,i,j+1]+2*Cdrb[:,i,j]) +\
                dttauinv2*(-Cdrb[:,i,j]+Crb[:Ntyp,ij_idx]+Crb[Ntyp:,ij_idx]-2*Rphib)
            Cdra[:,i+1,j+1] = Cdra[:,i,j+1]+Cdra[:,i+1,j]-Cdra[:,i,j] +\
                dttauinv*(-Cdra[:,i+1,j]-Cdra[:,i,j+1]+2*Cdra[:,i,j]) +\
                dttauinv2*(-Cdra[:,i,j]+Cra[:Ntyp,ij_idx]+Cra[Ntyp:,ij_idx]-2*Rphia)
            Cdrp[:,i+1,j+1] = Cdrp[:,i,j+1]+Cdrp[:,i+1,j]-Cdrp[:,i,j] +\
                dttauinv*(-Cdrp[:,i+1,j]-Cdrp[:,i,j+1]+2*Cdrp[:,i,j]) +\
                dttauinv2*(-Cdrp[:,i,j]+Crp[:Ntyp,ij_idx]+Crp[Ntyp:,ij_idx]-2*Rphip)
            
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

def run_first_stage_dmft(prms,rX,CVh,res_dir,rc,Twrm,Tsav,dt,which='both',return_full=False):
    Nsav = round(Tsav/dt)+1
    
    K = prms['K']
    J = prms['J']
    beta = prms['beta']
    gE = prms['gE']
    gI = prms['gI']
    hE = prms['hE']
    hI = prms['hI']
    L = prms['L']
    CVL = prms['CVL']
    
    tau = np.array([rc.tE,rc.tI],dtype=np.float32)
    W = J*np.array([[1,-gE],[1./beta,-gI/beta]],dtype=np.float32)
    Ks = np.array([K,K/4],dtype=np.float32)
    H = rX*K*J*np.array([hE,hI/beta],dtype=np.float32)
    eH = CVh
    
    muH = tau*H
    SigH = (muH*eH)**2
    
    FE,FI,ME,MI,CE,CI = base_itp_moments(res_dir)
    FL,ML,CL = opto_itp_moments(res_dir,L,CVL)
    
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
    L = prms['L']
    CVL = prms['CVL']
    
    tau = np.array([rc.tE,rc.tI],dtype=np.float32)
    W = J*np.array([[1,-gE],[1./beta,-gI/beta]],dtype=np.float32)
    Ks = np.array([K,K/4],dtype=np.float32)
    H = rX*K*J*np.array([hE,hI/beta],dtype=np.float32)
    eH = CVh
    
    muH = tau*H
    SigH = (muH*eH)**2
    
    FE,FI,ME,MI,CE,CI = base_itp_moments(res_dir)
    FL,ML,CL = opto_itp_moments(res_dir,L,CVL)

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
    L = prms['L']
    CVL = prms['CVL']
    
    tau = np.array([rc.tE,rc.tI],dtype=np.float32)
    W = J*np.array([[1,-gE],[1./beta,-gI/beta]],dtype=np.float32)
    Ks = np.array([K,K/4],dtype=np.float32)
    H = rX*K*J*np.array([hE,hI/beta],dtype=np.float32)
    eH = CVh
    
    muH = tau*H
    SigH = (muH*eH)**2
    
    FE,FI,ME,MI,CE,CI = base_itp_moments(res_dir)
    FL,ML,CL = opto_itp_moments(res_dir,L,CVL)
    
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

def run_first_stage_ring_dmft(prms,rX,cA,CVh,res_dir,rc,Twrm,Tsav,dt,sa=15,which='both',return_full=False):
    S = 180
    
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
    L = prms['L']
    CVL = prms['CVL']
    basefrac = prms.get('basefrac',0)
    
    tau = np.array([rc.tE,rc.tI],dtype=np.float32)
    W = J*np.array([[1,-gE],[1./beta,-gI/beta]],dtype=np.float32)
    Ks = (1-basefrac)*np.array([K,K/4],dtype=np.float32)
    Kbs =   basefrac *np.array([K,K/4],dtype=np.float32)
    Hb = rX*(1+basefrac*cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    Hp = rX*(1+         cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    eH = CVh
    sW = np.array([[SoriE,SoriI],[SoriE,SoriI]],dtype=np.float32)
    sH = np.array([SoriF,SoriF],dtype=np.float32)
    
    sa2 = sa**2
    sW2 = sW**2
    sH2 = sH**2
    
    muHb = tau*Hb
    SigHb = (muHb*eH)**2
    muHa = tau*(Hb+(Hp-Hb)*np.exp(-0.5*sa2/sH2))
    SigHa = (muHa*eH)**2
    muHp = tau*Hp
    SigHp = (muHp*eH)**2
    
    FE,FI,ME,MI,CE,CI = base_itp_moments(res_dir)
    FL,ML,CL = opto_itp_moments(res_dir,L,CVL)
    
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
            convb,conva,convp = sparse_ring_dmft(tau,W,Ks,Hb,Hp,eH,sW,sH,sa,base_M,base_C,Twrm,Tsav,dt,Kb=Kbs)
    elif which=='opto':
        full_rb,full_ra,full_rp,full_Crb,full_Cra,full_Crp,\
            convb,conva,convp = sparse_ring_dmft(tau,W,Ks,Hb,Hp,eH,sW,sH,sa,opto_M,opto_C,Twrm,Tsav,dt,Kb=Kbs)
    elif which=='both':
        full_rb,full_ra,full_rp,full_Crb,full_Cra,full_Crp,\
            convb,conva,convp = doub_sparse_ring_dmft(tau,W,Ks,Hb,Hp,eH,sW,sH,sa,[base_M,opto_M],[base_C,opto_C],
                                                      Twrm,Tsav,dt,Kb=Kbs)
    else:
        raise NotImplementedError('Only implemented options for \'which\' keyword are: \'base\', \'opto\', and \'both\'')
        
    print('integrating first stage took',time.process_time() - start,'s')

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
    
    sr2 = np.fmax(1e-1,0.5*sa2/np.log(np.fmax(np.abs((rp-rb)/(ra-rb)),1+1e-4)))
    sr = np.sqrt(sr2)
    sCr2 = np.fmax(1e-1,0.5*sa2/np.log(np.fmax(np.abs((Crp-Crb)/(Cra-Crb)),1+1e-4)))
    sCr = np.sqrt(sCr2)
    
    if which in ('base','opto'):
        sWr2 = sW2+sr2
        srdivsWr = np.sqrt(sr2/sWr2)
        rpmb = rp-rb
        sWCr2 = sW2[:,:,None]+sCr2[None,:,:]
        sCrdivsWCr2 = np.sqrt(sCr2/sWCr2)
        Crpmb = Crp-Crb
        mub = (muW+muWb)@rb + (sr*sr2pi/S*erf(S/(sr8*sr))*muWb)@rpmb + muHb
        mua = mub + (srdivsWr*np.exp(-0.5*sa2/sWr2)*muW)@rpmb + muHa-muHb
        mup = mub + (srdivsWr*muW)@rpmb + muHp-muHb
        Sigb = each_matmul(SigW+SigWb,Crb) + each_matmul(sCr*sr2pi/S*erf(S/(sr8*sCr))*SigWb,Crpmb) + (SigHb)[:,None]
        Siga = Sigb + each_matmul(sCrdivsWCr2*np.exp(-0.5*sa2/sWCr2)*SigW,Crpmb) + (SigHa-SigHb)[:,None]
        Sigp = Sigb + each_matmul(sCrdivsWCr2*SigW,Crpmb) + (SigHp-SigHb)[:,None]
    elif which=='both':
        doub_muW = doub_mat(muW)
        doub_SigW = doub_mat(SigW)[:,:,None]
        doub_muWb = doub_mat(muWb)
        doub_SigWb = doub_mat(SigWb)[:,:,None]
        
        sWr2 = doub_mat(sW2)+sr2
        srdivsWr = np.sqrt(sr2/sWr2)
        rpmb = rp-rb
        sWCr2 = doub_mat(sW2)[:,:,None]+sCr2[None,:,:]
        sCrdivsWCr2 = np.sqrt(sCr2/sWCr2)
        Crpmb = Crp-Crb
        mub = (doub_muW+doub_muWb)@rb + (sr*sr2pi/S*erf(S/(sr8*sr))*doub_muWb)@rpmb + doub_vec(muHb)
        mua = mub + (srdivsWr*np.exp(-0.5*sa2/sWr2)*doub_muW)@rpmb + doub_vec(muHa-muHb)
        mup = mub + (srdivsWr*doub_muW)@rpmb + doub_vec(muHp-muHb)
        Sigb = each_matmul(doub_SigW+doub_SigWb,Crb) + each_matmul(sCr*sr2pi/S*erf(S/(sr8*sCr))*doub_SigWb,Crpmb) +\
            doub_vec(SigHb)[:,None]
        Siga = Sigb + each_matmul(sCrdivsWCr2*np.exp(-0.5*sa2/sWCr2)*doub_SigW,Crpmb) + doub_vec(SigHa-SigHb)[:,None]
        Sigp = Sigb + each_matmul(sCrdivsWCr2*doub_SigW,Crpmb) + doub_vec(SigHp-SigHb)[:,None]
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

def run_second_stage_ring_dmft(first_res_dict,prms,rX,cA,CVh,res_dir,rc,Twrm,Tsav,dt,sa=15,return_full=False):
    S = 180
    
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
    L = prms['L']
    CVL = prms['CVL']
    basefrac = prms.get('basefrac',0)
    
    tau = np.array([rc.tE,rc.tI],dtype=np.float32)
    W = J*np.array([[1,-gE],[1./beta,-gI/beta]],dtype=np.float32)
    Ks = (1-basefrac)*np.array([K,K/4],dtype=np.float32)
    Kbs =   basefrac *np.array([K,K/4],dtype=np.float32)
    Hb = rX*(1+basefrac*cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    Hp = rX*(1+         cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    eH = CVh
    sW = np.array([[SoriE,SoriI],[SoriE,SoriI]],dtype=np.float32)
    sH = np.array([SoriF,SoriF],dtype=np.float32)
    
    sa2 = sa**2
    sW2 = sW**2
    sH2 = sH**2
    
    muHb = tau*Hb
    SigHb = (muHb*eH)**2
    muHa = tau*(Hb+(Hp-Hb)*np.exp(-0.5*sa2/sH2))
    SigHa = (muHa*eH)**2
    muHp = tau*Hp
    SigHp = (muHp*eH)**2
    
    FE,FI,ME,MI,CE,CI = base_itp_moments(res_dir)
    FL,ML,CL = opto_itp_moments(res_dir,L,CVL)

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
    doub_SigW = doub_mat(SigW)[:,:,None]
    doub_muWb = doub_mat(muWb)
    doub_SigWb = doub_mat(SigWb)[:,:,None]
    
    sr2 = np.fmax(1e-1,0.5*sa2/np.log(np.fmax(np.abs((rp-rb)/(ra-rb)),1+1e-4)))
    sWr2 = doub_mat(sW2)+sr2
    sr = np.sqrt(sr2)
    srdivsWr = np.sqrt(sr2/sWr2)
    rpmb = rp-rb
    sCr2 = np.fmax(1e-1,0.5*sa2/np.log(np.fmax(np.abs((Crp-Crb)/(Cra-Crb)),1+1e-4)))
    sWCr2 = doub_mat(sW2)[:,:,None]+sCr2[None,:,:]
    sCr = np.sqrt(sCr2)
    sCrdivsWCr2 = np.sqrt(sCr2/sWCr2)
    Crpmb = Crp-Crb
    
    mub = (doub_muW+doub_muWb)@rb + (sr*sr2pi/S*erf(S/(sr8*sr))*doub_muWb)@rpmb + doub_vec(muHb)
    mua = mub + (srdivsWr*np.exp(-0.5*sa2/sWr2)*doub_muW)@rpmb + doub_vec(muHa-muHb)
    mup = mub + (srdivsWr*doub_muW)@rpmb + doub_vec(muHp-muHb)
    Sigb = each_matmul(doub_SigW+doub_SigWb,Crb) + each_matmul(sCr*sr2pi/S*erf(S/(sr8*sCr))*doub_SigWb,Crpmb) +\
        doub_vec(SigHb)[:,None]
    Siga = Sigb + each_matmul(sCrdivsWCr2*np.exp(-0.5*sa2/sWCr2)*doub_SigW,Crpmb) + doub_vec(SigHa-SigHb)[:,None]
    Sigp = Sigb + each_matmul(sCrdivsWCr2*doub_SigW,Crpmb) + doub_vec(SigHp-SigHb)[:,None]

    start = time.process_time()

    full_Cdrb,full_Cdra,full_Cdrp,\
        convdb,convda,convdp = diff_sparse_ring_dmft(tau,W,Ks,Hb,Hp,eH,sW,sH,sa,diff_R,Twrm,Tsav,dt,
                                                rb,ra,rp,Crb,Cra,Crp,Kb=Kbs)

    print('integrating second stage took',time.process_time() - start,'s')

    drb = rb[:2] - rb[2:]
    dra = ra[:2] - ra[2:]
    drp = rp[:2] - rp[2:]
    Cdrb = full_Cdrb[:,-1,-1:-Nsav-1:-1]
    Cdra = full_Cdra[:,-1,-1:-Nsav-1:-1]
    Cdrp = full_Cdrp[:,-1,-1:-Nsav-1:-1]

    dmub = mub[:2] - mub[2:]
    dmua = mua[:2] - mua[2:]
    dmup = mup[:2] - mup[2:]

    sCdr2 = np.fmax(1e-1,0.5*sa2/np.log(np.fmax(np.abs((Cdrp-Cdrb)/(Cdra-Cdrb)),1+1e-4)))
    sWCdr2 = sW2[:,:,None]+sCdr2[None,:,:]
    sCdr = np.sqrt(sCdr2)
    sCdrdivsWCdr2 = np.sqrt(sCdr2/sWCdr2)
    Cdrpmb = Cdrp - Cdrb
    Sigdb = each_matmul(SigW[:,:,None]+SigWb[:,:,None],Cdrb) +\
        each_matmul(sCdr*sr2pi/S*erf(S/(sr8*sCdr))*SigWb[:,:,None],Cdrpmb)
    Sigda = Sigdb + each_matmul(sCdrdivsWCdr2*np.exp(-0.5*sa2/sWCdr2)*SigW[:,:,None],Cdrpmb)
    Sigdp = Sigdb + each_matmul(sCdrdivsWCdr2*SigW[:,:,None],Cdrpmb)
    Sigdb = each_matmul(SigW[:,:,None]+(1-sCdr*sr2pi/180)*SigWb[:,:,None],Cdrb)
    
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

def run_two_stage_ring_dmft(prms,rX,cA,CVh,res_dir,rc,Twrm,Tsav,dt,sa=15,return_full=False):
    S = 180
    
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
    L = prms['L']
    CVL = prms['CVL']
    basefrac = prms.get('basefrac',0)
    
    tau = np.array([rc.tE,rc.tI],dtype=np.float32)
    W = J*np.array([[1,-gE],[1./beta,-gI/beta]],dtype=np.float32)
    Ks = (1-basefrac)*np.array([K,K/4],dtype=np.float32)
    Kbs =   basefrac *np.array([K,K/4],dtype=np.float32)
    Hb = rX*(1+basefrac*cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    Hp = rX*(1+         cA)*K*J*np.array([hE,hI/beta],dtype=np.float32)
    eH = CVh
    sW = np.array([[SoriE,SoriI],[SoriE,SoriI]],dtype=np.float32)
    sH = np.array([SoriF,SoriF],dtype=np.float32)
    
    sa2 = sa**2
    sW2 = sW**2
    sH2 = sH**2
    
    muHb = tau*Hb
    SigHb = (muHb*eH)**2
    muHa = tau*(Hb+(Hp-Hb)*np.exp(-0.5*sa2/sH2))
    SigHa = (muHa*eH)**2
    muHp = tau*Hp
    SigHp = (muHp*eH)**2
    
    FE,FI,ME,MI,CE,CI = base_itp_moments(res_dir)
    FL,ML,CL = opto_itp_moments(res_dir,L,CVL)
    
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
                                                  Twrm,Tsav,dt,Kb=Kbs)

    print('integrating first stage took',time.process_time() - start,'s')

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
    doub_SigW = doub_mat(SigW)[:,:,None]
    doub_muWb = doub_mat(muWb)
    doub_SigWb = doub_mat(SigWb)[:,:,None]
    
    sr2 = np.fmax(1e-1,0.5*sa2/np.log(np.fmax(np.abs((rp-rb)/(ra-rb)),1+1e-4)))
    sWr2 = doub_mat(sW2)+sr2
    sr = np.sqrt(sr2)
    srdivsWr = np.sqrt(sr2/sWr2)
    rpmb = rp-rb
    sCr2 = np.fmax(1e-1,0.5*sa2/np.log(np.fmax(np.abs((Crp-Crb)/(Cra-Crb)),1+1e-4)))
    sWCr2 = doub_mat(sW2)[:,:,None]+sCr2[None,:,:]
    sCr = np.sqrt(sCr2)
    sCrdivsWCr2 = np.sqrt(sCr2/sWCr2)
    Crpmb = Crp-Crb
    
    mub = (doub_muW+doub_muWb)@rb + (sr*sr2pi/S*erf(S/(sr8*sr))*doub_muWb)@rpmb + doub_vec(muHb)
    mua = mub + (srdivsWr*np.exp(-0.5*sa2/sWr2)*doub_muW)@rpmb + doub_vec(muHa-muHb)
    mup = mub + (srdivsWr*doub_muW)@rpmb + doub_vec(muHp-muHb)
    Sigb = each_matmul(doub_SigW+doub_SigWb,Crb) + each_matmul(sCr*sr2pi/S*erf(S/(sr8*sCr))*doub_SigWb,Crpmb) +\
        doub_vec(SigHb)[:,None]
    Siga = Sigb + each_matmul(sCrdivsWCr2*np.exp(-0.5*sa2/sWCr2)*doub_SigW,Crpmb) + doub_vec(SigHa-SigHb)[:,None]
    Sigp = Sigb + each_matmul(sCrdivsWCr2*doub_SigW,Crpmb) + doub_vec(SigHp-SigHb)[:,None]

    start = time.process_time()

    full_Cdrb,full_Cdra,full_Cdrp,\
        convdb,convda,convdp = diff_sparse_ring_dmft(tau,W,Ks,Hb,Hp,eH,sW,sH,sa,diff_R,Twrm,Tsav,dt,
                                                rb,ra,rp,Crb,Cra,Crp,Kb=Kbs)

    print('integrating second stage took',time.process_time() - start,'s')

    drb = rb[:2] - rb[2:]
    dra = ra[:2] - ra[2:]
    drp = rp[:2] - rp[2:]
    Cdrb = full_Cdrb[:,-1,-1:-Nsav-1:-1]
    Cdra = full_Cdra[:,-1,-1:-Nsav-1:-1]
    Cdrp = full_Cdrp[:,-1,-1:-Nsav-1:-1]

    dmub = mub[:2] - mub[2:]
    dmua = mua[:2] - mua[2:]
    dmup = mup[:2] - mup[2:]

    sCdr2 = np.fmax(1e-1,0.5*sa2/np.log(np.fmax(np.abs((Cdrp-Cdrb)/(Cdra-Cdrb)),1+1e-4)))
    sWCdr2 = sW2[:,:,None]+sCdr2[None,:,:]
    sCdr = np.sqrt(sCdr2)
    sCdrdivsWCdr2 = np.sqrt(sCdr2/sWCdr2)
    Cdrpmb = Cdrp - Cdrb
    Sigdb = each_matmul(SigW[:,:,None]+SigWb[:,:,None],Cdrb) +\
        each_matmul(sCdr*sr2pi/S*erf(S/(sr8*sCdr))*SigWb[:,:,None],Cdrpmb)
    Sigda = Sigdb + each_matmul(sCdrdivsWCdr2*np.exp(-0.5*sa2/sWCdr2)*SigW[:,:,None],Cdrpmb)
    Sigdp = Sigdb + each_matmul(sCdrdivsWCdr2*SigW[:,:,None],Cdrpmb)
    Sigdb = each_matmul(SigW[:,:,None]+(1-sCdr*sr2pi/180)*SigWb[:,:,None],Cdrb)
    
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