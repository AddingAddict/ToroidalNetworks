import os
import pickle
import numpy as np
from scipy.linalg import toeplitz
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import quad,simpson
import time

sr2pi = np.sqrt(2*np.pi)

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
    Msrange = ranges_dict['M']['sigrange']
    Mss = np.linspace(Msrange[0],Msrange[1],round(Msrange[2])).astype(np.float32)
    MEs = np.load(os.path.join(res_dir,'ME_itp.npy')).astype(np.float32)
    MIs = np.load(os.path.join(res_dir,'MI_itp.npy')).astype(np.float32)
    
    Cxrange = ranges_dict['C']['xrange']
    Cxs = np.linspace(Cxrange[0],Cxrange[1],round(Cxrange[2])).astype(np.float32)
    Csrange = ranges_dict['C']['sigrange']
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
    FLs = np.load(os.path.join(res_dir,'PhL_itpL={:.2f}_CVL={:.2f}.npy'.format(L,CVL))).astype(np.float32)
    
    Mxrange = ranges_dict['ML']['xrange']
    Mxs = np.linspace(Mxrange[0],Mxrange[1],round(Mxrange[2])).astype(np.float32)
    Msrange = ranges_dict['ML']['sigrange']
    Mss = np.linspace(Msrange[0],Msrange[1],round(Msrange[2])).astype(np.float32)
    MLs = np.load(os.path.join(res_dir,'ML_itpL={:.2f}_CVL={:.2f}.npy'.format(L,CVL))).astype(np.float32)
    
    Cxrange = ranges_dict['CL']['xrange']
    Cxs = np.linspace(Cxrange[0],Cxrange[1],round(Cxrange[2])).astype(np.float32)
    Csrange = ranges_dict['CL']['sigrange']
    Css = np.linspace(Csrange[0],Csrange[1],round(Csrange[2])).astype(np.float32)
    Ccrange = ranges_dict['CL']['crange']
    Ccs = np.linspace(Ccrange[0],Ccrange[1],round(Ccrange[2])).astype(np.float32)
    CLs = np.load(os.path.join(res_dir,'CL_itpL={:.2f}_CVL={:.2f}.npy'.format(L,CVL))).astype(np.float32)
    
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

sr2pi = np.sqrt(2*np.pi)

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

def each_diag(A):
    return np.einsum('ijj->ij',A)

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

def sparse_dmft(tau,W,K,Hb,Hp,εH,sW,sH,sa,M_fn,C_fn,Twrm,Tsav,dt,
                rb0=None,ra0=None,rp0=None,Crb0=None,Cra0=None,Crp0=None):
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
        rb0 = 1e-8*np.ones((Ntyp),dtype=np.float32)
    if ra0 is None:
        ra0 = 2e-8*np.ones((Ntyp),dtype=np.float32)
    if rp0 is None:
        rp0 = 5e-8*np.ones((Ntyp),dtype=np.float32)
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
        
    muW = tau[:,None]*W*K
    SigW = tau[:,None]**2*W**2*K
    
    muHb = tau*Hb
    SigHb = (muHb*εH)**2
    muHa = tau*(Hb+(Hp-Hb)*np.exp(-0.5*sa2/sH2))
    SigHa = (muHa*εH)**2
    muHp = tau*Hp
    SigHp = (muHp*εH)**2
    
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
    
    for i in range(Nint-1):
        rbi = rb[:,i]
        rai = ra[:,i]
        rpi = rp[:,i]
        sr2i = 0.5*sa2/np.log(np.fmax(np.abs((rpi-rbi)/(rai-rbi)),1+1e-4))
        sWr2i = sW2+sr2i
        Crbii = Crb[:,i,i]
        Craii = Cra[:,i,i]
        Crpii = Crp[:,i,i]
        sCr2ii = 0.5*sa2/np.log(np.fmax(np.abs((Crpii-Crbii)/(Craii-Crbii)),1+1e-4))
        sWCr2ii = sW2+sCr2ii
        mubi = muW@rbi + muHb
        muai = ((1-np.sqrt(sr2i/sWr2i)*np.exp(-0.5*sa2/sWr2i))*muW)@rbi +\
            (np.sqrt(sr2i/sWr2i)*np.exp(-0.5*sa2/sWr2i)*muW)@rpi + muHa
        mupi = ((1-np.sqrt(sr2i/sWr2i))*muW)@rbi + (np.sqrt(sr2i/sWr2i)*muW)@rpi + muHp
        Sigbii = SigW@Crbii + SigHb
        Sigaii = ((1-np.sqrt(sCr2ii/sWCr2ii)*np.exp(-0.5*sa2/sWCr2ii))*SigW)@Crbii +\
            (np.sqrt(sCr2ii/sWCr2ii)*np.exp(-0.5*sa2/sWCr2ii)*SigW)@Crpii + SigHa
        Sigpii = ((1-np.sqrt(sCr2ii/sWCr2ii))*SigW)@Crbii + (np.sqrt(sCr2ii/sWCr2ii)*SigW)@Crpii + SigHp
        M_fn(mubi,Sigbii,Mphib)
        M_fn(muai,Sigaii,Mphia)
        M_fn(mupi,Sigpii,Mphip)
        kb1 = tauinv*(-rbi + Mphib)
        ka1 = tauinv*(-rai + Mphia)
        kp1 = tauinv*(-rpi + Mphip)
        
        rbi = rb[:,i] + 0.5*dt*kb1
        rai = ra[:,i] + 0.5*dt*ka1
        rpi = rp[:,i] + 0.5*dt*kp1
        sr2i = 0.5*sa2/np.log(np.fmax(np.abs((rpi-rbi)/(rai-rbi)),1+1e-4))
        sWr2i = sW2+sr2i
        mubi = muW@rbi + muHb
        muai = ((1-np.sqrt(sr2i/sWr2i)*np.exp(-0.5*sa2/sWr2i))*muW)@rbi +\
            (np.sqrt(sr2i/sWr2i)*np.exp(-0.5*sa2/sWr2i)*muW)@rpi + muHa
        mupi = ((1-np.sqrt(sr2i/sWr2i))*muW)@rbi + (np.sqrt(sr2i/sWr2i)*muW)@rpi + muHp
        M_fn(mubi,Sigbii,Mphib)
        M_fn(muai,Sigaii,Mphia)
        M_fn(mupi,Sigpii,Mphip)
        kb2 = tauinv*(-rbi + Mphib)
        ka2 = tauinv*(-rai + Mphia)
        kp2 = tauinv*(-rpi + Mphip)
        
        rbi = rb[:,i] + 0.5*dt*kb2
        rai = ra[:,i] + 0.5*dt*ka2
        rpi = rp[:,i] + 0.5*dt*kp2
        sr2i = 0.5*sa2/np.log(np.fmax(np.abs((rpi-rbi)/(rai-rbi)),1+1e-4))
        sWr2i = sW2+sr2i
        mubi = muW@rbi + muHb
        muai = ((1-np.sqrt(sr2i/sWr2i)*np.exp(-0.5*sa2/sWr2i))*muW)@rbi +\
            (np.sqrt(sr2i/sWr2i)*np.exp(-0.5*sa2/sWr2i)*muW)@rpi + muHa
        mupi = ((1-np.sqrt(sr2i/sWr2i))*muW)@rbi + (np.sqrt(sr2i/sWr2i)*muW)@rpi + muHp
        M_fn(mubi,Sigbii,Mphib)
        M_fn(muai,Sigaii,Mphia)
        M_fn(mupi,Sigpii,Mphip)
        kb3 = tauinv*(-rbi + Mphib)
        ka3 = tauinv*(-rai + Mphia)
        kp3 = tauinv*(-rpi + Mphip)
        
        rbi = rb[:,i] + dt*kb3
        rai = ra[:,i] + dt*ka3
        rpi = rp[:,i] + dt*kp3
        sr2i = 0.5*sa2/np.log(np.fmax(np.abs((rpi-rbi)/(rai-rbi)),1+1e-4))
        sWr2i = sW2+sr2i
        mubi = muW@rbi + muHb
        muai = ((1-np.sqrt(sr2i/sWr2i)*np.exp(-0.5*sa2/sWr2i))*muW)@rbi +\
            (np.sqrt(sr2i/sWr2i)*np.exp(-0.5*sa2/sWr2i)*muW)@rpi + muHa
        mupi = ((1-np.sqrt(sr2i/sWr2i))*muW)@rbi + (np.sqrt(sr2i/sWr2i)*muW)@rpi + muHp
        M_fn(mubi,Sigbii,Mphib)
        M_fn(muai,Sigaii,Mphia)
        M_fn(mupi,Sigpii,Mphip)
        kb4 = tauinv*(-rbi + Mphib)
        ka4 = tauinv*(-rai + Mphia)
        kp4 = tauinv*(-rpi + Mphip)
        
        rb[:,i+1] = rb[:,i] + dt/6*(kb1+2*kb2+2*kb3+kb4)
        ra[:,i+1] = ra[:,i] + dt/6*(ka1+2*ka2+2*ka3+ka4)
        rp[:,i+1] = rp[:,i] + dt/6*(kp1+2*kp2+2*kp3+kp4)
        rbi = rb[:,i]
        rai = ra[:,i]
        rpi = rp[:,i]
        sr2i = 0.5*sa2/np.log(np.fmax(np.abs((rpi-rbi)/(rai-rbi)),1+1e-4))
        sWr2i = sW2+sr2i
        mubi = muW@rbi + muHb
        muai = ((1-np.sqrt(sr2i/sWr2i)*np.exp(-0.5*sa2/sWr2i))*muW)@rbi +\
            (np.sqrt(sr2i/sWr2i)*np.exp(-0.5*sa2/sWr2i)*muW)@rpi + muHa
        mupi = ((1-np.sqrt(sr2i/sWr2i))*muW)@rbi + (np.sqrt(sr2i/sWr2i)*muW)@rpi + muHp
        
        if np.any(np.abs(rb[:,i+1]) > 1e10) or np.any(np.isnan(rb[:,i+1])):
            print("system diverged when integrating rb")
            return rb,ra,rp,Crb,Cra,Crp,False,False,False
        if np.any(np.abs(ra[:,i+1]) > 1e10) or np.any(np.isnan(ra[:,i+1])):
            print("system diverged when integrating ra")
            return rb,ra,rp,Crb,Cra,Crp,False,False,False
        if np.any(np.abs(rp[:,i+1]) > 1e10) or np.any(np.isnan(rp[:,i+1])):
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
            sCr2ij = 0.5*sa2/np.log(np.fmax(np.abs((Crpij-Crbij)/(Craij-Crbij)),1+1e-4))
            sWCr2ij = sW2+sCr2ij
            Sigbij = SigW@Crbij + SigHb
            Sigaij = ((1-np.sqrt(sCr2ij/sWCr2ij)*np.exp(-0.5*sa2/sWCr2ij))*SigW)@Crbij +\
                (np.sqrt(sCr2ij/sWCr2ij)*np.exp(-0.5*sa2/sWCr2ij)*SigW)@Crpij + SigHa
            Sigpij = ((1-np.sqrt(sCr2ij/sWCr2ij))*SigW)@Crbij + (np.sqrt(sCr2ij/sWCr2ij)*SigW)@Crpij + SigHp
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
                print("system diverged when integrating Crb")
                return rb,ra,rp,Crb,Cra,Crp,False,False,False
            if np.any(np.abs(Cra[:,i+1,j+1]) > 1e10) or np.any(np.isnan(Cra[:,i+1,j+1])):
                print("system diverged when integrating Cra")
                return rb,ra,rp,Crb,Cra,Crp,False,False,False
            if np.any(np.abs(Crp[:,i+1,j+1]) > 1e10) or np.any(np.isnan(Crp[:,i+1,j+1])):
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

def doub_sparse_dmft(tau,W,K,Hb,Hp,εH,sW,sH,sa,M_fns,C_fns,Twrm,Tsav,dt,
                     rb0=None,ra0=None,rp0=None,Crb0=None,Cra0=None,Crp0=None):
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
        
    return sparse_dmft(doub_tau,doub_W,doub_K,doub_Hb,doub_Hp,εH,doub_sW,doub_sH,sa,doub_M,doub_C,Twrm,Tsav,dt,
                      rb0,ra0,rp0,Crb0,Cra0,Crp0)

def diff_sparse_dmft(tau,W,K,Hb,Hp,εH,sW,sH,sa,R_fn,Twrm,Tsav,dt,rb,ra,rp,Crb,Cra,Crp,
                     Cdrb0=None,Cdra0=None,Cdrp0=None):
    Ntyp = len(Hb)
    Nint = round((Twrm+Tsav)/dt)+1
    Nclc = round(1.5*Tsav/dt)+1
    Nsav = round(Tsav/dt)+1
    
    Cdrb = np.zeros((Ntyp,Nint,Nint),dtype=np.float32)
    Cdra = np.zeros((Ntyp,Nint,Nint),dtype=np.float32)
    Cdrp = np.zeros((Ntyp,Nint,Nint),dtype=np.float32)
    
    if Cdrb0 is None:
        Cdrb0 = (rb[Ntyp:] - rb[:Ntyp]).astype(np.float32)[:,None]**2 + 1e2
    if Cdra0 is None:
        Cdra0 = (ra[Ntyp:] - ra[:Ntyp]).astype(np.float32)[:,None]**2 + 4e2
    if Cdrp0 is None:
        Cdrp0 = (rp[Ntyp:] - rp[:Ntyp]).astype(np.float32)[:,None]**2 + 9e2
        
    tauinv = 1/tau
    dttauinv = dt/tau
    dttauinv2 = dttauinv**2
    
    sa2 = sa**2
    sW2 = sW**2
    sH2 = sH**2
        
    muW = tau[:,None]*W*K
    SigW = tau[:,None]**2*W**2*K
    
    muHb = tau*Hb
    SigHb = (muHb*εH)**2
    muHa = tau*(Hb+(Hp-Hb)*np.exp(-0.5*sa2/sH2))
    SigHa = (muHa*εH)**2
    muHp = tau*Hp
    SigHp = (muHp*εH)**2
    
    doub_muW = doub_mat(muW)
    doub_SigW = doub_mat(SigW)
    
    sr2 = 0.5*sa2/np.log(np.fmax(np.abs((rp-rb)/(ra-rb)),1+1e-4))
    sWr2 = doub_mat(sW2)+sr2
    sCr2 = 0.5*sa2/np.log(np.fmax(np.abs((Crp-Crb)/(Cra-Crb)),1+1e-4))
    sWCr2 = doub_mat(sW2)[:,:,None]+sCr2[None,:,:]
    mub = doub_muW@rb + doub_vec(muHb)
    mua = ((1-np.sqrt(sr2/sWr2)*np.exp(-0.5*sa2/sWr2))*doub_muW)@rb +\
        (np.sqrt(sr2/sWr2)*np.exp(-0.5*sa2/sWr2)*doub_muW)@rp + doub_vec(muHa)
    mup = ((1-np.sqrt(sr2/sWr2))*doub_muW)@rb + (np.sqrt(sr2/sWr2)*doub_muW)@rp + doub_vec(muHp)
    Sigb = doub_SigW@Crb + doub_vec(SigHb)[:,None]
    Siga = each_matmul((1-np.sqrt(sCr2/sWCr2)*np.exp(-0.5*sa2/sWCr2))*doub_SigW[:,:,None],Crb) +\
        each_matmul(np.sqrt(sCr2/sWCr2)*np.exp(-0.5*sa2/sWCr2)*doub_SigW[:,:,None],Crp) + doub_vec(SigHa)[:,None]
    Sigp = each_matmul((1-np.sqrt(sCr2/sWCr2))*doub_SigW[:,:,None],Crb) +\
        each_matmul(np.sqrt(sCr2/sWCr2)*doub_SigW[:,:,None],Crp) + doub_vec(SigHp)[:,None]
    
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
            sCdr2ij = 0.5*sa2/np.log(np.fmax(np.abs((Cdrpij-Cdrbij)/(Cdraij-Cdrbij)),1+1e-4))
            sWCdr2ij = sW2+sCdr2ij
            Sigdbij = SigW@Cdrbij
            Sigdaij = ((1-np.sqrt(sCdr2ij/sWCdr2ij)*np.exp(-0.5*sa2/sWCdr2ij))*SigW)@Cdrbij +\
                (np.sqrt(sCdr2ij/sWCdr2ij)*np.exp(-0.5*sa2/sWCdr2ij)*SigW)@Cdrpij
            Sigdpij = ((1-np.sqrt(sCdr2ij/sWCdr2ij))*SigW)@Cdrbij + (np.sqrt(sCdr2ij/sWCdr2ij)*SigW)@Cdrpij
            
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
                print("system diverged when integrating Cdrb")
                return Cdrb,Cdra,Cdrp,False,False,False
            if np.any(np.abs(Cdra[:,i+1,j+1]) > 1e10) or np.any(np.isnan(Cdra[:,i+1,j+1])):
                print("system diverged when integrating Cdra")
                return Cdrb,Cdra,Cdrp,False,False,False
            if np.any(np.abs(Cdrp[:,i+1,j+1]) > 1e10) or np.any(np.isnan(Cdrp[:,i+1,j+1])):
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

def run_dmft(rc,prms,CVh,res_dir,Twrm,Tsav,dt,sa=15):
    Nsav = round(Tsav/dt)+1