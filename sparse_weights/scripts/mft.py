import numpy as np
import time

import dmft
from dmft_util import *

def gauss_mft(tau,W,H,F_fn,Twrm,Tsav,dt,r0=None):
    Ntyp = len(H)
    Nint = round((Twrm+Tsav)/dt)+1
    
    r = np.zeros((Ntyp,Nint),dtype=np.float32)
    
    if r0 is None:
        r0 = 1e-8*np.ones((Ntyp),dtype=np.float32)
        
    tauinv = 1/tau
    
    r[:,0] = r0
    
    Fphi = np.empty((Ntyp,),dtype=np.float32)
    
    def drdt(ri):
        mui = W@ri + H
        F_fn(mui,Fphi)
        return tauinv*(-ri + Fphi)
    
    for i in range(Nint-1):        
        k1 = drdt(r[:,i]          )
        k2 = drdt(r[:,i]+0.5*dt*k1)
        k3 = drdt(r[:,i]+0.5*dt*k2)
        k4 = drdt(r[:,i]+    dt*k3)
        
        r[:,i+1] = r[:,i] + dt/6*(k1+2*k2+2*k3+k4)
        
        if np.any(np.abs(r[:,i+1]) > 1e10) or np.any(np.isnan(r[:,i+1])):
            print("system diverged when integrating r")
            return r,False
            
        Ndiv = 5
        if (Ndiv*(i+1)) % (Nint-1) == 0:
            print("{:.2f}% completed".format((i+1)/(Nint-1)))
    
    return r

def doub_gauss_mft(tau,muW,muH,F_fns,Twrm,Tsav,dt,rb0=None):
    Ntyp = len(muH)
    
    doub_tau = doub_vec(tau)
    doub_muW = doub_mat(muW)
    doub_muH = doub_vec(muH)
    
    def doub_F(mui,out):
        F_fns[0](mui[:Ntyp],out[:Ntyp])
        F_fns[1](mui[Ntyp:],out[Ntyp:])
        
    return gauss_mft(doub_tau,doub_muW,doub_muH,doub_F,Twrm,Tsav,dt,rb0)

def gauss_struct_mft(tau,Ws,Hs,F_fn,mu_fn,Twrm,Tsav,dt,rs0):
    Nsit = Hs.shape[0]
    Ntyp = Hs.shape[1]
    Nint = round((Twrm+Tsav)/dt)+1
    
    rs = np.zeros((Nsit,Ntyp,Nint),dtype=np.float32)
        
    tauinv = 1/tau
    
    rs[:,:,0] = rs0
        
    Fphis = np.empty((Nsit,Ntyp),dtype=np.float32)
    musi = np.empty((Nsit,Ntyp),dtype=np.float32)
    
    def drdt(rsi):
        mu_fn(rsi,Ws,Hs,musi)
        for sit_idx in range(Nsit):
            F_fn(musi[sit_idx],Fphis[sit_idx])
        return tauinv*(-rsi + Fphis)
    
    for i in range(Nint-1):
        kb1 = drdt(rs[:,:,i]           )
        kb2 = drdt(rs[:,:,i]+0.5*dt*kb1)
        kb3 = drdt(rs[:,:,i]+0.5*dt*kb2)
        kb4 = drdt(rs[:,:,i]+    dt*kb2)
        
        rs[:,:,i+1] = rs[:,:,i] + dt/6*(kb1+2*kb2+2*kb3+kb4)
        rsi = rs[:,:,i]
        mu_fn(rsi,Ws,Hs,musi)
        
        if np.any(np.abs(rs[:,:,i+1]) > 1e10) or np.any(np.isnan(rs[:,:,i+1])):
            print(musi)
            print("system diverged when integrating rb")
            return rs
            
        Ndiv = 5
        if (Ndiv*(i+1)) % (Nint-1) == 0:
            print("{:.2f}% completed".format((i+1)/(Nint-1)))
    
    return rs

def sparse_mft(tau,W,K,H,F_fn,Twrm,Tsav,dt,r0=None,mult_tau=True):
    if mult_tau:
        return gauss_mft(tau,tau[:,None]*W*K,tau*H,F_fn,Twrm,Tsav,dt,r0=r0)
    else:
        return gauss_mft(tau,W*K,H,F_fn,Twrm,Tsav,dt,r0=r0)

def doub_sparse_mft(tau,W,K,H,F_fns,Twrm,Tsav,dt,rb0=None,mult_tau=True):
    Ntyp = len(H)
    
    doub_tau = doub_vec(tau)
    doub_W = doub_mat(W)
    doub_K = doub_vec(K)
    doub_H = doub_vec(H)
    
    def doub_F(mui,out):
        F_fns[0](mui[:Ntyp],out[:Ntyp])
        F_fns[1](mui[Ntyp:],out[Ntyp:])
        
    return sparse_mft(doub_tau,doub_W,doub_K,doub_H,doub_F,Twrm,Tsav,dt,rb0,mult_tau)

def sparse_ring_mft(tau,W,K,Hb,Hp,sW,sH,sa,F_fn,Twrm,Tsav,dt,
                rb0=None,ra0=None,rp0=None,Kb=None,L=180,mult_tau=True):
    if Kb is None:
        Kb = np.zeros_like(K)
        
    Ntyp = len(Hb)
    
    if rb0 is None:
        rb0 = 1*np.ones((Ntyp),dtype=np.float32)
    if ra0 is None:
        ra0 = 2*np.ones((Ntyp),dtype=np.float32)
    if rp0 is None:
        rp0 = 5*np.ones((Ntyp),dtype=np.float32)
    
    solve_width = get_solve_width(sa,L)
    
    sW2 = sW**2
    
    if mult_tau:
        muWb = tau[:,None]*W*Kb
        muW = tau[:,None]*W*K
        muHb = tau*Hb
        muHa = tau*(Hb+(Hp-Hb)*basesubwrapnorm(sa,sH,L))
        muHp = tau*Hp
    else:
        muWb = W*Kb
        muW = W*K
        muHb = Hb
        muHa = Hb+(Hp-Hb)*basesubwrapnorm(sa,sH,L)
        muHp = Hp
    
    muWs = np.concatenate([muWb[None,:,:],muW[None,:,:]],0)
    muHs = np.concatenate([muHb[None,:],muHa[None,:],muHp[None,:]],0)
    rs0 = np.concatenate([rb0[None,:],ra0[None,:],rp0[None,:]],0)
    
    def mu_fn(rsi,muWs,muHs,musi):
        """
        rbi = rsi[0]
        rai = rsi[1]
        rpi = rsi[2]
        
        muWb = muWs[0]
        muW = muWs[1]
        
        muHb = muHs[0]
        muHa = muHs[1]
        muHp = muHs[2]
        
        mubi = musi[0]
        muai = musi[1]
        mupi = musi[2]
        """
        sri = solve_width((rsi[1]-rsi[0])/(rsi[2]-rsi[0]))
        sWri = np.sqrt(sW2+sri**2)
        rpmbi = rsi[2] - rsi[0]
        musi[0] = (muWs[1]+muWs[0])@rsi[0] + (unstruct_fact(sri,L)*muWs[0])@rpmbi + muHs[0]
        musi[1] = musi[0] + (struct_fact(sa,sWri,sri,L)*muWs[1])@rpmbi + muHs[1]-muHs[0]
        musi[2] = musi[0] + (struct_fact(0,sWri,sri,L)*muWs[1])@rpmbi + muHs[2]-muHs[0]
        musi[0] = musi[0] + (struct_fact(L/2,sWri,sri,L)*muWs[1])@rpmbi
                
    rs = gauss_struct_mft(tau,muWs,muHs,F_fn,mu_fn,Twrm,Tsav,dt,rs0)
    
    return rs[0],rs[1],rs[2]

def doub_sparse_ring_mft(tau,W,K,Hb,Hp,sW,sH,sa,F_fns,Twrm,Tsav,dt,
                     rb0=None,ra0=None,rp0=None,Kb=None,L=180,mult_tau=True):
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
    
    def doub_F(mui,out):
        F_fns[0](mui[:Ntyp],out[:Ntyp])
        F_fns[1](mui[Ntyp:],out[Ntyp:])
        
    return sparse_ring_mft(doub_tau,doub_W,doub_K,doub_Hb,doub_Hp,doub_sW,doub_sH,sa,doub_F,Twrm,Tsav,dt,
                      rb0,ra0,rp0,Kb=doub_Kb,L=L,mult_tau=mult_tau)

def sparse_2feat_ring_mft(tau,W,K,Hb,Hp,sW,sH,sa,F_fn,Twrm,Tsav,dt,
                rb0=None,ra0=None,rp0=None,Kb=None,dori=45,L=180,mult_tau=True):
    if Kb is None:
        Kb = np.zeros_like(K)
        
    Ntyp = len(Hb)
    
    if rb0 is None:
        rb0 = 1*np.ones((Ntyp),dtype=np.float32)
    if ra0 is None:
        ra0 = 2*np.ones((Ntyp),dtype=np.float32)
    if rp0 is None:
        rp0 = 5*np.ones((Ntyp),dtype=np.float32)
    
    xpeaks = np.array([0,-dori])
    solve_width = get_2feat_solve_width(sa,dori,L)
    
    sW2 = sW**2
        
    if mult_tau:
        muWb = tau[:,None]*W*Kb
        muW = tau[:,None]*W*K
        muHb = tau*Hb
        muHa = tau*(Hb+(Hp-Hb)*(basesubwrapnorm(sa,sH,L)+basesubwrapnorm(dori+sa,sH,L)))
        muHp = tau*(Hp+(Hp-Hb)*basesubwrapnorm(dori,sH,L))
    else:
        muWb = W*Kb
        muW = W*K
        muHb = Hb
        muHa = Hb+(Hp-Hb)*(basesubwrapnorm(sa,sH,L)+basesubwrapnorm(dori+sa,sH,L))
        muHp = Hp+(Hp-Hb)*basesubwrapnorm(dori,sH,L)
    
    muWs = np.concatenate([muWb[None,:,:],muW[None,:,:]],0)
    muHs = np.concatenate([muHb[None,:],muHa[None,:],muHp[None,:]],0)
    rs0 = np.concatenate([rb0[None,:],ra0[None,:],rp0[None,:]],0)
    
    def mu_fn(rsi,muWs,muHs,musi):
        """
        rbi = rsi[0]
        rai = rsi[1]
        rpi = rsi[2]
        
        muWb = muWs[0]
        muW = muWs[1]
        
        muHb = muHs[0]
        muHa = muHs[1]
        muHp = muHs[2]
        
        mubi = musi[0]
        muai = musi[1]
        mupi = musi[2]
        """
        sri = solve_width((rsi[1]-rsi[0])/(rsi[2]-rsi[0]))
        rOinv = np.sum(inv_overlap(xpeaks,sri[:,None])[:,:,0],-1)
        sWri = np.sqrt(sW2+sri**2)
        rpmbi = (rsi[2] - rsi[0])*rOinv
        musi[0] = (muWs[1]+muWs[0])@rsi[0] + (unstruct_fact(sri,L)*muWs[0])@rpmbi + muHs[0]
        musi[1] = musi[0] + ((struct_fact(sa,sWri,sri,L)+struct_fact(sa+dori,sWri,sri,L))*muWs[1])@rpmbi +\
            muHs[1]-muHs[0]
        musi[2] = musi[0] + ((struct_fact(0,sWri,sri,L)+struct_fact(dori,sWri,sri,L))*muWs[1])@rpmbi + muHs[2]-muHs[0]
        musi[0] = musi[0] + (2*struct_fact(L/2,sWri,sri,L)*muWs[1])@rpmbi
                
    rs = gauss_struct_mft(tau,muWs,muHs,F_fn,mu_fn,Twrm,Tsav,dt,rs0)
    
    return rs[0],rs[1],rs[2]

def sparse_full_ring_mft(tau,W,K,Hb,Hp,sW,sH,F_fn,Twrm,Tsav,dt,
                          rs0=None,Kb=None,L=180,Nori=20,mult_tau=True):
    if Kb is None:
        Kb = np.zeros_like(K)
        
    Ntyp = len(Hb)
    
    oris = np.arange(Nori)/Nori * L
    
    if rs0 is None:
        rs0 = (1+4*basesubwrapnorm(oris,15))[:,None]*np.ones((Ntyp),dtype=np.float32)[None,:]
    
    if mult_tau:
        muWb = tau[:,None]*W*Kb
        muW = tau[:,None]*W*K
    else:
        muWb = W*Kb
        muW = W*K
    
    doris = oris[:,None] - oris[None,:]
    kerns = wrapnormdens(doris[:,None,:,None],sW[None,:,None,:],L)
    
    muWs = (muWb[None,:,None,:] + muW[None,:,None,:]*2*np.pi*kerns) / Nori
    
    if mult_tau:
        muHs = tau*(Hb[None,:]+(Hp-Hb)[None,:]*basesubwrapnorm(oris[:,None],sH[None,:],L))
    else:
        muHs = (Hb[None,:]+(Hp-Hb)[None,:]*basesubwrapnorm(oris[:,None],sH[None,:],L))
    
    def mu_fn(rsi,muWs,muHs,musi):
        musi[:] = np.einsum('ijkl,kl->ij',muWs,rsi) + muHs
                
    return gauss_struct_mft(tau,muWs,muHs,F_fn,mu_fn,Twrm,Tsav,dt,rs0)

def run_first_stage_mft(prms,rX,res_dir,rc,Twrm,Tsav,dt,which='both',return_full=False):
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
    H = rX*K*J*np.array([hE,hI/beta],dtype=np.float32) + prms.get('pert',np.zeros(2,dtype=np.float32))
    
    if prms.get('mult_tau',True):
        muH = tau*H
    else:
        muH = H
    
    if res_dir is None:
        FE,FI = rc.FE,rc.FI
    else:
        FE,FI,_,_,_,_ = dmft.base_itp_moments(res_dir)
    
    def base_F(mui,out):
        out[0] = FE(mui[0])
        out[1] = FI(mui[1])
    
    def opto_F(mui,out):
        out[0] = FE(mui[0]+prms['L'])
        out[1] = FI(mui[1])
    
    start = time.process_time()
    
    if which=='base':
        full_r = sparse_mft(tau,W,Ks,H,base_F,Twrm,Tsav,dt,mult_tau=prms.get('mult_tau',True))
    elif which=='opto':
        full_r = sparse_mft(tau,W,Ks,H,opto_F,Twrm,Tsav,dt,mult_tau=prms.get('mult_tau',True))
    elif which=='both':
        full_r = doub_sparse_mft(tau,W,Ks,H,[base_F,opto_F],Twrm,Tsav,dt,mult_tau=prms.get('mult_tau',True))
    else:
        raise NotImplementedError('Only implemented options for \'which\' keyword are: \'base\', \'opto\', and \'both\'')
        
    print('integrating first stage took',time.process_time() - start,'s')

    # extract predicted moments after long time evolution
    r = full_r[:,-1]
    
    if prms.get('mult_tau',True):
        muW = tau[:,None]*W*Ks
    else:
        muW = W*Ks
    
    if which in ('base','opto'):
        mu = muW@r + muH
    elif which=='both':
        doub_muW = doub_mat(muW)

        mu = doub_muW@r + doub_vec(muH)
    else:
        raise NotImplementedError('Only implemented options for \'which\' keyword are: \'base\', \'opto\', and \'both\'')
    
    res_dict = {}
    
    res_dict['r'] = r
    res_dict['mu'] = mu
    
    if return_full:
        res_dict['full_r'] = full_r
    
    return res_dict

def run_first_stage_ring_mft(prms,rX,cA,res_dir,rc,Twrm,Tsav,dt,sa=15,L=180,which='both',return_full=False,
                              rb0=None,ra0=None,rp0=None):
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
    Hpert = prms.get('pert',np.zeros(2,dtype=np.float32))
    
    tau = np.array([rc.tE,rc.tI],dtype=np.float32)
    W = J*np.array([[1,-gE],[1./beta,-gI/beta]],dtype=np.float32)
    Ks =    (1-basefrac)*(1-baseprob) *np.array([K,K/4],dtype=np.float32)
    Kbs =(1-(1-basefrac)*(1-baseprob))*np.array([K,K/4],dtype=np.float32)
    Hb = rX*(1+(1-(1-basefrac)*(1-baseinp))*cA)*K*J*np.array([hE,hI/beta],dtype=np.float32) + Hpert
    Hp = rX*(1+                             cA)*K*J*np.array([hE,hI/beta],dtype=np.float32) + Hpert
    sW = np.array([[SoriE,SoriI],[SoriE,SoriI]],dtype=np.float32)
    sH = np.array([SoriF,SoriF],dtype=np.float32)
    
    sW2 = sW**2
    
    solve_width = get_solve_width(sa,L)
    
    if prms.get('mult_tau',True):
        muHb = tau*Hb
        muHa = tau*(Hb+(Hp-Hb)*basesubwrapnorm(sa,sH,L))
        muHp = tau*Hp
    else:
        muHb = Hb
        muHa = Hb+(Hp-Hb)*basesubwrapnorm(sa,sH,L)
        muHp = Hp
    
    if res_dir is None:
        FE,FI = rc.FE,rc.FI
    else:
        FE,FI,_,_,_,_ = dmft.base_itp_moments(res_dir)
    
    def base_F(mui,out):
        out[0] = FE(mui[0])
        out[1] = FI(mui[1])
    
    def opto_F(mui,out):
        out[0] = FE(mui[0]+prms['L'])
        out[1] = FI(mui[1])
    
    start = time.process_time()
    
    if which=='base':
        full_rb,full_ra,full_rp =\
             sparse_ring_mft(tau,W,Ks,Hb,Hp,sW,sH,sa,base_F,Twrm,Tsav,dt,
                                                 rb0=rb0,ra0=ra0,rp0=rp0,
                                                 Kb=Kbs,L=L,mult_tau=prms.get('mult_tau',True))
    elif which=='opto':
        full_rb,full_ra,full_rp =\
             sparse_ring_mft(tau,W,Ks,Hb,Hp,sW,sH,sa,opto_F,Twrm,Tsav,dt,
                                                 rb0=rb0,ra0=ra0,rp0=rp0,
                                                 Kb=Kbs,L=L,mult_tau=prms.get('mult_tau',True))
    elif which=='both':
        full_rb,full_ra,full_rp =\
             doub_sparse_ring_mft(tau,W,Ks,Hb,Hp,sW,sH,sa,[base_F,opto_F],
                                                      Twrm,Tsav,dt,
                                                      rb0=rb0,ra0=ra0,rp0=rp0,
                                                      Kb=Kbs,L=L,mult_tau=prms.get('mult_tau',True))
    else:
        raise NotImplementedError('Only implemented options for \'which\' keyword are: \'base\', \'opto\', and \'both\'')
        
    print('integrating first stage took',time.process_time() - start,'s')

    # extract predicted moments after long time evolution
    rb = full_rb[:,-1]
    ra = full_ra[:,-1]
    rp = full_rp[:,-1]
    
    if prms.get('mult_tau',True):
        muW = tau[:,None]*W*Ks
        muWb = tau[:,None]*W*Kbs
    else:
        muW = W*Ks
        muWb = W*Kbs
    
    sr = solve_width((ra-rb)/(rp-rb))
    
    if which in ('base','opto'):
        sWr = np.sqrt(sW2+sr**2)
        rpmb = rp-rb
        mub = (muW+muWb)@rb + (unstruct_fact(sr,L)*muWb)@rpmb + muHb
        mua = mub + (struct_fact(sa,sWr,sr,L)*muW)@rpmb + muHa-muHb
        mup = mub + (struct_fact(0,sWr,sr,L)*muW)@rpmb + muHp-muHb
        mub = mub + (struct_fact(L/2,sWr,sr,L)*muW)@rpmb
    elif which=='both':
        doub_muW = doub_mat(muW)
        doub_muWb = doub_mat(muWb)
        
        sWr = np.sqrt(doub_mat(sW2)+sr**2)
        rpmb = rp-rb
        mub = (doub_muW+doub_muWb)@rb + (unstruct_fact(sr,L)*doub_muWb)@rpmb + doub_vec(muHb)
        mua = mub + (struct_fact(sa,sWr,sr,L)*doub_muW)@rpmb + doub_vec(muHa-muHb)
        mup = mub + (struct_fact(0,sWr,sr,L)*doub_muW)@rpmb + doub_vec(muHp-muHb)
        mub = mub + (struct_fact(L/2,sWr,sr,L)*doub_muW)@rpmb
    else:
        raise NotImplementedError('Only implemented options for \'which\' keyword are: \'base\', \'opto\', and \'both\'')
    
    res_dict = {}
    
    res_dict['rb'] = rb
    res_dict['ra'] = ra
    res_dict['rp'] = rp
    res_dict['sr'] = sr
    
    res_dict['mub'] = mub
    res_dict['mua'] = mua
    res_dict['mup'] = mup
    
    if return_full:
        res_dict['full_rb'] = full_rb
        res_dict['full_ra'] = full_ra
        res_dict['full_rp'] = full_rp
    
    return res_dict