import argparse
import os
try:
    import pickle5 as pickle
except:
    import pickle
import numpy as np
import time

from ssn import SSN
import dmft

parser = argparse.ArgumentParser()

parser.add_argument('--c_idx', '-c',  help='which contrast', type=int, default=0)
parser.add_argument('--b_idx', '-b',  help='which g1line', type=int, default=0)
parser.add_argument('--K', '-K',  help='number of connections', type=int, default=250)
args = vars(parser.parse_args())
print(parser.parse_args())
c_idx= args['c_idx']
b_idx= args['b_idx']
K= args['K']

NE = 200
NI = 50
Nori = 24
Sori = 32
p0 = K / NE * 2*np.pi / Nori / (np.sqrt(2*np.pi)*(Sori*2*np.pi/180))
J = np.array([
    [2.5, -1.3],
    [2.4, -1.0],
]) * np.pi / 24 / p0 / np.array([[NE,NI]])
Sstim = 30

prms = {
    'K': K,
    'SoriE': Sori,
    'SoriI': Sori,
    'SoriF': Sstim,
    'J': J[0,0],
    'beta': J[0,0]/J[1,0],
    'gE': -J[0,1]/J[0,0],
    'gI': -J[1,1]/J[1,0],
    'hE': 100/J[0,0]/K,
    'hI': 100/J[1,0]/K,
    'L': 0,
    'CVL': 1,
    'Nori': Nori,
    'NE': NE,
    'NI': NI,
    'mult_tau': False,
}

ri = SSN()

# Twrm = 0.4
# Tsav = 0.1
# Tsim = 1.0
# dt = 0.01/15
Twrm = 1.2
Tsav = 0.4
Tsim = 1.0
dt = 0.01/5

print('simulating baseline # '+str(b_idx+1)+' contrast # '+str(c_idx+1))
print('')
base = np.array([10,30,50,70])[b_idx]
con = 0.7*np.array([0,20,50,100])[c_idx]

cA = con / base
rX = base / 100

μrEs = np.zeros((2,Nori))
μrIs = np.zeros((2,Nori))
ΣrEs = np.zeros((2,Nori))
ΣrIs = np.zeros((2,Nori))
μhEs = np.zeros((2,Nori))
μhIs = np.zeros((2,Nori))
ΣhEs = np.zeros((2,Nori))
ΣhIs = np.zeros((2,Nori))
balEs = np.zeros((2,Nori))
balIs = np.zeros((2,Nori))
normCEs = np.zeros((2,Nori))
normCIs = np.zeros((2,Nori))
convs = np.zeros((2,2)).astype(bool)

def predict_networks(prms,rX,cA):
    tau = np.array([ri.tE,ri.tI],dtype=np.float32)
    W = prms['J']*np.array([[1,-prms['gE']],[1./prms['beta'],-prms['gI']/prms['beta']]],dtype=np.float32)
    Ks = (1-prms.get('basefrac',0))*np.array([prms['K'],prms['K']/4],dtype=np.float32)
    Kbs =   prms.get('basefrac',0) *np.array([prms['K'],prms['K']/4],dtype=np.float32)
    Hb = rX*(1+prms.get('basefrac',0)*cA)*prms['K']*prms['J']*\
        np.array([prms['hE'],prms['hI']/prms['beta']],dtype=np.float32)
    Hp = rX*(1+                       cA)*prms['K']*prms['J']*\
        np.array([prms['hE'],prms['hI']/prms['beta']],dtype=np.float32)
    eH = 0
    sW = np.array([[prms['SoriE'],prms['SoriI']],[prms['SoriE'],prms['SoriI']]],dtype=np.float32)
    sH = np.array([prms['SoriF'],prms['SoriF']],dtype=np.float32)
    
    if prms.get('mult_tau',True):
        muW = tau[:,None]*W*Ks
        SigW = tau[:,None]**2*W**2*Ks
        muWb = tau[:,None]*W*Kbs
        SigWb = tau[:,None]**2*W**2*Kbs
        muHb = tau*Hb
        muHp = tau*Hp
    else:
        muW = W*Ks
        SigW = W**2*Ks
        muWb = W*Kbs
        SigWb = W**2*Kbs
        muHb = Hb
        muHp = Hp
    
    sW2 = sW**2
    sH2 = sH**2
    
    smuH = sH
    SigHb = (muHb*eH)**2
    SigHp = (muHp*eH)**2
    sSigH = 2*sH

    μrs = np.zeros((2,2,Nori))
    Σrs = np.zeros((2,2,Nori))
    μmuEs = np.zeros((2,2,Nori))
    ΣmuEs = np.zeros((2,2,Nori))
    μmuIs = np.zeros((2,2,Nori))
    ΣmuIs = np.zeros((2,2,Nori))
    normC = np.zeros((2,2,Nori))
    conv = np.zeros((2,2)).astype(bool)
    
    oris = np.arange(Nori)*180/Nori
    oris[oris > 90] = 180 - oris[oris > 90]
    
    dori = 90
    xpeaks = np.array([0,-dori])
        
    def gauss(x,b,p,s):
        return b + (p-b)*dmft.basesubwrapnorm(x,s)
        
    def gauss2_naive(x,b,p,s):
        amp = (p-b)
        return b + amp*dmft.basesubwrapnorm(x,s) + amp*dmft.basesubwrapnorm(x-dori,s)
    
    def gauss2(x,b,p,s):
        if np.isscalar(s):
            amp = (p-b)*np.sum(dmft.inv_overlap(xpeaks,s*np.ones((1))[:,None])[:,:,0],-1)
        else:
            amp = (p-b)*np.sum(dmft.inv_overlap(xpeaks,s[:,None])[:,:,0],-1)
        return b + amp*dmft.basesubwrapnorm(x,s) + amp*dmft.basesubwrapnorm(x-dori,s)
    
    if cA == 0 or prms.get('basefrac',0)==1:
        res_dict = dmft.run_first_stage_dmft(prms,rX*(1+cA),0,None,ri,Twrm,Tsav,dt,which='base')
        r1b,r2b = res_dict['r'],res_dict['r']
        r1p,r2p = res_dict['r'],res_dict['r']
        sr1,sr2 = 1e2*np.ones(2),1e2*np.ones(2)
        Cr1b = dmft.grid_stat(np.mean,res_dict['Cr'],Tsim,dt)
        Cr2b = Cr1b
        Cr1p = dmft.grid_stat(np.mean,res_dict['Cr'],Tsim,dt)
        Cr2p = Cr1p
        sCr1,sCr2 = 1e2*np.ones(2),1e2*np.ones(2)
        normC[:,0] = (res_dict['Cr'][:,-1]/res_dict['Cr'][:,0])[:,None]
        normC[:,1] = normC[:,0]
        conv[:,0],conv[:,1] = res_dict['conv'],res_dict['conv']
        dmft_res1,dmft_res2 = res_dict.copy(),res_dict.copy()
    else:
        res_dict = dmft.run_first_stage_ring_dmft(prms,rX,cA,0,None,ri,Twrm,Tsav,dt,which='base',sa=15)
        r1b = res_dict['rb']
        r1p = res_dict['rp']
        sr1 = res_dict['sr']
        Cr1b = dmft.grid_stat(np.mean,res_dict['Crb'],Tsim,dt)
        Cr1p = dmft.grid_stat(np.mean,res_dict['Crp'],Tsim,dt)
        sCr1 = dmft.grid_stat(np.mean,res_dict['sCr'],Tsim,dt)
        normC[:,0] = gauss(oris[None,:],res_dict['Crb'][:,-1,None],res_dict['Crp'][:,-1,None],
                           res_dict['sCr'][:,-1,None]) /\
                     gauss(oris[None,:],res_dict['Crb'][:, 0,None],res_dict['Crp'][:, 0,None],
                           res_dict['sCr'][:, 0,None])
        conv[:,0] = res_dict['convp']
        dmft_res1 = res_dict.copy()
        
        res_dict = dmft.run_first_stage_2feat_ring_dmft(prms,rX,cA,0,None,ri,Twrm,Tsav,dt,which='base',sa=15,dori=90)
        r2b = res_dict['rb']
        r2p = res_dict['rp']
        sr2 = res_dict['sr']
        Cr2b = dmft.grid_stat(np.mean,res_dict['Crb'],Tsim,dt)
        Cr2p = dmft.grid_stat(np.mean,res_dict['Crp'],Tsim,dt)
        sCr2 = dmft.grid_stat(np.mean,res_dict['sCr'],Tsim,dt)
        normC[:,1] = gauss(oris[None,:],res_dict['Crb'][:,-1,None],res_dict['Crp'][:,-1,None],
                           res_dict['sCr'][:,-1,None]) /\
                     gauss(oris[None,:],res_dict['Crb'][:, 0,None],res_dict['Crp'][:, 0,None],
                           res_dict['sCr'][:, 0,None])
        conv[:,1] = res_dict['convp']
        dmft_res2 = res_dict.copy()
        
    sWr1 = np.sqrt(sW2+sr1**2)
    sWCr1 = np.sqrt(sW2+sCr1**2)
    
    mu1b = (muW+dmft.unstruct_fact(sr1)*muWb)*r1b
    mu1p = mu1b + dmft.struct_fact(0,sWr1,sr1)*muW*(r1p-r1b)
    mu1b = mu1b + dmft.struct_fact(90,sWr1,sr1)*muW*(r1p-r1b)
    smu1 = sWr1
    
    Sig1b = (SigW+dmft.unstruct_fact(sCr1)*SigWb)*Cr1b
    Sig1p = Sig1b + dmft.struct_fact(0,sWCr1,sCr1)*SigW*(Cr1p-Cr1b)
    Sig1b = Sig1b + dmft.struct_fact(90,sWCr1,sCr1)*SigW*(Cr1p-Cr1b)
    sSig1 = sWCr1
        
    sWr2 = np.sqrt(sW2+sr2**2)
    sWCr2 = np.sqrt(sW2+sCr2**2)
    
    mu2b = (muW+dmft.unstruct_fact(sr2)*muWb)*r2b
    mu2p = mu2b + (dmft.struct_fact(0,sWr2,sr2)+dmft.struct_fact(dori,sWr2,sr2))*muW*(r2p-r2b)
    mu2b = mu2b + 2*dmft.struct_fact(90,sWr2,sr2)*muW*(r2p-r2b)
    smu2 = sWr2
    
    Sig2b = (SigW+dmft.unstruct_fact(sCr2)*SigWb)*Cr2b
    Sig2p = Sig2b + (dmft.struct_fact(0,sWCr2,sCr2)+dmft.struct_fact(dori,sWCr2,sCr2))*SigW*(Cr2p-Cr2b)
    Sig2b = Sig2b + 2*dmft.struct_fact(90,sWCr2,sCr2)*SigW*(Cr2p-Cr2b)
    sSig2 = sWCr2
    
    for i in range(2):
        μrs[i,0] = gauss(oris,r1b[i],r1p[i],sr1[i])
        μrs[i,1] = gauss2(oris,r2b[i],r2p[i],sr2[i])
        Σrs[i,0] = np.fmax(gauss(oris,Cr1b[i],Cr1p[i],sCr1[i]) - μrs[i,0]**2,0)
        Σrs[i,1] = np.fmax(gauss2(oris,Cr2b[i],Cr2p[i],sCr2[i]) - μrs[i,1]**2,0)
        μmuEs[i,0] = gauss(oris,mu1b[i,0],mu1p[i,0],smu1[i,0]) + gauss(oris,muHb[i],muHp[i],smuH[i])
        μmuEs[i,1] = gauss2(oris,mu2b[i,0],mu2p[i,0],smu2[i,0]) + gauss2_naive(oris,muHb[i],muHp[i],smuH[i])
        ΣmuEs[i,0] = gauss(oris,Sig1b[i,0],Sig1p[i,0],sSig1[i,0]) + (gauss(oris,muHb[i],muHp[i],smuH[i])*eH)**2
        ΣmuEs[i,1] = gauss2(oris,Sig2b[i,0],Sig2p[i,0],sSig2[i,0]) + (gauss2_naive(oris,muHb[i],muHp[i],smuH[i])*eH)**2
        μmuIs[i,0] = gauss(oris,mu1b[i,1],mu1p[i,1],smu1[i,1])
        μmuIs[i,1] = gauss2(oris,mu2b[i,1],mu2p[i,1],smu2[i,1])
        ΣmuIs[i,0] = gauss(oris,Sig1b[i,1],Sig1p[i,1],sSig1[i,1])
        ΣmuIs[i,1] = gauss2(oris,Sig2b[i,1],Sig2p[i,1],sSig2[i,1])
    μmus = μmuEs + μmuIs
    Σmus = ΣmuEs + ΣmuIs

    return μrs,Σrs,μmus,Σmus,μmuEs,ΣmuEs,μmuIs,ΣmuIs,normC,conv,dmft_res1,dmft_res2

def calc_bal(μmuE,μmuI,ΣmuE,ΣmuI,N=20000,seed=0):
    rng = np.random.default_rng(seed)
    muEs = np.fmax(μmuE + np.sqrt(ΣmuE)*rng.normal(size=N),1e-12)
    muIs = np.fmin(μmuI + np.sqrt(ΣmuI)*rng.normal(size=N),-1e-12)
    return np.mean(np.abs(muEs+muIs)/muEs)

def calc_opto_bal(μmuE,μmuI,ΣmuE,ΣmuI,L,CVL,N=20000,seed=0):
    sigma_l = np.sqrt(np.log(1+CVL**2))
    mu_l = np.log(1e-3*L)-sigma_l**2/2
    rng = np.random.default_rng(seed)
    muEs = np.fmax(μmuE + np.sqrt(ΣmuE)*rng.normal(size=N) +\
        rng.lognormal(mu_l, sigma_l, N),1e-12)
    muIs = np.fmin(μmuI + np.sqrt(ΣmuI)*rng.normal(size=N),-1e-12)
    return np.mean(np.abs(muEs+muIs)/muEs)

# Simulate zero and full contrast networks with ring connectivity
print('simulating baseline fraction network')
print('')

μrs,Σrs,μmus,Σmus,μmuEs,ΣmuEs,μmuIs,ΣmuIs,normC,conv,dmft_res1,dmft_res2 = predict_networks(prms,rX,cA)

start = time.process_time()

μrEs[:] = μrs[0]
μrIs[:] = μrs[1]
ΣrEs[:] = Σrs[0]
ΣrIs[:] = Σrs[1]
μhEs[:] = μmus[0]
μhIs[:] = μmus[1]
ΣhEs[:] = Σmus[0]
ΣhIs[:] = Σmus[1]
for nloc in range(Nori):
    balEs[0,nloc] = calc_bal(μmuEs[0,0,nloc],μmuIs[0,0,nloc],ΣmuEs[0,0,nloc],ΣmuIs[0,0,nloc])
    balIs[0,nloc] = calc_bal(μmuEs[1,0,nloc],μmuIs[1,0,nloc],ΣmuEs[1,0,nloc],ΣmuIs[1,0,nloc])
    balEs[1,nloc] = calc_bal(μmuEs[0,1,nloc],μmuIs[0,1,nloc],ΣmuEs[0,1,nloc],ΣmuIs[0,1,nloc])
    balIs[1,nloc] = calc_bal(μmuEs[1,1,nloc],μmuIs[1,1,nloc],ΣmuEs[1,1,nloc],ΣmuIs[1,1,nloc])
normCEs[:] = normC[0]
normCIs[:] = normC[1]
convs[:] = conv

oris = np.arange(Nori)*180/Nori
oris[oris > 90] = 180 - oris[oris > 90]
vsm_mask = np.abs(oris) < 4.5
oris = np.abs(np.arange(Nori)*180/Nori - 90)
oris[oris > 90] = 180 - oris[oris > 90]
osm_mask = np.abs(oris) < 4.5

all_g1_means = 0.8*np.mean(μrEs[0]) + 0.2*np.mean(μrIs[0])
all_g1_stds = np.sqrt(0.8*np.mean(ΣrEs[0]+μrEs[0]**2) + 0.2*np.mean(ΣrIs[0]+μrIs[0]**2) - all_g1_means**2)
all_g2_means = 0.8*np.mean(μrEs[1]) + 0.2*np.mean(μrIs[1])
all_g2_stds = np.sqrt(0.8*np.mean(ΣrEs[1]+μrEs[1]**2) + 0.2*np.mean(ΣrIs[1]+μrIs[1]**2) - all_g2_means**2)

vsm_g1_means = 0.8*np.mean(μrEs[0,vsm_mask]) + 0.2*np.mean(μrIs[0,vsm_mask])
vsm_g1_stds = np.sqrt(0.8*np.mean(ΣrEs[0,vsm_mask]+μrEs[0,vsm_mask]**2) +\
    0.2*np.mean(ΣrIs[0,vsm_mask]+μrIs[0,vsm_mask]**2) - vsm_g1_means**2)
vsm_g2_means = 0.8*np.mean(μrEs[1,vsm_mask]) + 0.2*np.mean(μrIs[1,vsm_mask])
vsm_g2_stds = np.sqrt(0.8*np.mean(ΣrEs[1,vsm_mask]+μrEs[1,vsm_mask]**2) +\
    0.2*np.mean(ΣrIs[1,vsm_mask]+μrIs[1,vsm_mask]**2) - vsm_g2_means**2)

osm_g1_means = 0.8*np.mean(μrEs[0,osm_mask]) + 0.2*np.mean(μrIs[0,osm_mask])
osm_g1_stds = np.sqrt(0.8*np.mean(ΣrEs[0,osm_mask]+μrEs[0,osm_mask]**2) +\
    0.2*np.mean(ΣrIs[0,osm_mask]+μrIs[0,osm_mask]**2) - osm_g1_means**2)
osm_g2_means = 0.8*np.mean(μrEs[1,osm_mask]) + 0.2*np.mean(μrIs[1,osm_mask])
osm_g2_stds = np.sqrt(0.8*np.mean(ΣrEs[1,osm_mask]+μrEs[1,osm_mask]**2) +\
    0.2*np.mean(ΣrIs[1,osm_mask]+μrIs[1,osm_mask]**2) - osm_g2_means**2)

print("Saving statistics took ",time.process_time() - start," s")
print('')

res_dict = {}

res_dict['prms'] = prms
res_dict['μrEs'] = μrEs
res_dict['μrIs'] = μrIs
res_dict['ΣrEs'] = ΣrEs
res_dict['ΣrIs'] = ΣrIs
res_dict['μhEs'] = μhEs
res_dict['μhIs'] = μhIs
res_dict['ΣhEs'] = ΣhEs
res_dict['ΣhIs'] = ΣhIs
res_dict['balEs'] = balEs
res_dict['balIs'] = balIs
res_dict['normCEs'] = normCEs
res_dict['normCIs'] = normCIs
res_dict['convs'] = convs
res_dict['all_g1_means'] = all_g1_means
res_dict['all_g1_stds'] = all_g1_stds
res_dict['all_g2_means'] = all_g2_means
res_dict['all_g2_stds'] = all_g2_stds
res_dict['vsm_g1_means'] = vsm_g1_means
res_dict['vsm_g1_stds'] = vsm_g1_stds
res_dict['vsm_g2_means'] = vsm_g2_means
res_dict['vsm_g2_stds'] = vsm_g2_stds
res_dict['osm_g1_means'] = osm_g1_means
res_dict['osm_g1_stds'] = osm_g1_stds
res_dict['osm_g2_means'] = osm_g2_means
res_dict['osm_g2_stds'] = osm_g2_stds
res_dict['dmft_res1'] = dmft_res1
res_dict['dmft_res2'] = dmft_res2

res_file = './../results/dmft_ssn'
res_file = res_file + '_c_{:d}_b_{:d}_K_{:d}'.format(c_idx,b_idx,K)

with open(res_file+'.pkl', 'wb') as handle:
    pickle.dump(res_dict,handle)
