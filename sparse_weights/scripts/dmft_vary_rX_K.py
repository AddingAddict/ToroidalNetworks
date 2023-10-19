import argparse
import os
try:
    import pickle5 as pickle
except:
    import pickle
import numpy as np
import time

import ricciardi as ric
import dmft

parser = argparse.ArgumentParser()

parser.add_argument('--rX_idx', '-r',  help='which rX', type=int, default=0)
parser.add_argument('--K_idx', '-k',  help='which K', type=int, default=0)
args = vars(parser.parse_args())
print(parser.parse_args())
rX_idx= args['rX_idx']
K_idx= args['K_idx']

id = (133, 0)
with open('./../results/results_ring_'+str(id[0])+'.pkl', 'rb') as handle:
    res_dict = pickle.load(handle)[id[1]]
    prms = res_dict['prms']
    CVh = res_dict['best_monk_eX']
    bX = res_dict['best_monk_bX']
    aXs = res_dict['best_monk_aXs']
    # K = prms['K']
    # SoriE = prms['SoriE']
    # SoriI = prms['SoriI']
    # SoriF = prms['SoriF']
    # J = prms['J']
    # beta = prms['beta']
    # gE = prms['gE']
    # gI = prms['gI']
    # hE = prms['hE']
    # hI = prms['hI']
    # L = prms['L']
    # CVL = prms['CVL']

ri = ric.Ricciardi()

Twrm = 4.0
Tsav = 1.0
Tsim = 4.0
dt = 0.01/4

Nori = 20

cAs = aXs[[0,-1]]/bX
rXs = bX*10**np.arange(-1.4,0.4+0.1,0.2)
Ks = np.round(500*10**np.arange(-1.4,0.4+0.1,0.2)).astype(np.int32)

print('predicting rX # '+str(rX_idx+1))
print('')
rX = rXs[rX_idx]

print('predicting K # '+str(K_idx+1))
print('')
K = Ks[K_idx]

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

def predict_networks(prms,rX,cA,CVh):
    print(rX)
    tau = np.array([ri.tE,ri.tI],dtype=np.float32)
    W = prms['J']*np.array([[1,-prms['gE']],[1./prms['beta'],-prms['gI']/prms['beta']]],dtype=np.float32)
    Ks = np.array([prms['K'],prms['K']/4],dtype=np.float32)
    Hb = rX*prms['K']*prms['J']*np.array([prms['hE'],prms['hI']/prms['beta']],dtype=np.float32)
    Hp = rX*(1+cA)*prms['K']*prms['J']*np.array([prms['hE'],prms['hI']/prms['beta']],dtype=np.float32)
    eH = CVh
    sW = np.array([[prms['SoriE'],prms['SoriI']],[prms['SoriE'],prms['SoriI']]],dtype=np.float32)
    sH = np.array([prms['SoriF'],prms['SoriF']],dtype=np.float32)
    
    muW = tau[:,None]*W*Ks
    SigW = tau[:,None]**2*W**2*Ks
    
    sW2 = sW**2
    sH2 = sH**2
    
    muHb = tau*Hb
    muHp = tau*Hp
    smuH2 = sH2
    SigHb = (muHb*eH)**2
    SigHp = (muHp*eH)**2
    sSigH2 = sH2

    μrs = np.zeros((2,Nori))
    Σrs = np.zeros((2,Nori))
    μmuEs = np.zeros((2,Nori))
    ΣmuEs = np.zeros((2,Nori))
    μmuIs = np.zeros((2,Nori))
    ΣmuIs = np.zeros((2,Nori))
    normC = np.zeros((2,Nori))
    conv = np.zeros((2)).astype(bool)
    
    oris = np.arange(Nori)*180/Nori
    oris[oris > 90] = 180 - oris[oris > 90]
        
    def gauss(x,b,p,s2):
        return b + (p-b)*np.exp(-0.5*x**2/s2)
    
    if cA == 0:
        res_dict = dmft.run_first_stage_dmft(prms,rX,CVh,'./../results',ri,Twrm,Tsav,dt,which='base')
        rb = res_dict['r'][:2]
        rp = res_dict['r'][:2]
        sr2 = 1e4*np.ones(2)
        Crb = dmft.grid_stat(np.mean,res_dict['Cr'][:2],Tsim,dt)
        Crp = dmft.grid_stat(np.mean,res_dict['Cr'][:2],Tsim,dt)
        sCr2 = 1e4*np.ones(2)
        normC[:,:] = (res_dict['Cr'][:2,-1]/res_dict['Cr'][:2,0])[:,None]
        conv[:] = res_dict['conv'][:2]
    else:
        res_dict = dmft.run_first_stage_ring_dmft(prms,rX,cA,CVh,'./../results',ri,Twrm,Tsav,dt,which='base')
        rb = res_dict['rb'][:2]
        rp = res_dict['rp'][:2]
        sr2 = res_dict['sr'][:2]**2
        Crb = dmft.grid_stat(np.mean,res_dict['Crb'][:2],Tsim,dt)
        Crp = dmft.grid_stat(np.mean,res_dict['Crp'][:2],Tsim,dt)
        sCr2 = dmft.grid_stat(np.mean,res_dict['sCr'][:2],Tsim,dt)**2
        normC[:,:] = gauss(oris[None,:],res_dict['Crb'][:2,-1,None],res_dict['Crp'][:2,-1,None],
                           res_dict['sCr'][:2,-1,None]) /\
                     gauss(oris[None,:],res_dict['Crb'][:2, 0,None],res_dict['Crp'][:2, 0,None],
                           res_dict['sCr'][:2, 0,None])
        conv[:] = res_dict['convp'][:2]
        
    sWr2 = sW2+sr2
    sWCr2 = sW2+sCr2
    
    mub = muW*rb
    mup = mub + np.sqrt(sr2/sWr2)*muW*(rp-rb)
    smu2 = sWr2
    
    Sigb = SigW*Crb
    Sigp = Sigb + np.sqrt(sCr2/sWCr2)*SigW*(Crp-Crb)
    sSig2 = sWCr2
    
    for i in range(2):
        μrs[i] = gauss(oris,rb[i],rp[i],sr2[i])
        Σrs[i] = np.fmax(gauss(oris,Crb[i],Crp[i],sCr2[i]) - gauss(oris,rb[i],rp[i],sr2[i])**2,0)
        μmuEs[i] = gauss(oris,mub[0,i],mup[0,i],smu2[0,i]) + gauss(oris,muHb[i],muHp[i],smuH2[i])
        ΣmuEs[i] = gauss(oris,Sigb[0,i],Sigp[0,i],sSig2[0,i]) + gauss(oris,SigHb[i],SigHp[i],sSigH2[i])
        μmuIs[i] = gauss(oris,mub[1,i],mup[1,i],smu2[1,i])
        ΣmuIs[i] = gauss(oris,Sigb[1,i],Sigp[1,i],sSig2[1,i])
    μmus = μmuEs + μmuIs
    Σmus = ΣmuEs + ΣmuIs

    return μrs,Σrs,μmus,Σmus,μmuEs,ΣmuEs,μmuIs,ΣmuIs,normC,conv

def calc_bal(μmuE,μmuI,ΣmuE,ΣmuI,N=10000):
    muEs = np.fmax(μmuE + np.sqrt(ΣmuE)*np.random.randn(N),1e-12)
    muIs = np.fmin(μmuI + np.sqrt(ΣmuI)*np.random.randn(N),-1e-12)
    return np.mean(np.abs(muEs+muIs)/muEs)

# Simulate zero and full contrast networks with ring connectivity
for cA_idx,cA in enumerate(cAs):
    print('predicting contrast # '+str(cA_idx+1))
    print('')
    K_prms = prms.copy()
    K_prms['K'] = K
    K_prms['J'] = prms['J'] / np.sqrt(K/500)

    μrs,Σrs,μmus,Σmus,μmuEs,ΣmuEs,μmuIs,ΣmuIs,normC,conv = predict_networks(K_prms,rX,cA,CVh)

    start = time.process_time()

    μrEs[cA_idx] = μrs[0]
    μrIs[cA_idx] = μrs[1]
    ΣrEs[cA_idx] = Σrs[0]
    ΣrIs[cA_idx] = Σrs[1]
    μhEs[cA_idx] = μmus[0]
    μhIs[cA_idx] = μmus[1]
    ΣhEs[cA_idx] = Σmus[0]
    ΣhIs[cA_idx] = Σmus[1]
    for nloc in range(Nori):
        balEs[cA_idx,nloc] = calc_bal(μmuEs[0,nloc],μmuIs[0,nloc],ΣmuEs[0,nloc],ΣmuIs[0,nloc])
        balIs[cA_idx,nloc] = calc_bal(μmuEs[1,nloc],μmuIs[1,nloc],ΣmuEs[1,nloc],ΣmuIs[1,nloc])
    normCEs[cA_idx] = normC[0]
    normCIs[cA_idx] = normC[1]
    convs[cA_idx] = conv

    print("Saving statistics took ",time.process_time() - start," s")
    print('')

res_dict = {}

prms['CVh'] = CVh
res_dict['prms'] = prms

res_dict['cAs'] = cAs
res_dict['rX'] = rX
res_dict['K'] = K

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

with open('./../results/dmft_vary_rX_{:d}_K_{:d}'.format(rX_idx,K_idx)+'.pkl', 'wb') as handle:
    pickle.dump(res_dict,handle)
