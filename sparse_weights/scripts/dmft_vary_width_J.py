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

parser.add_argument('--width_idx', '-w',  help='which width', type=int, default=0)
parser.add_argument('--J_idx', '-j',  help='which J', type=int, default=0)
parser.add_argument('--sa_mult', '-s',  help='multiplier for auxiliary site location', type=float, default=1.0)
args = vars(parser.parse_args())
print(parser.parse_args())
width_idx= args['width_idx']
J_idx= args['J_idx']
sa_mult= args['sa_mult']

id = None#(133, 0)
if id is None:
    with open('./../results/best_fit.pkl', 'rb') as handle:
        res_dict = pickle.load(handle)
elif len(id)==1:
    with open('./../results/refit_candidate_prms_{:d}.pkl'.format(
            id[0]), 'rb') as handle:
        res_dict = pickle.load(handle)
elif len(id)==2:
    with open('./../results/results_ring_{:d}.pkl'.format(
            id[0]), 'rb') as handle:
        res_dict = pickle.load(handle)[id[-1]]
else:
    with open('./../results/results_ring_perturb_njob-{:d}_nrep-{:d}_ntry-{:d}.pkl'.format(
            id[0],id[1],id[2]), 'rb') as handle:
        res_dict = pickle.load(handle)[id[-1]]
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

Twrm = 1.2
Tsav = 0.4
Tsim = 1.0
if J_idx < 4:
    dt = 0.01/5
elif J_idx < 6:
    dt = 0.01/8
else:
    dt = 0.01/10

Nori = [80,50,40,20,16,10, 8][width_idx]

widths = 4**(2*np.arange(0,6+1)/6 - 1)
Js = prms['J']*5**(np.arange(0,6+1)/6-0.5)

print('simulating width # '+str(width_idx+1))
print('')
width = widths[width_idx]

print('simulating J # '+str(J_idx+1))
print('')
newJ = Js[J_idx]

cA = aXs[-1]/bX
rX = bX

μrEs = np.zeros((3,Nori))
μrIs = np.zeros((3,Nori))
ΣrEs = np.zeros((4,Nori))
ΣrIs = np.zeros((4,Nori))
μhEs = np.zeros((3,Nori))
μhIs = np.zeros((3,Nori))
ΣhEs = np.zeros((4,Nori))
ΣhIs = np.zeros((4,Nori))
balEs = np.zeros((2,Nori))
balIs = np.zeros((2,Nori))
normCEs = np.zeros((3,Nori))
normCIs = np.zeros((3,Nori))
convs = np.zeros((2,3)).astype(bool)

def predict_networks(prms,rX,cA,CVh):
    tau = np.array([ri.tE,ri.tI],dtype=np.float32)
    W = prms['J']*np.array([[1,-prms['gE']],[1./prms['beta'],-prms['gI']/prms['beta']]],dtype=np.float32)
    Ks = (1-prms.get('basefrac',0))*np.array([prms['K'],prms['K']/4],dtype=np.float32)
    Kbs =   prms.get('basefrac',0) *np.array([prms['K'],prms['K']/4],dtype=np.float32)
    Hb = rX*(1+(1-(1-prms.get('basefrac',0))*(1-prms.get('baseinp',0)))*cA)*prms['K']*prms['J']*\
        np.array([prms['hE'],prms['hI']/prms['beta']],dtype=np.float32)
    Hp = rX*(1+                                                         cA)*prms['K']*prms['J']*\
        np.array([prms['hE'],prms['hI']/prms['beta']],dtype=np.float32)
    eH = CVh
    sW = np.array([[prms['SoriE'],prms['SoriI']],[prms['SoriE'],prms['SoriI']]],dtype=np.float32)
    sH = np.array([prms['SoriF'],prms['SoriF']],dtype=np.float32)
    
    muW = tau[:,None]*W*Ks
    SigW = tau[:,None]**2*W**2*Ks
    muWb = tau[:,None]*W*Kbs
    SigWb = tau[:,None]**2*W**2*Kbs
    
    sW2 = sW**2
    sH2 = sH**2
    
    muHb = tau*Hb
    muHp = tau*Hp
    smuH2 = sH2
    SigHb = (muHb*eH)**2
    SigHp = (muHp*eH)**2
    sSigH2 = 2*sH2

    μrs = np.zeros((2,3,Nori))
    Σrs = np.zeros((2,4,Nori))
    μmuEs = np.zeros((2,3,Nori))
    ΣmuEs = np.zeros((2,4,Nori))
    μmuIs = np.zeros((2,3,Nori))
    ΣmuIs = np.zeros((2,4,Nori))
    normC = np.zeros((2,3,Nori))
    conv = np.zeros((2,3)).astype(bool)
    
    oris = np.arange(Nori)*180/Nori
    oris[oris > 90] = 180 - oris[oris > 90]
        
    def gauss(x,b,p,s2):
        return b + (p-b)*np.exp(-0.5*x**2/s2)
    
    if cA == 0 or prms.get('basefrac',0)==1:
        res_dict = dmft.run_two_stage_dmft(prms,rX*(1+cA),CVh,'./../results',ri,Twrm,Tsav,dt)
        rvb = res_dict['r'][:2]
        rvp = res_dict['r'][:2]
        srv2 = 1e4*np.ones(2)
        rob = res_dict['r'][2:]
        rop = res_dict['r'][2:]
        sro2 = 1e4*np.ones(2)
        Crvb = dmft.grid_stat(np.mean,res_dict['Cr'][:2],Tsim,dt)
        Crvp = dmft.grid_stat(np.mean,res_dict['Cr'][:2],Tsim,dt)
        sCrv2 = 1e4*np.ones(2)
        Crob = dmft.grid_stat(np.mean,res_dict['Cr'][2:],Tsim,dt)
        Crop = dmft.grid_stat(np.mean,res_dict['Cr'][2:],Tsim,dt)
        sCro2 = 1e4*np.ones(2)
        Cdrb = dmft.grid_stat(np.mean,res_dict['Cdr'],Tsim,dt)
        Cdrp = dmft.grid_stat(np.mean,res_dict['Cdr'],Tsim,dt)
        sCdr2 = 1e4*np.ones(2)
        normC[:,0,:] = (res_dict['Cr'][:2,-1]/res_dict['Cr'][:2,0])[:,None]
        normC[:,1,:] = (res_dict['Cr'][2:,-1]/res_dict['Cr'][2:,0])[:,None]
        normC[:,2,:] = (res_dict['Cdr'][:,-1]/res_dict['Cdr'][:,0])[:,None]
        conv[:,0] = res_dict['conv'][:2]
        conv[:,1] = res_dict['conv'][2:]
        conv[:,2] = res_dict['convd']
        dmft_res = res_dict.copy()
    else:
        res_dict = dmft.run_two_stage_ring_dmft(prms,rX,cA,CVh,'./../results',ri,Twrm,Tsav,dt,
                                                sa=15*width*sa_mult)
        rvb = res_dict['rb'][:2]
        rvp = res_dict['rp'][:2]
        srv2 = res_dict['sr'][:2]**2
        rob = res_dict['rb'][2:]
        rop = res_dict['rp'][2:]
        sro2 = res_dict['sr'][2:]**2
        Crvb = dmft.grid_stat(np.mean,res_dict['Crb'][:2],Tsim,dt)
        Crvp = dmft.grid_stat(np.mean,res_dict['Crp'][:2],Tsim,dt)
        sCrv2 = dmft.grid_stat(np.mean,res_dict['sCr'][:2],Tsim,dt)**2
        Crob = dmft.grid_stat(np.mean,res_dict['Crb'][2:],Tsim,dt)
        Crop = dmft.grid_stat(np.mean,res_dict['Crp'][2:],Tsim,dt)
        sCro2 = dmft.grid_stat(np.mean,res_dict['sCr'][2:],Tsim,dt)**2
        Cdrb = dmft.grid_stat(np.mean,res_dict['Cdrb'],Tsim,dt)
        Cdrp = dmft.grid_stat(np.mean,res_dict['Cdrp'],Tsim,dt)
        sCdr2 = dmft.grid_stat(np.mean,res_dict['sCdr'],Tsim,dt)**2
        normC[:,0] = gauss(oris[None,:],res_dict['Crb'][:2,-1,None],res_dict['Crp'][:2,-1,None],
                           res_dict['sCr'][:2,-1,None]) /\
                     gauss(oris[None,:],res_dict['Crb'][:2, 0,None],res_dict['Crp'][:2, 0,None],
                           res_dict['sCr'][:2, 0,None])
        normC[:,1] = gauss(oris[None,:],res_dict['Crb'][2:,-1,None],res_dict['Crp'][2:,-1,None],
                           res_dict['sCr'][2:,-1,None]) /\
                     gauss(oris[None,:],res_dict['Crb'][2:, 0,None],res_dict['Crp'][2:, 0,None],
                           res_dict['sCr'][2:, 0,None])
        normC[:,2] = gauss(oris[None,:],res_dict['Cdrb'][:,-1,None],res_dict['Cdrp'][:,-1,None],
                           res_dict['sCdr'][:,-1,None]) /\
                     gauss(oris[None,:],res_dict['Cdrb'][:, 0,None],res_dict['Cdrp'][:, 0,None],
                           res_dict['sCdr'][:, 0,None])
        conv[:,0] = res_dict['convp'][:2]
        conv[:,1] = res_dict['convp'][2:]
        conv[:,2] = res_dict['convdp']
        dmft_res = res_dict.copy()
        
    sWrv2 = sW2+srv2
    sWCrv2 = sW2+sCrv2
    sWro2 = sW2+sro2
    sWCro2 = sW2+sCro2
    sWCdr2 = sW2+sCdr2
    
    muvb = (muW+np.sqrt(srv2/(2*np.pi))*muWb)*rvb
    muvp = muvb + np.sqrt(srv2/sWrv2)*muW*(rvp-rvb)
    smuv2 = sWrv2
    muob = (muW+np.sqrt(sro2/(2*np.pi))*muWb)*rob
    muop = muob + np.sqrt(sro2/sWro2)*muW*(rop-rob)
    smuo2 = sWro2
    
    Sigvb = (SigW+np.sqrt(sCrv2/(2*np.pi))*SigWb)*Crvb
    Sigvp = Sigvb + np.sqrt(sCrv2/sWCrv2)*SigW*(Crvp-Crvb)
    sSigv2 = sWCrv2
    Sigob = (SigW+np.sqrt(sCro2/(2*np.pi))*SigWb)*Crob
    Sigop = Sigob + np.sqrt(sCro2/sWCro2)*SigW*(Crop-Crob)
    sSigo2 = sWCro2
    Sigdb = (SigW+np.sqrt(sCdr2/(2*np.pi))*SigWb)*Cdrb
    Sigdp = Sigdb + np.sqrt(sCdr2/sWCdr2)*SigW*(Cdrp-Cdrb)
    sSigd2 = sWCdr2
    
    for i in range(2):
        μrs[i,0] = gauss(oris,rvb[i],rvp[i],srv2[i])
        μrs[i,1] = gauss(oris,rob[i],rop[i],sro2[i])
        μrs[i,2] = μrs[i,1] - μrs[i,0]
        Σrs[i,0] = np.fmax(gauss(oris,Crvb[i],Crvp[i],sCrv2[i]) - μrs[i,0]**2,0)
        Σrs[i,1] = np.fmax(gauss(oris,Crob[i],Crop[i],sCro2[i]) - μrs[i,1]**2,0)
        Σrs[i,2] = np.fmax(gauss(oris,Cdrb[i],Cdrp[i],sCdr2[i]) - μrs[i,2]**2,0)
        Σrs[i,3] = 0.5*(Σrs[i,1] - Σrs[i,0] - Σrs[i,2])
        μmuEs[i,0] = gauss(oris,muvb[i,0],muvp[i,0],smuv2[i,0]) + gauss(oris,muHb[i],muHp[i],smuH2[i])
        μmuEs[i,1] = gauss(oris,muob[i,0],muop[i,0],smuo2[i,0]) + gauss(oris,muHb[i],muHp[i],smuH2[i]) + prms['L']*1e-3
        μmuEs[i,2] = μmuEs[i,1] - μmuEs[i,0]
        ΣmuEs[i,0] = gauss(oris,Sigvb[i,0],Sigvp[i,0],sSigv2[i,0]) + gauss(oris,SigHb[i],SigHp[i],sSigH2[i])
        ΣmuEs[i,1] = gauss(oris,Sigob[i,0],Sigop[i,0],sSigo2[i,0]) + gauss(oris,SigHb[i],SigHp[i],sSigH2[i]) +\
            (prms['CVL']*prms['L']*1e-3)**2
        ΣmuEs[i,2] = gauss(oris,Sigdb[i,0],Sigdp[i,0],sSigd2[i,0]) + (prms['CVL']*prms['L']*1e-3)**2
        ΣmuEs[i,3] =  0.5*(ΣmuEs[i,1] - ΣmuEs[i,0] - ΣmuEs[i,2])
        μmuIs[i,0] = gauss(oris,muvb[i,1],muvp[i,1],smuv2[i,1])
        μmuIs[i,1] = gauss(oris,muob[i,1],muop[i,1],smuo2[i,1])
        μmuIs[i,2] = μmuIs[i,1] - μmuIs[i,0]
        ΣmuIs[i,0] = gauss(oris,Sigvb[i,1],Sigvp[i,1],sSigv2[i,1])
        ΣmuIs[i,1] = gauss(oris,Sigob[i,1],Sigop[i,1],sSigo2[i,1])
        ΣmuIs[i,2] = gauss(oris,Sigdb[i,1],Sigdp[i,1],sSigd2[i,1])
        ΣmuIs[i,3] =  0.5*(ΣmuIs[i,1] - ΣmuIs[i,0] - ΣmuIs[i,2])
    μmus = μmuEs + μmuIs
    Σmus = ΣmuEs + ΣmuIs

    return μrs,Σrs,μmus,Σmus,μmuEs,ΣmuEs,μmuIs,ΣmuIs,normC,conv,dmft_res

def calc_bal(μmuE,μmuI,ΣmuE,ΣmuI,N=10000):
    muEs = np.fmax(μmuE + np.sqrt(ΣmuE)*np.random.randn(N),1e-12)
    muIs = np.fmin(μmuI + np.sqrt(ΣmuI)*np.random.randn(N),-1e-12)
    return np.mean(np.abs(muEs+muIs)/muEs)

# Simulate zero and full contrast networks with ring connectivity
print('simulating baseline fraction network')
print('')
this_prms = prms.copy()
this_prms['J'] = newJ
this_prms['SoriE'] *= width
this_prms['SoriI'] *= width
this_prms['SoriF'] *= width
this_prms['baseinp'] = dmft.wrapnormdens(90,this_prms['SoriF']) / dmft.wrapnormdens(0,this_prms['SoriF'])

μrs,Σrs,μmus,Σmus,μmuEs,ΣmuEs,μmuIs,ΣmuIs,normC,conv,dmft_res = predict_networks(this_prms,rX,cA,CVh)

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
    balEs[:,nloc] = calc_bal(μmuEs[0,0,nloc],μmuIs[0,0,nloc],ΣmuEs[0,0,nloc],ΣmuIs[0,0,nloc])
    balIs[:,nloc] = calc_bal(μmuEs[1,0,nloc],μmuIs[1,0,nloc],ΣmuEs[1,0,nloc],ΣmuIs[1,0,nloc])
normCEs[:] = normC[0]
normCIs[:] = normC[1]
convs[:] = conv

oris = np.arange(Nori)*180/Nori
oris[oris > 90] = 180 - oris[oris > 90]
vsm_mask = np.abs(oris) < 4.5*width
oris = np.abs(np.arange(Nori)*180/Nori - 90)
oris[oris > 90] = 180 - oris[oris > 90]
osm_mask = np.abs(oris) < 4.5*width

all_base_means = 0.8*np.mean(μrEs[0]) + 0.2*np.mean(μrIs[0])
all_base_stds = np.sqrt(0.8*np.mean(ΣrEs[0]+μrEs[0]**2) + 0.2*np.mean(ΣrIs[0]+μrIs[0]**2) - all_base_means**2)
all_opto_means = 0.8*np.mean(μrEs[1]) + 0.2*np.mean(μrIs[1])
all_opto_stds = np.sqrt(0.8*np.mean(ΣrEs[1]+μrEs[1]**2) + 0.2*np.mean(ΣrIs[1]+μrIs[1]**2) - all_opto_means**2)
all_diff_means = all_opto_means - all_base_means
all_diff_stds = np.sqrt(0.8*np.mean(ΣrEs[2]+μrEs[2]**2) + 0.2*np.mean(ΣrIs[2]+μrIs[2]**2) - all_diff_means**2)
all_norm_covs = (0.8*np.mean(ΣrEs[3]+μrEs[0]*μrEs[2]) + 0.2*np.mean(ΣrIs[3]+μrIs[0]*μrIs[2]) -\
    all_base_means*all_diff_means) / all_diff_stds**2

vsm_base_means = 0.8*np.mean(μrEs[0,vsm_mask]) + 0.2*np.mean(μrIs[0,vsm_mask])
vsm_base_stds = np.sqrt(0.8*np.mean(ΣrEs[0,vsm_mask]+μrEs[0,vsm_mask]**2) +\
    0.2*np.mean(ΣrIs[0,vsm_mask]+μrIs[0,vsm_mask]**2) - vsm_base_means**2)
vsm_opto_means = 0.8*np.mean(μrEs[1,vsm_mask]) + 0.2*np.mean(μrIs[1,vsm_mask])
vsm_opto_stds = np.sqrt(0.8*np.mean(ΣrEs[1,vsm_mask]+μrEs[1,vsm_mask]**2) +\
    0.2*np.mean(ΣrIs[1,vsm_mask]+μrIs[1,vsm_mask]**2) - vsm_opto_means**2)
vsm_diff_means = vsm_opto_means - vsm_base_means
vsm_diff_stds = np.sqrt(0.8*np.mean(ΣrEs[2,vsm_mask]+μrEs[2,vsm_mask]**2) +\
    0.2*np.mean(ΣrIs[2,vsm_mask]+μrIs[2,vsm_mask]**2) - vsm_diff_means**2)
vsm_norm_covs = (0.8*np.mean(ΣrEs[3,vsm_mask]+μrEs[0,vsm_mask]*μrEs[2,vsm_mask]) +\
    0.2*np.mean(ΣrIs[3,vsm_mask]+μrIs[0,vsm_mask]*μrIs[2,vsm_mask]) -\
    vsm_base_means*vsm_diff_means) / vsm_diff_stds**2

osm_base_means = 0.8*np.mean(μrEs[0,osm_mask]) + 0.2*np.mean(μrIs[0,osm_mask])
osm_base_stds = np.sqrt(0.8*np.mean(ΣrEs[0,osm_mask]+μrEs[0,osm_mask]**2) +\
    0.2*np.mean(ΣrIs[0,osm_mask]+μrIs[0,osm_mask]**2) - osm_base_means**2)
osm_opto_means = 0.8*np.mean(μrEs[1,osm_mask]) + 0.2*np.mean(μrIs[1,osm_mask])
osm_opto_stds = np.sqrt(0.8*np.mean(ΣrEs[1,osm_mask]+μrEs[1,osm_mask]**2) +\
    0.2*np.mean(ΣrIs[1,osm_mask]+μrIs[1,osm_mask]**2) - osm_opto_means**2)
osm_diff_means = osm_opto_means - osm_base_means
osm_diff_stds = np.sqrt(0.8*np.mean(ΣrEs[2,osm_mask]+μrEs[2,osm_mask]**2) +\
    0.2*np.mean(ΣrIs[2,osm_mask]+μrIs[2,osm_mask]**2) - osm_diff_means**2)
osm_norm_covs = (0.8*np.mean(ΣrEs[3,osm_mask]+μrEs[0,osm_mask]*μrEs[2,osm_mask]) +\
    0.2*np.mean(ΣrIs[3,osm_mask]+μrIs[0,osm_mask]*μrIs[2,osm_mask]) -\
    osm_base_means*osm_diff_means) / osm_diff_stds**2

print("Saving statistics took ",time.process_time() - start," s")
print('')

res_dict = {}

res_dict['prms'] = this_prms
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
res_dict['all_base_means'] = all_base_means
res_dict['all_base_stds'] = all_base_stds
res_dict['all_opto_means'] = all_opto_means
res_dict['all_opto_stds'] = all_opto_stds
res_dict['all_diff_means'] = all_diff_means
res_dict['all_diff_stds'] = all_diff_stds
res_dict['all_norm_covs'] = all_norm_covs
res_dict['vsm_base_means'] = vsm_base_means
res_dict['vsm_base_stds'] = vsm_base_stds
res_dict['vsm_opto_means'] = vsm_opto_means
res_dict['vsm_opto_stds'] = vsm_opto_stds
res_dict['vsm_diff_means'] = vsm_diff_means
res_dict['vsm_diff_stds'] = vsm_diff_stds
res_dict['vsm_norm_covs'] = vsm_norm_covs
res_dict['osm_base_means'] = osm_base_means
res_dict['osm_base_stds'] = osm_base_stds
res_dict['osm_opto_means'] = osm_opto_means
res_dict['osm_opto_stds'] = osm_opto_stds
res_dict['osm_diff_means'] = osm_diff_means
res_dict['osm_diff_stds'] = osm_diff_stds
res_dict['osm_norm_covs'] = osm_norm_covs
res_dict['dmft_res'] = dmft_res

if np.isclose(sa_mult,1.0):
    with open('./../results/dmft_vary_width_{:d}_J_{:d}'.format(width_idx,J_idx)+'.pkl', 'wb') as handle:
        pickle.dump(res_dict,handle)
else:
    with open('./../results/dmft_vary_width_{:d}_J_{:d}_sa_{:.2f}'.format(width_idx,J_idx,sa_mult)+'.pkl', 'wb') as handle:
        pickle.dump(res_dict,handle)
