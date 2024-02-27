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

parser.add_argument('--iter_idx', '-n',  help='which iteration', type=int, default=0)
parser.add_argument('--id', '-i',  help='which id', type=str, default=None)
args = vars(parser.parse_args())
print(parser.parse_args())
iter_idx = args['iter_idx']
id = args['id']

if iter_idx == 0:
    if id is None:
        with open('./../results/best_fit.pkl', 'rb') as handle:
            res_dict = pickle.load(handle)
    else:
        if '(' in id:
            id = tuple(map(int, id.replace('(','').replace(')','').split(',')))
        if isinstance(id,str):
            with open('./../results/'+id+'.pkl', 'rb') as handle:
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
else:
    with open('./../results/dmft_grad_descent_id_{:s}_n_{:d}.pkl'.format(str(id),iter_idx-1), 'rb') as handle:
        res_dict = pickle.load(handle)
prms = res_dict['prms']
CVh = res_dict['best_monk_eX']
bX = res_dict['best_monk_bX']
aXs = res_dict['best_monk_aXs']
K = prms['K']
prms['SoriE'] = 35
prms['SoriI'] = 35
prms['SoriF'] = 25
L = prms['L']
CVL = prms['CVL']

ri = ric.Ricciardi()

# Twrm = 0.4
# Tsav = 0.1
# Tsim = 1.0
# dt = 0.01/15
Twrm = 0.6
Tsav = 0.2
Tsim = 1.0
dt = 0.01/3

Nori = 20

def get_prm_vec(prms,bX,fc_aX,CVh):
    prm_vec = np.array(list(prms.values()))[[4,5,6,7,8,9]]
    prm_vec[0] = np.log10(prm_vec[0])
    prm_vec[1] = np.log10(prm_vec[1])
    prm_vec = np.concatenate((prm_vec,np.array([bX,fc_aX,CVh])))
    return prm_vec

def get_prms_inps(prm_vec):
    prms = {'K': K,
            'SoriE': 35,
            'SoriI': 35,
            'SoriF': 25,
            'J': 10**prm_vec[0],
            'beta': 10**prm_vec[1],
            'gE': prm_vec[2],
            'gI': prm_vec[3],
            'hE': prm_vec[4],
            'hI': prm_vec[5],
            'L': L,
            'CVL': CVL}
    return prms,prm_vec[6],prm_vec[7],prm_vec[8]

init_prm_vec = get_prm_vec(prms,bX,aXs[-1],CVh)

prm_vec_range = np.array([
    [-4,-3],    # log10J
    [-1,.5],    # log10beta
    [ 3, 7],    # gE
    [ 2, 6],    # gI
    [ 0.1, 6],    # hE
    [ 0.1, 2],    # hI
    [ 0,20],    # bX
    [ 3,28],    # aX
    [ 0,.4],    # CVh
])

dprm_vec = 0.005*(prm_vec_range[:,1]-prm_vec_range[:,0])

def predict_networks(prms,rX,cA,CVh):
    tau = np.array([ri.tE,ri.tI],dtype=np.float32)
    W = prms['J']*np.array([[1,-prms['gE']],[1./prms['beta'],-prms['gI']/prms['beta']]],dtype=np.float32)
    Ks = (1-prms.get('basefrac',0))*np.array([prms['K'],prms['K']/4],dtype=np.float32)
    Kbs =   prms.get('basefrac',0) *np.array([prms['K'],prms['K']/4],dtype=np.float32)
    Hb = rX*(1+prms.get('basefrac',0)*cA)*prms['K']*prms['J']*\
        np.array([prms['hE'],prms['hI']/prms['beta']],dtype=np.float32)
    Hp = rX*(1+                       cA)*prms['K']*prms['J']*\
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
    # μmuEs = np.zeros((2,3,Nori))
    # ΣmuEs = np.zeros((2,4,Nori))
    # μmuIs = np.zeros((2,3,Nori))
    # ΣmuIs = np.zeros((2,4,Nori))
    normC = np.zeros((2,3,Nori))
    conv = np.zeros((2,3)).astype(bool)
    
    oris = np.arange(Nori)*180/Nori
    oris[oris > 90] = 180 - oris[oris > 90]
        
    def gauss(x,b,p,s):
        return b + (p-b)*dmft.basesubwrapnorm(x,s)
    
    if cA == 0 or prms.get('basefrac',0)==1:
        res_dict = dmft.run_two_stage_dmft(prms,rX*(1+cA),CVh,'./../results',ri,Twrm,Tsav,dt)
        rvb = res_dict['r'][:2]
        rvp = res_dict['r'][:2]
        srv = 1e2*np.ones(2)
        rob = res_dict['r'][2:]
        rop = res_dict['r'][2:]
        sro = 1e2*np.ones(2)
        Crvb = dmft.grid_stat(np.mean,res_dict['Cr'][:2],Tsim,dt)
        Crvp = dmft.grid_stat(np.mean,res_dict['Cr'][:2],Tsim,dt)
        sCrv = 1e2*np.ones(2)
        Crob = dmft.grid_stat(np.mean,res_dict['Cr'][2:],Tsim,dt)
        Crop = dmft.grid_stat(np.mean,res_dict['Cr'][2:],Tsim,dt)
        sCro = 1e2*np.ones(2)
        Cdrb = dmft.grid_stat(np.mean,res_dict['Cdr'],Tsim,dt)
        Cdrp = dmft.grid_stat(np.mean,res_dict['Cdr'],Tsim,dt)
        sCdr = 1e2*np.ones(2)
        normC[:,0,:] = (res_dict['Cr'][:2,-1]/res_dict['Cr'][:2,0])[:,None]
        normC[:,1,:] = (res_dict['Cr'][2:,-1]/res_dict['Cr'][2:,0])[:,None]
        normC[:,2,:] = (res_dict['Cdr'][:,-1]/res_dict['Cdr'][:,0])[:,None]
        conv[:,0] = res_dict['conv'][:2]
        conv[:,1] = res_dict['conv'][2:]
        conv[:,2] = res_dict['convd']
        dmft_res = res_dict.copy()
        print('convergences:',conv)
    else:
        res_dict = dmft.run_two_stage_ring_dmft(prms,rX,cA,CVh,'./../results',ri,Twrm,Tsav,dt)
        rvb = res_dict['rb'][:2]
        rvp = res_dict['rp'][:2]
        srv = res_dict['sr'][:2]
        rob = res_dict['rb'][2:]
        rop = res_dict['rp'][2:]
        sro = res_dict['sr'][2:]
        Crvb = dmft.grid_stat(np.mean,res_dict['Crb'][:2],Tsim,dt)
        Crvp = dmft.grid_stat(np.mean,res_dict['Crp'][:2],Tsim,dt)
        sCrv = dmft.grid_stat(np.mean,res_dict['sCr'][:2],Tsim,dt)
        Crob = dmft.grid_stat(np.mean,res_dict['Crb'][2:],Tsim,dt)
        Crop = dmft.grid_stat(np.mean,res_dict['Crp'][2:],Tsim,dt)
        sCro = dmft.grid_stat(np.mean,res_dict['sCr'][2:],Tsim,dt)
        Cdrb = dmft.grid_stat(np.mean,res_dict['Cdrb'],Tsim,dt)
        Cdrp = dmft.grid_stat(np.mean,res_dict['Cdrp'],Tsim,dt)
        sCdr = dmft.grid_stat(np.mean,res_dict['sCdr'],Tsim,dt)
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
        print('convergences:',conv)
        dmft_res = res_dict.copy()
        
    # sWrv = np.sqrt(sW2+srv**2)
    # sWCrv = np.sqrt(sW2+sCrv**2)
    # sWro = np.sqrt(sW2+sro**2)
    # sWCro = np.sqrt(sW2+sCro**2)
    # sWCdr = np.sqrt(sW2+sCdr**2)
    
    # muvb = (muW+dmft.unstruct_fact(srv)*muWb)*rvb
    # muvp = muvb + dmft.struct_fact(0,sWrv,srv)*muW*(rvp-rvb)
    # muvb = muvb + dmft.struct_fact(90,sWrv,srv)*muW*(rvp-rvb)
    # smuv = sWrv
    # muob = (muW+dmft.unstruct_fact(sro)*muWb)*rob
    # muop = muob + dmft.struct_fact(0,sWro,sro)*muW*(rop-rob)
    # muob = muob + dmft.struct_fact(90,sWro,sro)*muW*(rop-rob)
    # smuo = sWro
    
    # Sigvb = (SigW+dmft.unstruct_fact(sCrv)*SigWb)*Crvb
    # Sigvp = Sigvb + dmft.struct_fact(0,sWCrv,sCrv)*SigW*(Crvp-Crvb)
    # Sigvb = Sigvb + dmft.struct_fact(90,sWCrv,sCrv)*SigW*(Crvp-Crvb)
    # sSigv = sWCrv
    # Sigob = (SigW+dmft.unstruct_fact(sCro)*SigWb)*Crob
    # Sigop = Sigob + dmft.struct_fact(0,sWCro,sCro)*SigW*(Crop-Crob)
    # Sigob = Sigob + dmft.struct_fact(90,sWCro,sCro)*SigW*(Crop-Crob)
    # sSigo = sWCro
    # Sigdb = (SigW+dmft.unstruct_fact(sCdr)*SigWb)*Cdrb
    # Sigdp = Sigdb + dmft.struct_fact(0,sWCdr,sCdr)*SigW*(Cdrp-Cdrb)
    # Sigdb = Sigdb + dmft.struct_fact(90,sWCdr,sCdr)*SigW*(Cdrp-Cdrb)
    # sSigd = sWCdr
    
    for i in range(2):
        μrs[i,0] = gauss(oris,rvb[i],rvp[i],srv[i])
        μrs[i,1] = gauss(oris,rob[i],rop[i],sro[i])
        μrs[i,2] = μrs[i,1] - μrs[i,0]
        Σrs[i,0] = np.fmax(gauss(oris,Crvb[i],Crvp[i],sCrv[i]) - μrs[i,0]**2,0)
        Σrs[i,1] = np.fmax(gauss(oris,Crob[i],Crop[i],sCro[i]) - μrs[i,1]**2,0)
        Σrs[i,2] = np.fmax(gauss(oris,Cdrb[i],Cdrp[i],sCdr[i]) - μrs[i,2]**2,0)
        Σrs[i,3] = 0.5*(Σrs[i,1] - Σrs[i,0] - Σrs[i,2])
        # μmuEs[i,0] = gauss(oris,muvb[i,0],muvp[i,0],smuv[i,0]) + gauss(oris,muHb[i],muHp[i],smuH[i])
        # μmuEs[i,1] = gauss(oris,muob[i,0],muop[i,0],smuo[i,0]) + gauss(oris,muHb[i],muHp[i],smuH[i]) + prms['L']*1e-3
        # μmuEs[i,2] = μmuEs[i,1] - μmuEs[i,0]
        # ΣmuEs[i,0] = gauss(oris,Sigvb[i,0],Sigvp[i,0],sSigv[i,0]) + (gauss(oris,muHb[i],muHp[i],smuH[i])*eH)**2
        # ΣmuEs[i,1] = gauss(oris,Sigob[i,0],Sigop[i,0],sSigo[i,0]) + (gauss(oris,muHb[i],muHp[i],smuH[i])*eH)**2 +\
        #     (prms['CVL']*prms['L']*1e-3)**2
        # ΣmuEs[i,2] = gauss(oris,Sigdb[i,0],Sigdp[i,0],sSigd[i,0]) + (prms['CVL']*prms['L']*1e-3)**2
        # ΣmuEs[i,3] =  0.5*(ΣmuEs[i,1] - ΣmuEs[i,0] - ΣmuEs[i,2])
        # μmuIs[i,0] = gauss(oris,muvb[i,1],muvp[i,1],smuv[i,1])
        # μmuIs[i,1] = gauss(oris,muob[i,1],muop[i,1],smuo[i,1])
        # μmuIs[i,2] = μmuIs[i,1] - μmuIs[i,0]
        # ΣmuIs[i,0] = gauss(oris,Sigvb[i,1],Sigvp[i,1],sSigv[i,1])
        # ΣmuIs[i,1] = gauss(oris,Sigob[i,1],Sigop[i,1],sSigo[i,1])
        # ΣmuIs[i,2] = gauss(oris,Sigdb[i,1],Sigdp[i,1],sSigd[i,1])
        # ΣmuIs[i,3] =  0.5*(ΣmuIs[i,1] - ΣmuIs[i,0] - ΣmuIs[i,2])
    # μmus = μmuEs + μmuIs
    # Σmus = ΣmuEs + ΣmuIs

    return μrs,Σrs#,μmus,Σmus,μmuEs,ΣmuEs,μmuIs,ΣmuIs,normC,conv,dmft_res

monk_base_means =       np.array([20.38, 43.32, 54.76, 64.54, 70.97, 72.69])
monk_base_stds =        np.array([17.06, 32.41, 38.93, 42.76, 45.17, 48.61])
monk_opto_means =       np.array([30.82, 44.93, 53.36, 60.46, 64.09, 68.87])
monk_opto_stds =        np.array([36.36, 42.87, 45.13, 49.31, 47.53, 52.24])
monk_diff_means =       np.array([10.44,  1.61, -1.41, -4.08, -6.88, -3.82])
monk_diff_stds =        np.array([37.77, 42.48, 42.24, 45.43, 41.78, 41.71])
monk_norm_covs =        np.array([-0.1456, -0.2999, -0.3792, -0.3831, -0.4664, -0.4226])

monk_base_means_err =   np.array([ 2.39,  4.49,  5.38,  5.90,  6.22,  6.69])
monk_base_stds_err =    np.array([ 2.29,  3.67,  4.48,  5.10,  5.61,  5.32])
monk_opto_means_err =   np.array([ 5.03,  5.86,  6.16,  6.74,  6.50,  7.15])
monk_opto_stds_err =    np.array([ 7.73,  6.47,  5.90,  6.20,  4.93,  4.74])
monk_diff_means_err =   np.array([ 5.28,  5.90,  5.84,  6.28,  5.75,  5.76])
monk_diff_stds_err =    np.array([ 8.36,  8.74,  8.01, 10.04,  8.51,  8.94])
monk_norm_covs_err =    np.array([ 0.1075,  0.1354,  0.1579,  0.1496,  0.1717,  0.1665])

def calc_loss(zc_μrs,fc_μrs,zc_Σrs,fc_Σrs):
    zc_μrEs = zc_μrs[0]
    zc_μrIs = zc_μrs[1]
    zc_ΣrEs = zc_Σrs[0]
    zc_ΣrIs = zc_Σrs[1]
    fc_μrEs = fc_μrs[0]
    fc_μrIs = fc_μrs[1]
    fc_ΣrEs = fc_Σrs[0]
    fc_ΣrIs = fc_Σrs[1]
    
    oris = np.arange(Nori)*180/Nori
    oris[oris > 90] = 180 - oris[oris > 90]
    vsm_mask = np.abs(oris) < 4.5

    zc_vsm_base_means = 0.8*np.mean(zc_μrEs[0,vsm_mask]) + 0.2*np.mean(zc_μrIs[0,vsm_mask])
    zc_vsm_base_stds = np.sqrt(0.8*np.mean(zc_ΣrEs[0,vsm_mask]+zc_μrEs[0,vsm_mask]**2) +\
        0.2*np.mean(zc_ΣrIs[0,vsm_mask]+zc_μrIs[0,vsm_mask]**2) - zc_vsm_base_means**2)
    zc_vsm_opto_means = 0.8*np.mean(zc_μrEs[1,vsm_mask]) + 0.2*np.mean(zc_μrIs[1,vsm_mask])
    zc_vsm_opto_stds = np.sqrt(0.8*np.mean(zc_ΣrEs[1,vsm_mask]+zc_μrEs[1,vsm_mask]**2) +\
        0.2*np.mean(zc_ΣrIs[1,vsm_mask]+zc_μrIs[1,vsm_mask]**2) - zc_vsm_opto_means**2)
    zc_vsm_diff_means = zc_vsm_opto_means - zc_vsm_base_means
    zc_vsm_diff_stds = np.sqrt(0.8*np.mean(zc_ΣrEs[2,vsm_mask]+zc_μrEs[2,vsm_mask]**2) +\
        0.2*np.mean(zc_ΣrIs[2,vsm_mask]+zc_μrIs[2,vsm_mask]**2) - zc_vsm_diff_means**2)
    zc_vsm_norm_covs = (0.8*np.mean(zc_ΣrEs[3,vsm_mask]+zc_μrEs[0,vsm_mask]*zc_μrEs[2,vsm_mask]) +\
        0.2*np.mean(zc_ΣrIs[3,vsm_mask]+zc_μrIs[0,vsm_mask]*zc_μrIs[2,vsm_mask]) -\
        zc_vsm_base_means*zc_vsm_diff_means) / zc_vsm_diff_stds**2

    fc_vsm_base_means = 0.8*np.mean(fc_μrEs[0,vsm_mask]) + 0.2*np.mean(fc_μrIs[0,vsm_mask])
    fc_vsm_base_stds = np.sqrt(0.8*np.mean(fc_ΣrEs[0,vsm_mask]+fc_μrEs[0,vsm_mask]**2) +\
        0.2*np.mean(fc_ΣrIs[0,vsm_mask]+fc_μrIs[0,vsm_mask]**2) - fc_vsm_base_means**2)
    fc_vsm_opto_means = 0.8*np.mean(fc_μrEs[1,vsm_mask]) + 0.2*np.mean(fc_μrIs[1,vsm_mask])
    fc_vsm_opto_stds = np.sqrt(0.8*np.mean(fc_ΣrEs[1,vsm_mask]+fc_μrEs[1,vsm_mask]**2) +\
        0.2*np.mean(fc_ΣrIs[1,vsm_mask]+fc_μrIs[1,vsm_mask]**2) - fc_vsm_opto_means**2)
    fc_vsm_diff_means = fc_vsm_opto_means - fc_vsm_base_means
    fc_vsm_diff_stds = np.sqrt(0.8*np.mean(fc_ΣrEs[2,vsm_mask]+fc_μrEs[2,vsm_mask]**2) +\
        0.2*np.mean(fc_ΣrIs[2,vsm_mask]+fc_μrIs[2,vsm_mask]**2) - fc_vsm_diff_means**2)
    fc_vsm_norm_covs = (0.8*np.mean(fc_ΣrEs[3,vsm_mask]+fc_μrEs[0,vsm_mask]*fc_μrEs[2,vsm_mask]) +\
        0.2*np.mean(fc_ΣrIs[3,vsm_mask]+fc_μrIs[0,vsm_mask]*fc_μrIs[2,vsm_mask]) -\
        fc_vsm_base_means*fc_vsm_diff_means) / fc_vsm_diff_stds**2
    
    res = np.array([
        (zc_vsm_base_means-monk_base_means[ 0])/monk_base_means_err[ 0],
        (zc_vsm_opto_means-monk_opto_means[ 0])/monk_opto_means_err[ 0],
        (zc_vsm_base_stds - monk_base_stds[ 0])/ monk_base_stds_err[ 0],
        (zc_vsm_opto_stds - monk_opto_stds[ 0])/ monk_opto_stds_err[ 0],
        (zc_vsm_diff_stds - monk_diff_stds[ 0])/ monk_diff_stds_err[ 0],
        (zc_vsm_norm_covs - monk_norm_covs[ 0])/ monk_norm_covs_err[ 0],
        (fc_vsm_base_means-monk_base_means[-1])/monk_base_means_err[-1]*2,
        (fc_vsm_opto_means-monk_opto_means[-1])/monk_opto_means_err[-1]*2,
        (fc_vsm_base_stds - monk_base_stds[-1])/ monk_base_stds_err[-1]*2,
        (fc_vsm_opto_stds - monk_opto_stds[-1])/ monk_opto_stds_err[-1]*2,
        (fc_vsm_diff_stds - monk_diff_stds[-1])/ monk_diff_stds_err[-1]*2,
        (fc_vsm_norm_covs - monk_norm_covs[-1])/ monk_norm_covs_err[-1]*2,
    ])
    
    return 0.5*np.sum(res**2)

def get_loss_from_prms_vec(prm_vec):
    this_prms,this_bX,this_fc_aX,this_CVh = get_prms_inps(prm_vec)
    
    start = time.process_time()
    zc_μrs,zc_Σrs = predict_networks(this_prms,this_bX,0,this_CVh)
    print("Solving DMFT of zero contrast network took ",time.process_time() - start," s")
    print('')

    start = time.process_time()
    fc_μrs,fc_Σrs = predict_networks(this_prms,this_bX,this_fc_aX/this_bX,this_CVh)
    print("Solving DMFT of full contrast network took ",time.process_time() - start," s")
    print('')

    start = time.process_time()
    loss = calc_loss(zc_μrs,fc_μrs,zc_Σrs,fc_Σrs)
    print("Calculating loss took ",time.process_time() - start," s")
    print('')
    
    return loss

# Simulate zero and full contrast networks with ring connectivity
print('simulating unperturbed network')
print('')

init_loss = get_loss_from_prms_vec(init_prm_vec)
pert_losses = np.zeros_like(dprm_vec)

for idx,dprm in enumerate(dprm_vec):
    print('simulating perturbed network, perturbed var = '+\
        ['SoriE','SoriI','log10J','log10beta',
         'gE','gI','hE','hI','bX','fc_aX','CVh'][idx])
    print('')
    pert_prm_vec = init_prm_vec.copy()
    pert_prm_vec[idx] += dprm
    pert_losses[idx] = get_loss_from_prms_vec(pert_prm_vec)
    
grad = (pert_losses - init_loss) / dprm_vec

final_prm_vec = init_prm_vec - 0.001*10**(-iter_idx/8)*grad
final_prm_vec = np.clip(final_prm_vec,prm_vec_range[:,0],prm_vec_range[:,1])

final_prms,final_bX,final_fc_aX,final_CVh = get_prms_inps(final_prm_vec)

CVh = res_dict['best_monk_eX']
bX = res_dict['best_monk_bX']
aXs = res_dict['best_monk_aXs']

res_dict = {}

res_dict['prms'] = final_prms
res_dict['best_monk_bX'] = final_bX
res_dict['best_monk_aXs'] = aXs * final_fc_aX / aXs[-1]
res_dict['best_monk_eX'] = final_CVh
res_dict['init_prm_vec'] = init_prm_vec
res_dict['final_prm_vec'] = final_prm_vec
res_dict['init_loss'] = init_loss
res_dict['pert_losses'] = pert_losses
res_dict['grad'] = grad
res_dict['prm_vec_range'] = prm_vec_range
res_dict['dprm_vec'] = dprm_vec

with open('./../results/dmft_grad_descent_id_{:s}_n_{:d}'.format(str(id),iter_idx)+'.pkl', 'wb') as handle:
    pickle.dump(res_dict,handle)

if id is None:
    os.system("python runjob_dmft_grad_desc.py -n {:d}".format(iter_idx+1))
else:
    os.system("python runjob_dmft_grad_desc.py -n {:d} -i {:s}".format(iter_idx+1,id))
