import inspect
import time

import numpy as np
import torch

import ring_network as ring_network
import spat_ori_network as spat_network
import integrate as integ

def gen_ring_disorder(seed,prm_dict,eX,vis_ori=None,opto_per_pop=None):
    net = ring_network.RingNetwork(seed=0,NC=[prm_dict.get('NE',4),prm_dict.get('NI',1)],
        Nori=prm_dict.get('Nori',180))

    K = prm_dict.get('K',500)
    SoriE = prm_dict.get('SoriE',30)
    SoriI = prm_dict.get('SoriI',30)
    SoriF = prm_dict.get('SoriF',30)
    J = prm_dict.get('J',1e-4)
    beta = prm_dict.get('beta',1)
    gE = prm_dict.get('gE',5)
    gI = prm_dict.get('gI',4)
    hE = prm_dict.get('hE',1)
    hI = prm_dict.get('hI',1)
    L = prm_dict.get('L',1)
    CVL = prm_dict.get('CVL',1)

    WMat = J*np.array([[1,-gE],[1./beta,-gI/beta]],dtype=np.float32)
    HVec = K*J*np.array([hE,hI/beta],dtype=np.float32)

    net.set_seed(seed)
    net.generate_disorder(WMat,np.array([[SoriE,SoriI],[SoriE,SoriI]]),HVec,SoriF*np.ones(2),K,
                          baseinp=1-(1-prm_dict.get('baseinp',0))*(1-prm_dict.get('basefrac',0)),
                          baseprob=1-(1-prm_dict.get('baseprob',0))*(1-prm_dict.get('basefrac',0)),
                          rho=prm_dict.get('rho',0),vis_ori=vis_ori)

    B = np.zeros(net.N,dtype=np.float32)
    B[net.C_all[0]] = HVec[0]
    B[net.C_all[1]] = HVec[1]

    if opto_per_pop is None:
        LAS = np.zeros(net.N,dtype=np.float32)
        sigma_l = np.sqrt(np.log(1+CVL**2))
        mu_l = np.log(1e-3*L)-sigma_l**2/2
        LAS_E = np.random.default_rng(seed).lognormal(mu_l, sigma_l, net.NC[0]*net.Nloc).astype(np.float32)
        LAS[net.C_all[0]] = LAS_E
    else:
        LAS = np.zeros(net.N,dtype=np.float32)
        for nc in range(net.n):
            sigma_l = np.sqrt(np.log(1+CVL**2))
            mu_l = np.log(1e-3*L)-sigma_l**2/2
            LAS_P = np.random.default_rng(seed).lognormal(mu_l, sigma_l, net.NC[nc]*net.Nloc).astype(np.float32)
            LAS[net.C_all[nc]] = opto_per_pop[nc]*LAS_P

    if eX == 0:
        eps = np.ones(net.N,dtype=np.float32)
    else:
        shape = 1/eX**2
        scale = 1/shape
        eps = np.random.default_rng(seed).gamma(shape,scale=scale,size=net.N).astype(np.float32)

    return net,net.M,net.H,B,LAS,eps

def gen_ring_disorder_tensor(seed,prm_dict,eX,vis_ori=None,opto_per_pop=None,device=None):
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    net = ring_network.RingNetwork(seed=0,NC=[prm_dict.get('NE',4),prm_dict.get('NI',1)],
        Nori=prm_dict.get('Nori',180))

    K = prm_dict.get('K',500)
    SoriE = prm_dict.get('SoriE',30)
    SoriI = prm_dict.get('SoriI',30)
    SoriF = prm_dict.get('SoriF',30)
    J = prm_dict.get('J',1e-4)
    beta = prm_dict.get('beta',1)
    gE = prm_dict.get('gE',5)
    gI = prm_dict.get('gI',4)
    hE = prm_dict.get('hE',1)
    hI = prm_dict.get('hI',1)
    L = prm_dict.get('L',1)
    CVL = prm_dict.get('CVL',1)

    WMat = J*np.array([[1,-gE],[1./beta,-gI/beta]],dtype=np.float32)
    HVec = K*J*np.array([hE,hI/beta],dtype=np.float32)

    net.set_seed(seed)
    net.generate_disorder(WMat,np.array([[SoriE,SoriI],[SoriE,SoriI]]),HVec,SoriF*np.ones(2),K,
                          baseinp=1-(1-prm_dict.get('baseinp',0))*(1-prm_dict.get('basefrac',0)),
                          baseprob=1-(1-prm_dict.get('baseprob',0))*(1-prm_dict.get('basefrac',0)),
                          rho=prm_dict.get('rho',0),vis_ori=vis_ori)
    net.generate_tensors(device=device)

    B = torch.where(net.C_conds[0],HVec[0],HVec[1])

    if opto_per_pop is None:
        LAS = torch.zeros(net.N,dtype=torch.float32)
        sigma_l = np.sqrt(np.log(1+CVL**2))
        mu_l = np.log(1e-3*L)-sigma_l**2/2
        LAS_E = np.random.default_rng(seed).lognormal(mu_l, sigma_l, net.NC[0]*net.Nloc).astype(np.float32)
        LAS[net.C_conds[0]] = torch.from_numpy(LAS_E)
    else:
        LAS = torch.zeros(net.N,dtype=torch.float32)
        for nc in range(net.n):
            sigma_l = np.sqrt(np.log(1+CVL**2))
            mu_l = np.log(1e-3*L)-sigma_l**2/2
            LAS_P = np.random.default_rng(seed).lognormal(mu_l, sigma_l, net.NC[nc]*net.Nloc).astype(np.float32)
            LAS[net.C_conds[nc]] = opto_per_pop[nc]*torch.from_numpy(LAS_P)

    B = B.to(device)
    LAS = LAS.to(device)

    if eX == 0:
        eps = torch.ones(net.N,dtype=torch.float32)
        eps = eps.to(device)
    else:
        shape = 1/eX**2
        scale = 1/shape
        this_eps = np.random.default_rng(seed).gamma(shape,scale=scale,size=net.N).astype(np.float32)
        eps = torch.from_numpy(this_eps)
        eps = eps.to(device)

    return net,net.M_torch,net.H_torch,B,LAS,eps

def sim_ring_tensor(prm_dict,eX,bX,aX,ri,T,mask_time,seeds,return_dynas=False,max_min=15):
    net = ring_network.RingNetwork(seed=0,NC=[prm_dict.get('NE',4),prm_dict.get('NI',1)],
        Nori=prm_dict.get('Nori',180))

    rates = np.zeros((2,len(seeds),net.N))
    if return_dynas:
        dynas = np.zeros((2,len(seeds),net.N,len(T)))

    for seed_idx,seed in enumerate(seeds):
        print('Doing seed '+str(seed_idx+1) +' of '+str(len(seeds)))

        start = time.process_time()

        net,M,H,B,LAS,eps = gen_ring_disorder_tensor(seed,prm_dict,eX)

        print("Generating disorder took ",time.process_time() - start," s")
        print('')
        
        start = time.process_time()
        
        sol,base_timeout = integ.sim_dyn_tensor(ri,T,0.0,M,(bX*B+aX*H)*eps,LAS,net.C_conds[0],
            mult_tau=True,max_min=max_min)
        rates[0,seed_idx,:]=torch.mean(sol[:,mask_time],axis=1).cpu().numpy()
        if return_dynas:
            dynas[0,seed_idx,:,:]=sol.cpu().numpy()

        print("Integrating base network took ",time.process_time() - start," s")
        print('')
        
        start = time.process_time()
        
        sol,opto_timeout = integ.sim_dyn_tensor(ri,T,1.0,M,(bX*B+aX*H)*eps,LAS,net.C_conds[0],
            mult_tau=True,max_min=max_min)
        rates[1,seed_idx,:]=torch.mean(sol[:,mask_time],axis=1).cpu().numpy()
        if return_dynas:
            dynas[1,seed_idx,:,:]=sol.cpu().numpy()

        print("Integrating opto network took ",time.process_time() - start," s")
        print('')
        
    if return_dynas:
        return net,rates.reshape((2,-1)),dynas.reshape((2,-1,len(T))),(base_timeout or opto_timeout)
    else:
        return net,rates.reshape((2,-1)),(base_timeout or opto_timeout)

def sim_ring(params_dict,ri,T,mask_time,seeds,return_dynas=False,max_min=15,stat_stop=True):
    this_params_dict = params_dict.copy()
    this_params_dict['seed_con'] = 0
    filtered_mydict_net = {k: v for k, v in this_params_dict.items() if k in [p.name for p in
                                                    inspect.signature(network.network).parameters.values()]}
    net = network.network(**filtered_mydict_net)
    rates = np.zeros((len(seeds),2,net.N))
    
    for seed_idx,seed in enumerate(seeds):
        print('Doing seed '+str(seed_idx+1) +' of '+str(len(seeds)))
        
        net.set_seed(int(seed))
        
        filtered_mydict_disorder = {k: v for k, v in this_params_dict.items() if k in [p.name for p in
                                            inspect.signature(net.generate_disorder).parameters.values()]}
        net.generate_disorder(**filtered_mydict_disorder)
        
        sol,base_timeout = integ.sim_dyn(ri,T,0.0,net.M,net.H,net.LAM,net.E_all,net.I_all,
            mult_tau=False,max_min=max_min,stat_stop=stat_stop)
        rates[seed_idx,0,:]=np.mean(sol[:,mask_time],axis=1)
        
        sol,opto_timeout = integ.sim_dyn(ri,T,params_dict['L'],net.M,net.H,net.LAM,net.E_all,net.I_all,
            mult_tau=False,max_min=max_min,stat_stop=stat_stop)
        rates[seed_idx,1,:]=np.mean(sol[:,mask_time],axis=1)
        
    return net,np.hstack([rates[i,:,:] for i in np.arange(len(seeds))]),(base_timeout or opto_timeout)

# def sim_ring_tensor(params_dict,ri,T,mask_time,seeds,max_min=15):
#     this_params_dict = params_dict.copy()
#     this_params_dict['seed_con'] = 0
#     filtered_mydict_net = {k: v for k, v in this_params_dict.items() if k in [p.name for p in
#                                                     inspect.signature(network.network).parameters.values()]}
#     net = network.network(**filtered_mydict_net)
#     rates = np.zeros((len(seeds),2,net.N))
    
#     for seed_idx,seed in enumerate(seeds):
#         print('Doing seed '+str(seed_idx+1) +' of '+str(len(seeds)))
        
#         net.set_seed(int(seed))
        
#         filtered_mydict_disorder = {k: v for k, v in this_params_dict.items() if k in [p.name for p in
#                                             inspect.signature(net.generate_disorder).parameters.values()]}
#         net.generate_disorder(**filtered_mydict_disorder)
#         net.generate_tensors()
        
#         sol,_ = integ.sim_dyn_tensor(ri,T,0.0,net.M_torch,net.H_torch,net.LAM_torch,net.E_cond,
#             mult_tau=False)
#         try:
#             rates[seed_idx,0,:]=torch.mean(sol[:,mask_time],axis=1).numpy()
#         except:
#             rates[seed_idx,0,:]=torch.mean(sol[:,mask_time],axis=1).cpu().numpy()
        
#         sol,_ = integ.sim_dyn_tensor(ri,T,params_dict['L'],net.M_torch,net.H_torch,net.LAM_torch,net.E_cond,
#             mult_tau=False)
#         try:
#             rates[seed_idx,1,:]=torch.mean(sol[:,mask_time],axis=1).numpy()
#         except:
#             rates[seed_idx,1,:]=torch.mean(sol[:,mask_time],axis=1).cpu().numpy()
        
#     return net,np.hstack([rates[i,:,:] for i in np.arange(len(seeds))])

def sim_spat(params_dict,ri,T,mask_time,seeds,max_min=15,ori_type='ring'):
    this_params_dict = params_dict.copy()
    this_params_dict['seed_con'] = 0
    this_params_dict['ori_type'] = ori_type

    filtered_mydict_net = {k: v for k, v in this_params_dict.items() if k in [p.name for p in
                                                    inspect.signature(spat_network.SpatOriNetwork).parameters.values()]}
    net = spat_network.SpatOriNetwork(**filtered_mydict_net)
    rates = np.zeros((len(seeds),2,net.N))
    
    for seed_idx,seed in enumerate(seeds):
        print('Doing seed '+str(seed_idx+1) +' of '+str(len(seeds)))
        
        net.set_seed(int(seed))
        
        filtered_mydict_disorder = {k: v for k, v in this_params_dict.items() if k in [p.name for p in
                                            inspect.signature(net.generate_disorder).parameters.values()]}
        net.generate_disorder(**filtered_mydict_disorder)
        
        sol,_ = integ.sim_dyn(ri,T,0.0,net.M,net.H,net.H,net.C_all[0],net.C_all[1],mult_tau=False,max_min=30)
        rates[seed_idx,0,:]=np.mean(sol[:,mask_time],axis=1)
        
        # sol,_ = integ.sim_dyn(ri,T,params_dict['L'],net.M,net.H,net.LAM,net.E_all,net.I_all,mult_tau=False,max_min=30)
        # rates[seed_idx,1,:]=np.mean(sol[:,mask_time],axis=1)
        
    return net,np.hstack([rates[i,:,:] for i in np.arange(len(seeds))])

def sim_spat_tensor(params_dict,ri,T,mask_time,seeds,max_min=15,ori_type='ring'):
    this_params_dict = params_dict.copy()
    this_params_dict['seed_con'] = 0
    this_params_dict['ori_type'] = ori_type

    filtered_mydict_net = {k: v for k, v in this_params_dict.items() if k in [p.name for p in
                                                    inspect.signature(spat_network.SpatOriNetwork).parameters.values()]}
    net = spat_network.SpatOriNetwork(**filtered_mydict_net)
    rates = np.zeros((len(seeds),2,net.N))
    
    for seed_idx,seed in enumerate(seeds):
        print('Doing seed '+str(seed_idx+1) +' of '+str(len(seeds)))
        
        net.set_seed(int(seed))
        
        filtered_mydict_disorder = {k: v for k, v in this_params_dict.items() if k in [p.name for p in
                                            inspect.signature(net.generate_disorder).parameters.values()]}
        net.generate_disorder(**filtered_mydict_disorder)
        net.generate_tensors()
        
        sol = integ.sim_dyn_tensor(ri,T,0.0,net.M_torch,net.H_torch,net.H_torch,net.C_conds[0],mult_tau=False)
        try:
            rates[seed_idx,0,:]=torch.mean(sol[mask_time],axis=0).numpy()
        except:
            rates[seed_idx,0,:]=torch.mean(sol[mask_time],axis=0).cpu().numpy()
        
        # sol = integ.sim_dyn_tensor(ri,T,params_dict['L'],net.M_torch,net.H_torch,net.LAM_torch,net.E_cond,mult_tau=False)
        # try:
        #     rates[seed_idx,1,:]=torch.mean(sol[mask_time],axis=0).numpy()
        # except:
        #     rates[seed_idx,1,:]=torch.mean(sol[mask_time],axis=0).cpu().numpy()
        
    return net,np.hstack([rates[i,:,:] for i in np.arange(len(seeds))])

def get_ring_input_rate(net,seeds,rates):
    oris = net.make_periodic(np.concatenate([net.Z for i in range(len(seeds))]),90)

    H_base_input=np.zeros((len(seeds),net.N))
    H_opto_input=np.zeros((len(seeds),net.N))
    H_diff_input=np.zeros((len(seeds),net.N))

    E_base_input=np.zeros((len(seeds),net.N))
    E_opto_input=np.zeros((len(seeds),net.N))
    E_diff_input=np.zeros((len(seeds),net.N))

    I_base_input=np.zeros((len(seeds),net.N))
    I_opto_input=np.zeros((len(seeds),net.N))
    I_diff_input=np.zeros((len(seeds),net.N))

    for seed_idx,seed in enumerate(seeds):
        net.set_seed(seed)
        filtered_mydict_disorder = {k: v for k, v in params_dict.items() if k in [p.name for p in
                                            inspect.signature(net.generate_disorder).parameters.values()]}
        net.generate_disorder(**filtered_mydict_disorder)
        
        this_rates = rates[:,seed_idx*net.N:(seed_idx+1)*net.N]
        
        MRbE = np.matmul(net.M[:,net.E_all],rates[0,net.E_all])
        MRbI = np.matmul(net.M[:,net.I_all],rates[0,net.I_all])
        MRoE = np.matmul(net.M[:,net.E_all],rates[1,net.E_all])
        MRoI = np.matmul(net.M[:,net.I_all],rates[1,net.I_all])
        
        H_base_input[seed_idx] = net.H
        H_opto_input[seed_idx] = net.H+L*net.LAM
        H_diff_input[seed_idx] = L*net.LAM
        
        E_base_input[seed_idx] = MRbE
        E_opto_input[seed_idx] = MRoE
        E_diff_input[seed_idx] = MRoE-MRbE
        
        I_base_input[seed_idx] = MRbI
        I_opto_input[seed_idx] = MRoI
        I_diff_input[seed_idx] = MRoI-MRbI

    rate_nob = net.Nl//2
    rate_ori_bounds=np.linspace(0,90,rate_nob+1)
    rate_ori_centers = 0.5*(rate_ori_bounds[1:]+rate_ori_bounds[:-1])
    rate_diff_width = (rate_ori_bounds[1]-rate_ori_bounds[0])/2

    EF_base_input_per_ori_mean=np.zeros((rate_nob+1))
    IF_base_input_per_ori_mean=np.zeros((rate_nob+1))
    EF_base_input_per_ori_std=np.zeros((rate_nob+1))
    IF_base_input_per_ori_std=np.zeros((rate_nob+1))

    EE_base_input_per_ori_mean=np.zeros((rate_nob+1))
    IE_base_input_per_ori_mean=np.zeros((rate_nob+1))
    EE_base_input_per_ori_std=np.zeros((rate_nob+1))
    IE_base_input_per_ori_std=np.zeros((rate_nob+1))

    EI_base_input_per_ori_mean=np.zeros((rate_nob+1))
    II_base_input_per_ori_mean=np.zeros((rate_nob+1))
    EI_base_input_per_ori_std=np.zeros((rate_nob+1))
    II_base_input_per_ori_std=np.zeros((rate_nob+1))

    ET_base_input_per_ori_mean=np.zeros((rate_nob+1))
    IT_base_input_per_ori_mean=np.zeros((rate_nob+1))
    ET_base_input_per_ori_std=np.zeros((rate_nob+1))
    IT_base_input_per_ori_std=np.zeros((rate_nob+1))

    EF_opto_input_per_ori_mean=np.zeros((rate_nob+1))
    IF_opto_input_per_ori_mean=np.zeros((rate_nob+1))
    EF_opto_input_per_ori_std=np.zeros((rate_nob+1))
    IF_opto_input_per_ori_std=np.zeros((rate_nob+1))

    EE_opto_input_per_ori_mean=np.zeros((rate_nob+1))
    IE_opto_input_per_ori_mean=np.zeros((rate_nob+1))
    EE_opto_input_per_ori_std=np.zeros((rate_nob+1))
    IE_opto_input_per_ori_std=np.zeros((rate_nob+1))

    EI_opto_input_per_ori_mean=np.zeros((rate_nob+1))
    II_opto_input_per_ori_mean=np.zeros((rate_nob+1))
    EI_opto_input_per_ori_std=np.zeros((rate_nob+1))
    II_opto_input_per_ori_std=np.zeros((rate_nob+1))

    ET_opto_input_per_ori_mean=np.zeros((rate_nob+1))
    IT_opto_input_per_ori_mean=np.zeros((rate_nob+1))
    ET_opto_input_per_ori_std=np.zeros((rate_nob+1))
    IT_opto_input_per_ori_std=np.zeros((rate_nob+1))

    E_base_rate_per_ori_mean=np.zeros((rate_nob+1))
    I_base_rate_per_ori_mean=np.zeros((rate_nob+1))
    E_base_rate_per_ori_std=np.zeros((rate_nob+1))
    I_base_rate_per_ori_std=np.zeros((rate_nob+1))

    E_opto_rate_per_ori_mean=np.zeros((rate_nob+1))
    I_opto_rate_per_ori_mean=np.zeros((rate_nob+1))
    E_opto_rate_per_ori_std=np.zeros((rate_nob+1))
    I_opto_rate_per_ori_std=np.zeros((rate_nob+1))

    E_diff_rate_per_ori_mean=np.zeros((rate_nob+1))
    I_diff_rate_per_ori_mean=np.zeros((rate_nob+1))
    E_diff_rate_per_ori_std=np.zeros((rate_nob+1))
    I_diff_rate_per_ori_std=np.zeros((rate_nob+1))
    E_diff_rate_per_ori_cov=np.zeros((rate_nob+1))
    I_diff_rate_per_ori_cov=np.zeros((rate_nob+1))
        
    E_all = np.concatenate([net.E_all+i*net.N for i in range(len(seeds))])
    I_all = np.concatenate([net.I_all+i*net.N for i in range(len(seeds))])

    θEs = net.make_periodic(oris[E_all],90)
    θIs = net.make_periodic(oris[I_all],90)
    FbEs = H_base_input[:,net.E_all].flatten()
    FbIs = H_base_input[:,net.I_all].flatten()
    EbEs = E_base_input[:,net.E_all].flatten()
    EbIs = E_base_input[:,net.I_all].flatten()
    IbEs = I_base_input[:,net.E_all].flatten()
    IbIs = I_base_input[:,net.I_all].flatten()
    FoEs = H_opto_input[:,net.E_all].flatten()
    FoIs = H_opto_input[:,net.I_all].flatten()
    EoEs = E_opto_input[:,net.E_all].flatten()
    EoIs = E_opto_input[:,net.I_all].flatten()
    IoEs = I_opto_input[:,net.E_all].flatten()
    IoIs = I_opto_input[:,net.I_all].flatten()
    rbEs = rates[0,E_all]
    rbIs = rates[0,I_all]
    roEs = rates[1,E_all]
    roIs = rates[1,I_all]

    for l in range(rate_nob+1):
        this_FbEs = FbEs[np.logical_and(θEs >= rate_ori_bounds[l]-rate_diff_width,
                                        θEs <  rate_ori_bounds[l]+rate_diff_width)]
        this_FbIs = FbIs[np.logical_and(θIs >= rate_ori_bounds[l]-rate_diff_width,
                                        θIs <  rate_ori_bounds[l]+rate_diff_width)]
        this_EbEs = EbEs[np.logical_and(θEs >= rate_ori_bounds[l]-rate_diff_width,
                                        θEs <  rate_ori_bounds[l]+rate_diff_width)]
        this_EbIs = EbIs[np.logical_and(θIs >= rate_ori_bounds[l]-rate_diff_width,
                                        θIs <  rate_ori_bounds[l]+rate_diff_width)]
        this_IbEs = IbEs[np.logical_and(θEs >= rate_ori_bounds[l]-rate_diff_width,
                                        θEs <  rate_ori_bounds[l]+rate_diff_width)]
        this_IbIs = IbIs[np.logical_and(θIs >= rate_ori_bounds[l]-rate_diff_width,
                                        θIs <  rate_ori_bounds[l]+rate_diff_width)]
        this_TbEs = this_FbEs + this_EbEs + this_IbEs
        this_TbIs = this_FbIs + this_EbIs + this_IbIs
        
        this_FoEs = FoEs[np.logical_and(θEs >= rate_ori_bounds[l]-rate_diff_width,
                                        θEs <  rate_ori_bounds[l]+rate_diff_width)]
        this_FoIs = FoIs[np.logical_and(θIs >= rate_ori_bounds[l]-rate_diff_width,
                                        θIs <  rate_ori_bounds[l]+rate_diff_width)]
        this_EoEs = EoEs[np.logical_and(θEs >= rate_ori_bounds[l]-rate_diff_width,
                                        θEs <  rate_ori_bounds[l]+rate_diff_width)]
        this_EoIs = EoIs[np.logical_and(θIs >= rate_ori_bounds[l]-rate_diff_width,
                                        θIs <  rate_ori_bounds[l]+rate_diff_width)]
        this_IoEs = IoEs[np.logical_and(θEs >= rate_ori_bounds[l]-rate_diff_width,
                                        θEs <  rate_ori_bounds[l]+rate_diff_width)]
        this_IoIs = IoIs[np.logical_and(θIs >= rate_ori_bounds[l]-rate_diff_width,
                                        θIs <  rate_ori_bounds[l]+rate_diff_width)]
        this_ToEs = this_FoEs + this_EoEs + this_IoEs
        this_ToIs = this_FoIs + this_EoIs + this_IoIs
        
        this_rbEs = rbEs[np.logical_and(θEs >= rate_ori_bounds[l]-rate_diff_width,
                                        θEs <  rate_ori_bounds[l]+rate_diff_width)]
        this_rbIs = rbIs[np.logical_and(θIs >= rate_ori_bounds[l]-rate_diff_width,
                                        θIs <  rate_ori_bounds[l]+rate_diff_width)]
        this_roEs = roEs[np.logical_and(θEs >= rate_ori_bounds[l]-rate_diff_width,
                                        θEs <  rate_ori_bounds[l]+rate_diff_width)]
        this_roIs = roIs[np.logical_and(θIs >= rate_ori_bounds[l]-rate_diff_width,
                                        θIs <  rate_ori_bounds[l]+rate_diff_width)]
        
        EF_base_input_per_ori_mean[l]=np.mean(this_FbEs)
        IF_base_input_per_ori_mean[l]=np.mean(this_FbIs)
        EF_base_input_per_ori_std[l]=np.std(this_FbEs)
        IF_base_input_per_ori_std[l]=np.std(this_FbIs)
        
        EE_base_input_per_ori_mean[l]=np.mean(this_EbEs)
        IE_base_input_per_ori_mean[l]=np.mean(this_EbIs)
        EE_base_input_per_ori_std[l]=np.std(this_EbEs)
        IE_base_input_per_ori_std[l]=np.std(this_EbIs)
        
        EI_base_input_per_ori_mean[l]=np.mean(this_IbEs)
        II_base_input_per_ori_mean[l]=np.mean(this_IbIs)
        EI_base_input_per_ori_std[l]=np.std(this_IbEs)
        II_base_input_per_ori_std[l]=np.std(this_IbIs)
        
        ET_base_input_per_ori_mean[l]=np.mean(this_TbEs)
        IT_base_input_per_ori_mean[l]=np.mean(this_TbIs)
        ET_base_input_per_ori_std[l]=np.std(this_TbEs)
        IT_base_input_per_ori_std[l]=np.std(this_TbIs)
        
        EF_opto_input_per_ori_mean[l]=np.mean(this_FoEs)
        IF_opto_input_per_ori_mean[l]=np.mean(this_FoIs)
        EF_opto_input_per_ori_std[l]=np.std(this_FoEs)
        IF_opto_input_per_ori_std[l]=np.std(this_FoIs)
        
        EE_opto_input_per_ori_mean[l]=np.mean(this_EoEs)
        IE_opto_input_per_ori_mean[l]=np.mean(this_EoIs)
        EE_opto_input_per_ori_std[l]=np.std(this_EoEs)
        IE_opto_input_per_ori_std[l]=np.std(this_EoIs)
        
        EI_opto_input_per_ori_mean[l]=np.mean(this_IoEs)
        II_opto_input_per_ori_mean[l]=np.mean(this_IoIs)
        EI_opto_input_per_ori_std[l]=np.std(this_IoEs)
        II_opto_input_per_ori_std[l]=np.std(this_IoIs)
        
        ET_opto_input_per_ori_mean[l]=np.mean(this_ToEs)
        IT_opto_input_per_ori_mean[l]=np.mean(this_ToIs)
        ET_opto_input_per_ori_std[l]=np.std(this_ToEs)
        IT_opto_input_per_ori_std[l]=np.std(this_ToIs)
        
        E_base_rate_per_ori_mean[l]=np.mean(this_rbEs)
        I_base_rate_per_ori_mean[l]=np.mean(this_rbIs)
        E_base_rate_per_ori_std[l]=np.std(this_rbEs)
        I_base_rate_per_ori_std[l]=np.std(this_rbIs)
        
        E_opto_rate_per_ori_mean[l]=np.mean(this_roEs)
        I_opto_rate_per_ori_mean[l]=np.mean(this_roIs)
        E_opto_rate_per_ori_std[l]=np.std(this_roEs)
        I_opto_rate_per_ori_std[l]=np.std(this_roIs)
        
        E_diff_rate_per_ori_mean[l]=np.mean(this_roEs-this_rbEs)
        I_diff_rate_per_ori_mean[l]=np.mean(this_roIs-this_rbIs)
        E_diff_rate_per_ori_std[l]=np.std(this_roEs-this_rbEs)
        I_diff_rate_per_ori_std[l]=np.std(this_roIs-this_rbIs)
        E_diff_rate_per_ori_cov[l]=np.cov(this_rbEs,this_roEs-this_rbEs)[0,1]
        I_diff_rate_per_ori_cov[l]=np.cov(this_rbIs,this_roIs-this_rbIs)[0,1]

    results_dict={}

    results_dict['EF_base_input_per_ori_mean']=EF_base_input_per_ori_mean
    results_dict['IF_base_input_per_ori_mean']=IF_base_input_per_ori_mean
    results_dict['EF_base_input_per_ori_std']=EF_base_input_per_ori_std
    results_dict['IF_base_input_per_ori_std']=IF_base_input_per_ori_std
    results_dict['EF_opto_input_per_ori_mean']=EF_opto_input_per_ori_mean
    results_dict['IF_opto_input_per_ori_mean']=IF_opto_input_per_ori_mean
    results_dict['EF_opto_input_per_ori_std']=EF_opto_input_per_ori_std
    results_dict['IF_opto_input_per_ori_std']=IF_opto_input_per_ori_std

    results_dict['EE_base_input_per_ori_mean']=EE_base_input_per_ori_mean
    results_dict['IE_base_input_per_ori_mean']=IE_base_input_per_ori_mean
    results_dict['EE_base_input_per_ori_std']=EE_base_input_per_ori_std
    results_dict['IE_base_input_per_ori_std']=IE_base_input_per_ori_std
    results_dict['EE_opto_input_per_ori_mean']=EE_opto_input_per_ori_mean
    results_dict['IE_opto_input_per_ori_mean']=IE_opto_input_per_ori_mean
    results_dict['EE_opto_input_per_ori_std']=EE_opto_input_per_ori_std
    results_dict['IE_opto_input_per_ori_std']=IE_opto_input_per_ori_std

    results_dict['EI_base_input_per_ori_mean']=EI_base_input_per_ori_mean
    results_dict['II_base_input_per_ori_mean']=II_base_input_per_ori_mean
    results_dict['EI_base_input_per_ori_std']=EI_base_input_per_ori_std
    results_dict['II_base_input_per_ori_std']=II_base_input_per_ori_std
    results_dict['EI_opto_input_per_ori_mean']=EI_opto_input_per_ori_mean
    results_dict['II_opto_input_per_ori_mean']=II_opto_input_per_ori_mean
    results_dict['EI_opto_input_per_ori_std']=EI_opto_input_per_ori_std
    results_dict['II_opto_input_per_ori_std']=II_opto_input_per_ori_std

    results_dict['ET_base_input_per_ori_mean']=ET_base_input_per_ori_mean
    results_dict['IT_base_input_per_ori_mean']=IT_base_input_per_ori_mean
    results_dict['ET_base_input_per_ori_std']=ET_base_input_per_ori_std
    results_dict['IT_base_input_per_ori_std']=IT_base_input_per_ori_std
    results_dict['ET_opto_input_per_ori_mean']=ET_opto_input_per_ori_mean
    results_dict['IT_opto_input_per_ori_mean']=IT_opto_input_per_ori_mean
    results_dict['ET_opto_input_per_ori_std']=ET_opto_input_per_ori_std
    results_dict['IT_opto_input_per_ori_std']=IT_opto_input_per_ori_std

    results_dict['E_base_rate_per_ori_mean']=E_base_rate_per_ori_mean
    results_dict['I_base_rate_per_ori_mean']=I_base_rate_per_ori_mean
    results_dict['E_base_rate_per_ori_std']=E_base_rate_per_ori_std
    results_dict['I_base_rate_per_ori_std']=I_base_rate_per_ori_std
    results_dict['E_opto_rate_per_ori_mean']=E_opto_rate_per_ori_mean
    results_dict['I_opto_rate_per_ori_mean']=I_opto_rate_per_ori_mean
    results_dict['E_opto_rate_per_ori_std']=E_opto_rate_per_ori_std
    results_dict['I_opto_rate_per_ori_std']=I_opto_rate_per_ori_std
    results_dict['E_diff_rate_per_ori_mean']=E_diff_rate_per_ori_mean
    results_dict['I_diff_rate_per_ori_mean']=I_diff_rate_per_ori_mean
    results_dict['E_diff_rate_per_ori_std']=E_diff_rate_per_ori_std
    results_dict['I_diff_rate_per_ori_std']=I_diff_rate_per_ori_std
    results_dict['E_diff_rate_per_ori_cov']=E_diff_rate_per_ori_cov
    results_dict['I_diff_rate_per_ori_cov']=I_diff_rate_per_ori_cov

    return results_dict
