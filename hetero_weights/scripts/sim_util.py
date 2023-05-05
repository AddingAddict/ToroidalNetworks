import inspect

import numpy as np

import ring_network as network
import integrate as integ

def sim_ring(params_dict,ri,T,mask_time,seeds,max_min=15):
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
        
        sol = integ.sim_dyn(ri,T,0.0,net.M_torch,net.H_torch,net.LAM_torch,net.E_cond,mult_tau=False)
        rates[seed_idx,0,:]=torch.mean(sol[mask_time],axis=0).numpy()
        
        sol = integ.sim_dyn(ri,T,params_dict['L'],net.M_torch,net.H_torch,net.LAM_torch,net.E_cond,mult_tau=False)
        rates[seed_idx,1,:]=torch.mean(sol[mask_time],axis=0).numpy()
        
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