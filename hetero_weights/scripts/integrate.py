import time
import numpy as np
from scipy.integrate import solve_ivp
import torch
from torchdiffeq import odeint, odeint_event

def sim_dyn(rc,T,L,M,H,LAM,E_all,I_all,mult_tau=False,max_min=7.5,stat_stop=True):
    LAS = LAM*L

    F=np.zeros_like(H)
    start = time.process_time()
    max_time = max_min*60
    timeout = False

    # This function computes the dynamics of the rate model
    if mult_tau:
        def ode_fn(t,R):
            MU=np.matmul(M,R)+H
            MU[E_all]=rc.tE*MU[E_all]
            MU[I_all]=rc.tI*MU[I_all]
            MU=MU+LAS
            F[E_all] =(-R[E_all]+rc.phiE(MU[E_all]))/rc.tE;
            F[I_all] =(-R[I_all]+rc.phiI(MU[I_all]))/rc.tI;
            return F
    else:
        def ode_fn(t,R):
            MU=np.matmul(M,R)+H+LAS
            F[E_all] =(-R[E_all]+rc.phiE(MU[E_all]))/rc.tE;
            F[I_all] =(-R[I_all]+rc.phiI(MU[I_all]))/rc.tI;
            return F

    # This function determines if the system is stationary or not
    def stat_event(t,R):
        meanF = np.mean(np.abs(F)/np.maximum(R,1e-1)) - 5e-3
        if meanF < 0: meanF = 0
        return meanF
    stat_event.terminal = True

    # This function forces the integration to stop after 15 minutes
    def time_event(t,R):
        int_time = (start + max_time) - time.process_time()
        if int_time < 0: int_time = 0
        return int_time
    time_event.terminal = True

    rates=np.zeros((len(H),len(T)));
    if stat_stop:
        sol = solve_ivp(ode_fn,[np.min(T),np.max(T)],rates[:,0], method='RK45', t_eval=T, events=[stat_event,time_event])
    else:
        sol = solve_ivp(ode_fn,[np.min(T),np.max(T)],rates[:,0], method='RK45', t_eval=T, events=[time_event])
    if sol.t.size < len(T):
        print("      Integration stopped after " + str(np.around(T[sol.t.size-1],2)) + "s of simulation time")
        if time.process_time() - start > max_time:
            print("            Integration reached time limit")
            timeout = True
        rates[:,0:sol.t.size] = sol.y
        rates[:,sol.t.size:] = sol.y[:,-1:]
    else:
        rates=sol.y
    
    return rates,timeout

def sim_dyn_tensor(rc,T,L,M,H,LAM,E_cond,mult_tau=False):
    MU = torch.zeros_like(H,dtype=torch.float32)
    F = torch.ones_like(H,dtype=torch.float32)
    LAS = LAM*L

    # This function computes the dynamics of the rate model
    if mult_tau:
        def ode_fn(t,R):
            MU=torch.matmul(M,R)#,out=MU)
            MU=torch.add(MU,H)#,out=MU)
            MU=torch.where(E_cond,rc.tE*MU,rc.tI*MU)#,out=MU)
            MU=torch.add(MU,LAS)#,out=MU)
            F=torch.where(E_cond,(-R+rc.phiE_tensor(MU))/rc.tE,(-R+rc.phiI_tensor(MU))/rc.tI)#,out=F)
            return F
    else:
        def ode_fn(t,R):
            MU=torch.matmul(M,R)#,out=MU)
            MU=torch.add(MU,H + LAS)#,out=MU)
            F=torch.where(E_cond,(-R+rc.phiE_tensor(MU))/rc.tE,(-R+rc.phiI_tensor(MU))/rc.tI)#,out=F)
            return F

    def event_fn(t,R):
        meanF = torch.mean(torch.abs(F)/torch.maximum(R,1e-1*torch.ones_like(R))) - 5e-3
        if meanF < 0: meanF = 0
        return torch.tensor(meanF)

    # rates = odeint(ode_fn,torch.zeros_like(H),T[[0,-1]],event_fn=event_fn)
    rates = odeint(ode_fn,torch.zeros_like(H,dtype=torch.float32),T)

    return torch.transpose(rates,0,1),False

