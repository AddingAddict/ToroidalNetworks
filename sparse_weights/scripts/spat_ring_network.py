from numba import jit

import numpy as np
import torch
from scipy.special import erf, erfi, i0
from scipy.integrate import solve_ivp
from scipy.linalg import circulant
from mpmath import fp

jtheta = np.vectorize(fp.jtheta, 'D')

def make_vector(x,S,L,N,kernel='nonnormgaussian'):
    if kernel == 'gaussian':
        out = np.exp(-x**2/(2*S**2))/np.sqrt(2*np.pi)/S
    elif kernel == 'nonnormgaussian':
        out = np.exp(-x**2/(2*S**2))
    elif kernel == 'exponential':
        out = np.exp(-np.abs(x)/S)/S
    elif kernel == 'vonmisses':
        x_rad=x/(L/2)*np.pi
        S_rad=S/(L/2)*np.pi
        d_rad=np.pi/(L/2)
        out = np.exp(np.cos(x_rad)/S_rad)/(2*np.pi*i0(1/S_rad))*d_rad
    elif kernel == 'jtheta':
        x_rad=x/(L/2)*np.pi
        S_rad=S/(L/2)*np.pi
        d_rad=np.pi/(L/2)
        out = np.real(jtheta(3,x_rad/2,np.exp(-S_rad**2/2)))/(2*np.pi)*d_rad
    elif kernel == 'nonnormjtheta':
        x_rad=x/(L/2)*np.pi
        S_rad=S/(L/2)*np.pi
        out = np.real(jtheta(3,dori_rad/2,np.exp(-S_rad**2/2)))/np.real(jtheta(3,0,np.exp(-S_rad**2/2)))
    return out

def make_kernel(x,S,L,N,kernel='gaussian'):
    if kernel == 'gaussian':
        out = np.exp(-x**2/(2*S**2))/np.sqrt(2*np.pi)/S
    elif kernel == 'nonnormgaussian':
        out = np.exp(-x**2/(2*S**2))
    elif kernel == 'exponential':
        out = np.exp(-np.abs(x)/S)
    elif kernel == 'vonmisses':
        x_rad=x/(L/2)*np.pi
        S_rad=S/(L/2)*np.pi
        d_rad=np.pi/(L/2)
        out = np.exp(np.cos(x_rad)/S_rad)/(2*np.pi*i0(1/S_rad))*d_rad
    elif kernel == 'jtheta':
        x_rad=x/(L/2)*np.pi
        S_rad=S/(L/2)*np.pi
        d_rad=np.pi/(L/2)
        out = np.real(jtheta(3,x_rad/2,np.exp(-S_rad**2/2)))/(2*np.pi)*d_rad
    elif kernel == 'nonnormjtheta':
        x_rad=x/(L/2)*np.pi
        S_rad=S/(L/2)*np.pi
        out = np.real(jtheta(3,dori_rad/2,np.exp(-S_rad**2/2)))/np.real(jtheta(3,0,np.exp(-S_rad**2/2)))
    return out*L/N

class network(object):

    """
    Creates a network with n cell types on a 3-torus representing two RF dimensions and an orientation dimension.
    The spatial grid is periodic, covering Lrf degrees of RF in each direction. The ring covers Nori=180 degrees.
    """

    def __init__(self, seed=0, Nrf=16, Nori=8, n=None, NC=None, NX=None, Lrf=None):
        self.rng = np.random.default_rng(seed=seed)

        if Lrf is None:
            self.Lrf = 60
        else:
            self.Lrf = Lrf
        self.Lori = 180

        self.Nrf = int(Nrf)
        self.Nori = int(Nori)

        if NC is None:
            if n is None:
                self.NC = np.ones(1,dtype=int)
                self.n = 1
            else:
                self.NC = np.ones(n,dtype=int)
                self.n = int(n)
        elif np.isscalar(NC):
            if n is None:
                self.NC = np.ones(1)
                self.n = 1
            else:
                self.NC = NC*np.ones(n,dtype=int)
                self.n = int(n)
        else:
            if n is None:
                self.NC = np.array(NC,dtype=int)
                self.n = len(self.NC)
            else:
                self.NC = np.array(NC,dtype=int)
                self.n = int(n)
                if self.n != len(self.NC): raise Exception("Length of NC and n do not match")

        if NX is None:
            self.NX = int(self.NC[0])
        else:
            self.NX = int(NX)

        self.NT = np.sum(self.NC)
        self.Nloc = self.Nrf**2*self.Nori

        self.C_idxs = []
        self.C_all = []
        prev_NC = 0
        for cidx in range(self.n):
            prev_NC += 0 if cidx == 0 else self.NC[cidx-1]
            this_NC = self.NC[cidx]
            this_C_idxs = [slice(int(loc*self.NT+prev_NC),int(loc*self.NT+prev_NC+this_NC)) for loc in range(self.Nloc)]
            self.C_idxs.append(this_C_idxs)

            this_C_all = np.zeros(0,np.int8)
            for loc in range(self.Nloc):
                this_C_loc_idxs = np.arange(this_C_idxs[loc].start,this_C_idxs[loc].stop,dtype=int)
                this_C_all = np.append(this_C_all,this_C_loc_idxs)
            self.C_all.append(this_C_all)
        self.X_idxs = [slice(loc*self.NX,(loc+1)*self.NX) for loc in range(self.Nloc)]
        self.X_all = np.arange(self.NX*self.Nloc)
        
        # Total number of neurons
        self.N = self.Nloc*self.NT

        self.set_XYZ()

        self.profile = 'gaussian'
        self.normalize_by_mean = False

    def set_XYZ(self):
        XYZ = np.array(np.unravel_index(np.arange(self.Nloc),(self.Nrf,self.Nrf,self.Nori))).astype(float)
        XYZ[0] *= self.Lrf/self.Nrf
        XYZ[1] *= self.Lrf/self.Nrf
        XYZ[2] *= self.Lori/self.Nori
        XYZ = np.repeat(XYZ,self.NT,axis=1)

        self.XY = XYZ[0:2].T
        self.Z = XYZ[2]
        
    def make_periodic(self,vec_in,half_period):
        vec_out = np.copy(vec_in);
        vec_out[vec_out >  half_period] =  2*half_period-vec_out[vec_out >  half_period];
        vec_out[vec_out < -half_period] = -2*half_period-vec_out[vec_out < -half_period];
        return vec_out

    def set_seed(self,seed_con):
        np.random.seed(seed_con)

    def get_ori_dist(self,vis_ori=None,byloc=False):
        if vis_ori is None:
            vis_ori = 0.0
        if byloc:
            idxs = slice(0,self.N,self.NT)
        else:
            idxs = slice(0,self.N)
        ori_dist = self.make_periodic(np.abs(self.Z[idxs] - vis_ori),self.Lori/2)
        return ori_dist

    def get_ori_diff(self,byloc=False):
        if byloc:
            idxs = slice(0,self.N,self.NT)
        else:
            idxs = slice(0,self.N)
        ori_diff = self.make_periodic(np.abs(self.Z[idxs] - self.Z[idxs][:,None]),self.Lori/2)
        return ori_diff

    def get_rf_dist(self,vis_loc=None,byloc=False,return_mag=True):
        if vis_loc is None:
            vis_loc = np.zeros(2)
        if byloc:
            idxs = slice(0,self.N,self.NT)
        else:
            idxs = slice(0,self.N)
        rf_xdist = self.make_periodic(np.abs(self.XY[idxs,0] - vis_loc[0]),self.Lrf/2)
        rf_ydist = self.make_periodic(np.abs(self.XY[idxs,1] - vis_loc[1]),self.Lrf/2)
        if return_mag:
            return np.sqrt(rf_xdist**2+rf_ydist**2)
        else:
            return rf_xdist,rf_ydist

    def get_rf_diff(self,byloc=False,return_mag=True):
        if byloc:
            idxs = slice(0,self.N,self.NT)
        else:
            idxs = slice(0,self.N)
        rf_xdiff = self.make_periodic(np.abs(self.XY[idxs,0] - self.XY[idxs,0][:,None]),self.Lrf/2)
        rf_ydiff = self.make_periodic(np.abs(self.XY[idxs,1] - self.XY[idxs,0][:,None]),self.Lrf/2)
        if return_mag:
            return np.sqrt(rf_xdiff**2+rf_ydiff**2)
        else:
            return rf_xdiff,rf_ydiff

    def get_oriented_neurons(self,delta_ori=15,vis_ori=None):
        ori_dist = self.get_ori_dist(vis_ori)
        return np.where(ori_dist < delta_ori)

    def get_centered_neurons(self,delta_dist=15,vis_loc=None):
        rf_dist = self.get_rf_dist(vis_loc)
        return np.where(rf_dist < delta_dist)

    def generate_full_vector(self,Srf,Sori,kernel="gaussian",byloc=True):
        ori_dist = self.get_ori_dist(byloc=byloc)
        rf_xdist,rf_ydist = self.get_rf_dist(byloc=byloc,return_mag=False)
        full_vector = make_vector(ori_dist,Sori,self.Lori,self.Nori,kernel=kernel)
        full_vector *= make_vector(rf_xdist,Srf,self.Lrf,self.Nrf,kernel=kernel)
        full_vector *= make_vector(rf_ydist,Srf,self.Lrf,self.Nrf,kernel=kernel)
        return full_vector

    def generate_full_kernel(self,Srf,Sori,kernel="gaussian",byloc=True):
        ori_diff = self.get_ori_diff(byloc=byloc)
        rf_xdiff,rf_ydiff = self.get_rf_diff(byloc=byloc,return_mag=False)
        full_kernel = make_kernel(ori_diff,Sori,self.Lori,self.Nori,kernel=kernel)
        full_kernel *= make_kernel(rf_xdiff,Srf,self.Lrf,self.Nrf,kernel=kernel)
        full_kernel *= make_kernel(rf_ydiff,Srf,self.Lrf,self.Nrf,kernel=kernel)
        return full_kernel

    def generate_full_rec_conn(self,WMat,VarMat,SrfMat,SoriMat,K,vanilla_or_not=False,return_mean_var=False):
        C_full = np.zeros((self.N,self.N),np.float32)
        W_mean_full = np.zeros((self.N,self.N),np.float32)
        W_var_full = np.zeros((self.N,self.N),np.float32)

        for pstC in range(self.n):
            pstC_idxs = self.C_idxs[pstC]
            pstC_all = self.C_all[pstC]
            NpstC = self.NC[pstC]
            for preC in range(self.n):
                preC_idxs = self.C_idxs[preC]
                preC_all = self.C_all[preC]
                NpreC = self.NC[preC]

                if np.isclose(WMat[pstC,preC],0.): continue    # Skip if no connection of this type

                if vanilla_or_not=='vanilla' or vanilla_or_not==True:
                    W = np.ones((self.Nloc,self.Nloc))
                else:
                    W_aux = self.generate_full_kernel(SrfMat[pstC,preC],SoriMat[pstC,preC])
                    if self.normalize_by_mean:
                        W = W_aux/np.mean(W_aux)
                    else:
                        W = W_aux
                ps = np.fmax(K/self.NC[0] * W,1e-12)
                if np.any(ps > 1):
                    raise Exception("Error: p > 1, please decrease K or increase NC")

                for pst_loc in range(self.Nloc):
                    pst_idxs = pstC_idxs[pst_loc]
                    for pre_loc in range(self.Nloc):
                        p = ps[pst_loc,pre_loc]
                        pre_idxs = preC_idxs[pre_loc]
                        C_full[pst_idxs,pre_idxs] = self.rng.binomial(1,p,size=(NpstC,NpreC))

                    W_mean_full[pst_idxs,preC_all] = WMat[pstC,preC]
                    W_var_full[pst_idxs,preC_all] = VarMat[pstC,preC]

        if return_mean_var:
            return C_full, W_mean_full,W_var_full
        else:
            return C_full

    def generate_full_ff_conn(self,WVec,VarVec,SrfVec,SoriVec,K,vanilla_or_not=False,return_mean_var=False):
        C_full = np.zeros((self.N,self.NX*self.Nloc),np.float32)
        W_mean_full = np.zeros((self.N,self.NX*self.Nloc),np.float32)
        W_var_full = np.zeros((self.N,self.NX*self.Nloc),np.float32)

        preC_idxs = self.X_idxs
        preC_all = self.X_all
        NpreC = self.NX

        for pstC in range(self.n):
            pstC_idxs = self.C_idxs[pstC]
            pstC_all = self.C_all[pstC]
            NpstC = self.NC[pstC]

            if np.isclose(WVec[pstC],0.): continue    # Skip if no connection of this type

            if vanilla_or_not=='vanilla' or vanilla_or_not==True:
                W = np.ones((self.Nloc,self.Nloc))
            else:
                W_aux = self.generate_full_kernel(SrfVec[pstC],SoriVec[pstC])
                if self.normalize_by_mean:
                    W = W_aux/np.mean(W_aux)
                else:
                    W = W_aux
            ps = np.fmax(K/self.NX * W,1e-12)
            if np.any(ps > 1):
                raise Exception("Error: p > 1, please decrease K or increase NC")

            for pst_loc in range(self.Nloc):
                pst_idxs = pstC_idxs[pst_loc]
                for pre_loc in range(self.Nloc):
                    p = ps[pst_loc,pre_loc]
                    pre_idxs = preC_idxs[pre_loc]
                    C_full[pst_idxs,pre_idxs] = self.rng.binomial(1,p,size=(NpstC,NpreC))

                W_mean_full[pst_idxs,preC_all] = WVec[pstC]
                W_var_full[pst_idxs,preC_all] = VarVec[pstC]

        if return_mean_var:
            return C_full, W_mean_full,W_var_full
        else:
            return C_full

    def generate_full_input(self,HVec,VarVec,SrfVec,SoriVec,vanilla_or_not=False):
        H_mean_full = np.zeros((self.N),np.float32)
        H_var_full = np.zeros((self.N),np.float32)

        for pstC in range(self.n):
            pstC_idxs = self.C_idxs[pstC]
            pstC_all = self.C_all[pstC]
            NpstC = self.NC[pstC]

            if np.isclose(HVec[pstC],0.): continue    # Skip if no connection of this type

            if vanilla_or_not=='vanilla' or vanilla_or_not==True:
                H = np.ones((self.Nloc))
            else:
                H_aux = self.generate_full_vector(SrfVec[pstC],SoriVec[pstC])

            H_mean_full[pstC_all] = HVec[pstC]*np.repeat(H_aux,NpstC)
            H_var_full[pstC_all] = VarVec[pstC]*np.repeat(H_aux,NpstC)

        return H_mean_full,H_var_full

    # def generate_disorder(self,W,SWrf,SWori,WX,SWrfX,SWoriX,K,vanilla_or_not=False):
    def generate_disorder(self,W,SWrf,SWori,H,SHrf,SHori,K,vanilla_or_not=False):
        C_full, W_mean_full,W_var_full = self.generate_full_rec_conn(W,np.zeros((self.n,self.n)),
            SWrf,SWori,K,vanilla_or_not,True)
        self.M = C_full*(W_mean_full+np.random.normal(size=(self.N,self.N))*np.sqrt(W_var_full))

        # CX_full, WX_mean_full,WX_var_full = self.generate_full_ff_conn(WX,np.zeros(self.n),
        #     SWrfX,SWoriX,K,vanilla_or_not,True)
        # self.MX = CX_full*(WX_mean_full+np.random.normal(size=(self.N,self.NX*self.Nloc))*np.sqrt(WX_var_full))

        H_mean_full,H_var_full = self.generate_full_input(H,np.zeros((self.n)),SHrf,SHori,vanilla_or_not)
        self.H = H_mean_full+np.random.normal(size=(self.N))*np.sqrt(H_var_full)

    def generate_tensors(self):
        self.C_conds = []
        for cidx in range(self.n):
            this_C_cond = torch.zeros(self.N,dtype=torch.bool)
            this_C_cond = this_C_cond.scatter(0,torch.from_numpy(self.C_all[cidx]),
                torch.ones(self.C_all[cidx].size,dtype=torch.bool))
            self.C_conds.append(this_C_cond)

        self.M_torch = torch.from_numpy(self.M)
        # self.MX_torch = torch.from_numpy(self.MX)
        self.H_torch = torch.from_numpy(self.H)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using",device)

        for cidx in range(self.n):
            self.C_conds[cidx] = self.C_conds[cidx].to(device)
        self.M_torch = self.M_torch.to(device)
        # self.MX_torch = self.MX_torch.to(device)
        self.H_torch = self.H_torch.to(device)

