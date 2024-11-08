import numpy as np
import torch

import base_network as network

class SpatNetwork(network.BaseNetwork):

    """
    Creates a network with n cell types on a discrete d-torus representing d RF dimensions.
    The spatial grid covers Lrf degrees of RF in each direction.
    """

    def __init__(self, seed=0, Nrf=16, Nori=8, n=None, NC=None, NX=None, Lrf=None, Nd=2):
        if Lrf is None:
            self.Lrf = 60
        else:
            self.Lrf = Lrf
        self.Lori = 180

        self.Nrf = int(Nrf)
        self.Nd = int(Nd)
        self.Nloc = self.Nrf**self.Nd

        super().__init__(seed=seed, n=n, NC=NC, Nloc=self.Nloc, profile='gaussian', normalize_by_mean=False)

        self.set_X()

    def set_X(self):
        X = np.array(np.unravel_index(np.arange(self.Nloc),[self.Nrf,]*self.Nd)).astype(float)
        X *= self.Lrf/self.Nrf
        X = np.repeat(X,self.NT,axis=1)

        self.X = X.T

    def get_rf_dist(self,vis_loc=None,byloc=False,return_mag=True):
        if vis_loc is None:
            vis_loc = np.zeros(self.Nd)
        if byloc:
            idxs = slice(0,self.N,self.NT)
        else:
            idxs = slice(0,self.N)
        rf_dists = network.make_periodic(np.abs(self.X[idxs,:] - vis_loc[None,:]),self.Lrf/2)
        if return_mag:
            return np.sqrt(np.sum(rf_dists**2),-1)
        else:
            return rf_dists

    def get_rf_diff(self,byloc=False,return_mag=True):
        if byloc:
            idxs = slice(0,self.N,self.NT)
        else:
            idxs = slice(0,self.N)
        rf_diffs = network.make_periodic(np.abs(self.X[idxs,None,:] - self.X[None,idxs,:]),self.Lrf/2)
        if return_mag:
            return np.sqrt(np.sum(rf_diffs**2),-1)
        else:
            return rf_diffs

    def get_centered_neurons(self,delta_dist=15,vis_loc=None):
        rf_dist = self.get_rf_dist(vis_loc)
        return np.where(rf_dist < delta_dist)

    def generate_full_vector(self,Srf,kernel="nonnormgaussian",byloc=True):
        rf_dists = self.get_rf_dist(byloc=byloc,return_mag=False)
        full_vector = network.apply_kernel(rf_dists[:,0],Srf,self.Lrf,kernel=kernel)
        for i in range(1,self.Nd):
            full_vector *= network.apply_kernel(rf_dists[:,i],Srf,self.Lrf,kernel=kernel)
        return full_vector

    def generate_full_kernel(self,Srf,kernel="gaussian",byloc=True):
        rf_diffs = self.get_rf_diff(byloc=byloc,return_mag=False)
        full_kernel = network.apply_kernel(rf_diffs[:,:,0],Srf,self.Lrf,self.Lrf/self.Nrf,kernel=kernel)
        for i in range(1,self.Nd):
            full_kernel *= network.apply_kernel(rf_diffs[:,:,1],Srf,self.Lrf,self.Lrf/self.Nrf,kernel=kernel)
        full_kernel /= np.sum(full_kernel,1)[:,None]
        return full_kernel

    def generate_full_rec_conn(self,WMat,VarMat,SrfMat,K,basefrac=0,return_mean_var=False):
        WKern = [[None]*self.n]*self.n
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

                if basefrac==1:
                    W = np.ones((self.Nloc,self.Nloc)) / self.Nloc
                else:
                    W_aux = basefrac/self.Nloc + (1-basefrac)*self.generate_full_kernel(SrfMat[pstC,preC])
                    if self.normalize_by_mean:
                        W = W_aux/np.mean(W_aux)
                    else:
                        W = W_aux

                WKern[pstC][preC] = W

                for pst_loc in range(self.Nloc):
                    pst_idxs = pstC_idxs[pst_loc]

                    W_mean_full[pst_idxs,preC_all] = WMat[pstC,preC]
                    W_var_full[pst_idxs,preC_all] = VarMat[pstC,preC]

        C_full = self.generate_sparse_rec_conn(WKern=WKern,K=K)

        if return_mean_var:
            return C_full, W_mean_full,W_var_full
        else:
            return C_full

    def generate_full_ff_conn(self,WVec,VarVec,SrfVec,K,basefrac=0,return_mean_var=False):
        WKern = [None]*self.n
        W_mean_full = np.zeros((self.N,self.NX*self.Nloc),np.float32)
        W_var_full = np.zeros((self.N,self.NX*self.Nloc),np.float32)

        for pstC in range(self.n):
            pstC_idxs = self.C_idxs[pstC]
            pstC_all = self.C_all[pstC]
            NpstC = self.NC[pstC]

            if np.isclose(WVec[pstC],0.): continue    # Skip if no connection of this type

            if basefrac==1:
                W = np.ones((self.Nloc,self.Nloc)) / self.Nloc
            else:
                W_aux = basefrac/self.Nloc + (1-basefrac)*self.generate_full_kernel(SrfVec[pstC])
                if self.normalize_by_mean:
                    W = W_aux/np.mean(W_aux)
                else:
                    W = W_aux

            WKern[pstC] = W

            for pst_loc in range(self.Nloc):
                pst_idxs = pstC_idxs[pst_loc]

                W_mean_full[pst_idxs,preC_all] = WVec[pstC]
                W_var_full[pst_idxs,preC_all] = VarVec[pstC]

        C_full = self.generate_sparse_rec_conn(WKern=WKern,K=K)

        if return_mean_var:
            return C_full, W_mean_full,W_var_full
        else:
            return C_full

    def generate_full_input(self,HVec,VarVec,SrfVec,basefrac=0,vis_loc=None,vis_ori=None):
        H_mean_full = np.zeros((self.N),np.float32)
        H_var_full = np.zeros((self.N),np.float32)

        for pstC in range(self.n):
            pstC_idxs = self.C_idxs[pstC]
            pstC_all = self.C_all[pstC]
            NpstC = self.NC[pstC]

            if np.isclose(HVec[pstC],0.): continue    # Skip if no connection of this type

            if basefrac==1:
                H = np.ones((self.Nloc))
            else:
                H = basefrac + (1-basefrac)*self.generate_full_vector(SrfVec[pstC])

            H_mean_full[pstC_all] = HVec[pstC]*np.repeat(H,NpstC)
            H_var_full[pstC_all] = VarVec[pstC]*np.repeat(H,NpstC)

        return H_mean_full,H_var_full

    def generate_M(self,W,SWrf,K,basefrac=0):
        C_full, W_mean_full,W_var_full = self.generate_full_rec_conn(W,np.zeros((self.n,self.n)),
            SWrf,K,basefrac,True)
        return C_full*(W_mean_full+np.random.normal(size=(self.N,self.N))*np.sqrt(W_var_full))

    def generate_MX(self,WX,SWrfX,K,basefrac=0):
        CX_full, WX_mean_full,WX_var_full = self.generate_full_ff_conn(WX,np.zeros(self.n),
            SWrfX,K,basefrac,True)
        return CX_full*(WX_mean_full+np.random.normal(size=(self.N,self.NX*self.Nloc))*np.sqrt(WX_var_full))

    def generate_H(self,H,SHrf,basefrac=0,vis_loc=None,vis_ori=None):
        H_mean_full,H_var_full = self.generate_full_input(H,np.zeros((self.n)),SHrf,basefrac,vis_loc,vis_ori)
        return H_mean_full+np.random.normal(size=(self.N))*np.sqrt(H_var_full)

    # def generate_disorder(self,W,SWrf,SWori,WX,SWrfX,SWoriX,K,basefrac=0):
    def generate_disorder(self,W,SWrf,H,SHrf,K,basefrac=0):
        self.M = self.generate_M(W,SWrf,ori,K,basefrac)
        # self.MX = self.generate_MX(W,SWrf,K,basefrac)
        self.H = self.generate_H(H,SHrf,basefrac)

    def generate_tensors(self):
        self.C_conds = []
        for cidx in range(self.n):
            this_C_cond = torch.zeros(self.N,dtype=torch.bool)
            this_C_cond = this_C_cond.scatter(0,torch.from_numpy(self.C_all[cidx]),
                torch.ones(self.C_all[cidx].size,dtype=torch.bool))
            self.C_conds.append(this_C_cond)

        self.M_torch = torch.from_numpy(self.M.astype(dtype=np.float32))
        # self.MX_torch = torch.from_numpy(self.MX.astype(dtype=np.float32))
        self.H_torch = torch.from_numpy(self.H.astype(dtype=np.float32))
        
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        print("Using",device)
        print("Using",device)

        for cidx in range(self.n):
            self.C_conds[cidx] = self.C_conds[cidx].to(device)
        self.M_torch = self.M_torch.to(device)
        # self.MX_torch = self.MX_torch.to(device)
        self.H_torch = self.H_torch.to(device)
