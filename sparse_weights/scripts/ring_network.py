import numpy as np
import torch

import base_network as network

class RingNetwork(network.BaseNetwork):

    """
    Creates a network with n cell types on a torus representing an orientation dimension.
    The ring covers Nori=180 degrees.
    """

    def __init__(self, seed=0, Nori=8, n=None, NC=None, NX=None):
        self.Lori = 180

        self.Nori = int(Nori)
        self.Nloc = self.Nori

        super().__init__(seed=seed, n=n, NC=NC, NX=NX, Nloc=self.Nloc, profile='wrapgauss', normalize_by_mean=False)

        self.set_Z()

    def set_Z(self):
        Z = np.array(np.unravel_index(np.arange(self.Nloc),(self.Nori,))).astype(float)
        Z *= self.Lori/self.Nori
        Z = np.repeat(Z,self.NT,axis=1)

        self.Z = Z[0]

    def get_ori_dist(self,vis_ori=None,byloc=False):
        if vis_ori is None:
            vis_ori = 0.0
        if byloc:
            idxs = slice(0,self.N,self.NT)
        else:
            idxs = slice(0,self.N)
        ori_dist = network.make_periodic(np.abs(self.Z[idxs] - vis_ori),self.Lori/2)
        return ori_dist

    def get_ori_diff(self,byloc=False):
        if byloc:
            idxs = slice(0,self.N,self.NT)
        else:
            idxs = slice(0,self.N)
        ori_diff = network.make_periodic(np.abs(self.Z[idxs] - self.Z[idxs][:,None]),self.Lori/2)
        return ori_diff

    def get_oriented_neurons(self,delta_ori=22.5,vis_ori=None):
        ori_dist = self.get_ori_dist(vis_ori)
        return np.where(ori_dist < delta_ori)

    def generate_full_vector(self,Sori,kernel="basesubwrapgauss",byloc=True,vis_ori=None):
        ori_dist = self.get_ori_dist(vis_ori=vis_ori,byloc=byloc)
        full_vector = network.apply_kernel(ori_dist,Sori,self.Lori,kernel=kernel)
        return full_vector

    def generate_full_kernel(self,Sori,kernel="wrapgauss",byloc=True):
        ori_diff = self.get_ori_diff(byloc=byloc)
        full_kernel = network.apply_kernel(ori_diff,Sori,self.Lori,self.Lori/self.Nori,kernel=kernel)
        full_kernel /= np.sum(full_kernel,1)[:,None]
        return full_kernel

    def generate_full_rec_conn(self,WMat,VarMat,SoriMat,K,baseprob=0,return_mean_var=False):
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

                if baseprob==1:
                    W = np.ones((self.Nloc,self.Nloc)) / self.Nloc
                else:
                    W_aux = baseprob/self.Nloc + (1-baseprob)*self.generate_full_kernel(SoriMat[pstC,preC])
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

    def generate_full_ff_conn(self,WVec,VarVec,SoriVec,K,baseprob=0,return_mean_var=False):
        WKern = [None]*self.n
        W_mean_full = np.zeros((self.N,self.NX*self.Nloc),np.float32)
        W_var_full = np.zeros((self.N,self.NX*self.Nloc),np.float32)

        for pstC in range(self.n):
            pstC_idxs = self.C_idxs[pstC]
            pstC_all = self.C_all[pstC]
            NpstC = self.NC[pstC]

            if np.isclose(WVec[pstC],0.): continue    # Skip if no connection of this type

            if baseprob==1:
                W = np.ones((self.Nloc,self.Nloc)) / self.Nloc
            else:
                W_aux = baseprob/self.Nloc + (1-baseprob)*self.generate_full_kernel(SoriVec[pstC])
                if self.normalize_by_mean:
                    W = W_aux/np.mean(W_aux)
                else:
                    W = W_aux

            WKern[pstC] = W

            for pst_loc in range(self.Nloc):
                pst_idxs = pstC_idxs[pst_loc]

                W_mean_full[pst_idxs,:] = WVec[pstC]
                W_var_full[pst_idxs,:] = VarVec[pstC]

        C_full = self.generate_sparse_ff_conn(WKern=WKern,K=K)

        if return_mean_var:
            return C_full, W_mean_full,W_var_full
        else:
            return C_full

    def generate_full_input(self,HVec,VarVec,SoriVec,baseinp=0,vis_ori=None):
        H_mean_full = np.zeros((self.N),np.float32)
        H_var_full = np.zeros((self.N),np.float32)

        for pstC in range(self.n):
            pstC_idxs = self.C_idxs[pstC]
            pstC_all = self.C_all[pstC]
            NpstC = self.NC[pstC]

            if np.isclose(HVec[pstC],0.): continue    # Skip if no connection of this type

            if baseinp==1:
                H = np.ones((self.Nloc))
            else:
                H = baseinp + (1-baseinp)*self.generate_full_vector(SoriVec[pstC],vis_ori=vis_ori)

            H_mean_full[pstC_all] = HVec[pstC]*np.repeat(H,NpstC)
            H_var_full[pstC_all] = VarVec[pstC]*np.repeat(H,NpstC)

        return H_mean_full,H_var_full

    def generate_M(self,W,SWori,K,baseprob=0):
        C_full, W_mean_full,W_var_full = self.generate_full_rec_conn(W,np.zeros((self.n,self.n)),
            SWori,K,baseprob,True)
        return C_full*(W_mean_full+np.random.normal(size=(self.N,self.N))*np.sqrt(W_var_full))

    def generate_MX(self,WX,SWoriX,K,baseprob=0):
        CX_full, WX_mean_full,WX_var_full = self.generate_full_ff_conn(WX,np.zeros(self.n),
            SWoriX,K,baseprob,True)
        return CX_full*(WX_mean_full+np.random.normal(size=(self.N,self.NX*self.Nloc))*np.sqrt(WX_var_full))

    def generate_H(self,H,SHori,baseinp=0,vis_ori=None):
        H_mean_full,H_var_full = self.generate_full_input(H,np.zeros((self.n)),SHori,baseinp,vis_ori)
        return H_mean_full+np.random.normal(size=(self.N))*np.sqrt(H_var_full)

    def generate_disorder(self,W,SWori,H,SHori,K,baseinp=0,baseprob=0,vis_ori=None):
        self.M = self.generate_M(W,SWori,K,baseprob)
        # self.MX = self.generate_MX(W,SWori,K,baseprob)
        self.H = self.generate_H(H,SHori,baseinp,vis_ori=vis_ori)
        
    def generate_disorder_two_layer(self,W,SWori,K,WX,SWoriX,KX,baseprob=0):
        self.M = self.generate_M(W,SWori,K,baseprob)
        self.MX = self.generate_MX(WX,SWoriX,KX,baseprob)

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

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using",device)

        for cidx in range(self.n):
            self.C_conds[cidx] = self.C_conds[cidx].to(device)
        self.M_torch = self.M_torch.to(device)
        # self.MX_torch = self.MX_torch.to(device)
        self.H_torch = self.H_torch.to(device)
