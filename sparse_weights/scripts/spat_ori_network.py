import numpy as np
import torch

import network

class SpatOriNetwork(network.BaseNetwork):

    """
    Creates a network with n cell types on a discrete 2-torus representing two RF dimensions.
    If ori_type=='ring' then there is an addiontal toroidal dimension representing an orientation dimension.
    Otherise, we assign an orientation preference map depending on ori_type with Nori distinct orientations.
    The spatial grid covers Lrf degrees of RF in each direction. The ring covers Lori=180 degrees.
    """

    def __init__(self, seed=0, Nrf=16, Nori=8, n=None, NC=None, NX=None, Lrf=None, ori_type='ring'):
        if Lrf is None:
            self.Lrf = 60
        else:
            self.Lrf = Lrf
        self.Lori = 180

        self.Nrf = int(Nrf)
        self.Nori = int(Nori)

        self.ori_type = ori_type
        if self.ori_type=='ring':
            self.Nloc = self.Nrf**2*self.Nori
        elif self.ori_type in ('snp',):
            self.Nloc = self.Nrf**2
        else:
            raise Exception('ori_type not implemented')

        super().__init__(seed=seed, n=n, NC=NC, Nloc=self.Nloc, profile='gaussian', normalize_by_mean=False)

        self.set_XYZ()

    def set_XYZ(self):
        if self.ori_type=='ring':
            XYZ = np.array(np.unravel_index(np.arange(self.Nloc),(self.Nrf,self.Nrf,self.Nori))).astype(float)
            XYZ[0] *= self.Lrf/self.Nrf
            XYZ[1] *= self.Lrf/self.Nrf
            XYZ[2] *= self.Lori/self.Nori
            XYZ = np.repeat(XYZ,self.NT,axis=1)

            self.XY = XYZ[0:2].T
            self.Z = XYZ[2]
        elif self.ori_type=='snp':
            XY = np.array(np.unravel_index(np.arange(self.Nloc),(self.Nrf,self.Nrf))).astype(float)
            XY[0] *= self.Lrf/self.Nrf
            XY[1] *= self.Lrf/self.Nrf
            XY = np.repeat(XY,self.NT,axis=1)

            Z = np.mod(np.arange(self.Nloc),self.Nori).astype(float)
            self.rng.shuffle(Z)
            Z *= self.Lori/self.Nori
            Z = np.repeat(Z,self.NT)

            self.XY = XY.T
            self.Z = Z

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

    def get_rf_dist(self,vis_loc=None,byloc=False,return_mag=True):
        if vis_loc is None:
            vis_loc = np.zeros(2)
        if byloc:
            idxs = slice(0,self.N,self.NT)
        else:
            idxs = slice(0,self.N)
        rf_xdist = network.make_periodic(np.abs(self.XY[idxs,0] - vis_loc[0]),self.Lrf/2)
        rf_ydist = network.make_periodic(np.abs(self.XY[idxs,1] - vis_loc[1]),self.Lrf/2)
        if return_mag:
            return np.sqrt(rf_xdist**2+rf_ydist**2)
        else:
            return rf_xdist,rf_ydist

    def get_rf_diff(self,byloc=False,return_mag=True):
        if byloc:
            idxs = slice(0,self.N,self.NT)
        else:
            idxs = slice(0,self.N)
        rf_xdiff = network.make_periodic(np.abs(self.XY[idxs,0] - self.XY[idxs,0][:,None]),self.Lrf/2)
        rf_ydiff = network.make_periodic(np.abs(self.XY[idxs,1] - self.XY[idxs,0][:,None]),self.Lrf/2)
        if return_mag:
            return np.sqrt(rf_xdiff**2+rf_ydiff**2)
        else:
            return rf_xdiff,rf_ydiff

    def get_oriented_neurons(self,delta_ori=22.5,vis_ori=None):
        ori_dist = self.get_ori_dist(vis_ori)
        return np.where(ori_dist < delta_ori)

    def get_centered_neurons(self,delta_dist=15,vis_loc=None):
        rf_dist = self.get_rf_dist(vis_loc)
        return np.where(rf_dist < delta_dist)

    def generate_full_vector(self,Srf,Sori,kernel="nonnormgaussian",byloc=True):
        ori_dist = self.get_ori_dist(byloc=byloc)
        rf_xdist,rf_ydist = self.get_rf_dist(byloc=byloc,return_mag=False)
        full_vector = network.apply_kernel(ori_dist,Sori,self.Lori,kernel=kernel)
        full_vector *= network.apply_kernel(rf_xdist,Srf,self.Lrf,kernel=kernel)
        full_vector *= network.apply_kernel(rf_ydist,Srf,self.Lrf,kernel=kernel)
        return full_vector

    def generate_full_kernel(self,Srf,Sori,kernel="gaussian",byloc=True):
        ori_diff = self.get_ori_diff(byloc=byloc)
        rf_xdiff,rf_ydiff = self.get_rf_diff(byloc=byloc,return_mag=False)
        full_kernel = network.apply_kernel(ori_diff,Sori,self.Lori,self.Lori/self.Nori,kernel=kernel)
        full_kernel *= network.apply_kernel(rf_xdiff,Srf,self.Lrf,self.Lrf/self.Nrf,kernel=kernel)
        full_kernel *= network.apply_kernel(rf_ydiff,Srf,self.Lrf,self.Lrf/self.Nrf,kernel=kernel)
        return full_kernel

    def generate_full_rec_conn(self,WMat,VarMat,SrfMat,SoriMat,K,vanilla_or_not=False,return_mean_var=False):
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

                if vanilla_or_not=='vanilla' or vanilla_or_not==True:
                    W = np.ones((self.Nloc,self.Nloc))
                else:
                    W_aux = self.generate_full_kernel(SrfMat[pstC,preC],SoriMat[pstC,preC])
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

    def generate_full_ff_conn(self,WVec,VarVec,SrfVec,SoriVec,K,vanilla_or_not=False,return_mean_var=False):
        WKern = [None]*self.n
        W_mean_full = np.zeros((self.N,self.NX*self.Nloc),np.float32)
        W_var_full = np.zeros((self.N,self.NX*self.Nloc),np.float32)

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

    def generate_full_input(self,HVec,VarVec,SrfVec,SoriVec,vanilla_or_not=False,vis_loc=None,vis_ori=None):
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
                H = self.generate_full_vector(SrfVec[pstC],SoriVec[pstC])

            H_mean_full[pstC_all] = HVec[pstC]*np.repeat(H,NpstC)
            H_var_full[pstC_all] = VarVec[pstC]*np.repeat(H,NpstC)

        return H_mean_full,H_var_full

    def generate_M(self,W,SWrf,SWori,K,vanilla_or_not=False):
        C_full, W_mean_full,W_var_full = self.generate_full_rec_conn(W,np.zeros((self.n,self.n)),
            SWrf,SWori,K,vanilla_or_not,True)
        return C_full*(W_mean_full+np.random.normal(size=(self.N,self.N))*np.sqrt(W_var_full))

    def generate_MX(self,W,SWrf,SWori,K,vanilla_or_not=False):
        CX_full, WX_mean_full,WX_var_full = self.generate_full_ff_conn(WX,np.zeros(self.n),
            SWrfX,SWoriX,K,vanilla_or_not,True)
        return CX_full*(WX_mean_full+np.random.normal(size=(self.N,self.NX*self.Nloc))*np.sqrt(WX_var_full))

    def generate_H(self,H,SHrf,SHori,vanilla_or_not=False,vis_loc=None,vis_ori=None):
        H_mean_full,H_var_full = self.generate_full_input(H,np.zeros((self.n)),SHrf,SHori,vanilla_or_not,vis_loc,vis_ori)
        return H_mean_full+np.random.normal(size=(self.N))*np.sqrt(H_var_full)

    # def generate_disorder(self,W,SWrf,SWori,WX,SWrfX,SWoriX,K,vanilla_or_not=False):
    def generate_disorder(self,W,SWrf,SWori,H,SHrf,SHori,K,vanilla_or_not=False):
        self.M = self.generate_M(W,SWrf,SWori,K,vanilla_or_not)
        # self.MX = self.generate_MX(W,SWrf,SWori,K,vanilla_or_not)
        self.H = self.generate_H(H,SHrf,SHori,vanilla_or_not)

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


