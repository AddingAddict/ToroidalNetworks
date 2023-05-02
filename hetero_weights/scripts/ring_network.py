import numpy as np
import tensorflow as tf
from scipy.special import erf, erfi
from scipy.integrate import solve_ivp
from scipy.linalg import circulant
from mpmath import fp

jtheta = np.vectorize(fp.jtheta, 'D')

#test xcodeproj
    
class network(object):
    """

    Parameters
    ----------
    seed_con : int
        Random seed
    n : int
        Number of cell types
    NE : int
        Number of excitatory neurons per location
    ori_map : 2darray
        Orientation preference map
    dl : float
        Length of the entire 2D grid
    Sl : 1darray
        2-vector with spatial connection probability widths
    Sori : 1darray
        2-vector with orientation connection probability widths
        
    """    

    def __init__(self, seed_con=0, n=2, Nl=60, NE=50, SW=None, SvarW=None, SH=None, SvarH=None):
        self.seed_con = seed_con

        # Set external seed
        np.random.seed(seed_con)

        self.Nl = Nl
        self.NE = NE
        self.NI = NE
        self.NT = self.NE+self.NI
        self.Nloc = self.Nl

        self.E_idxs = [slice(loc*self.NT,loc*self.NT+self.NE) for loc in range(self.Nloc)]
        self.I_idxs = [slice(loc*self.NT+self.NE,(loc+1)*self.NT) for loc in range(self.Nloc)]
        self.E_all = np.zeros(0,np.int8)
        self.I_all = np.zeros(0,np.int8)
        self.ori_map = np.zeros((self.Nl,self.NT))
        for loc in range(self.Nloc):
            Elocinds = np.arange(self.E_idxs[loc].start,self.E_idxs[loc].stop)
            Ilocinds = np.arange(self.I_idxs[loc].start,self.I_idxs[loc].stop)
            self.E_all = np.append(self.E_all,Elocinds)
            self.I_all = np.append(self.I_all,Ilocinds)
            self.ori_map[loc,:] = loc*180/self.Nloc
        
        # Total number of neurons
        self.N = self.ori_map.size
        self.Z = self.ori_map.flatten()

        self.E_cond = tf.zeros(self.N,dtype=tf.bool)
        self.E_cond = tf.tensor_scatter_nd_update(self.E_cond,self.E_all[:,None],tf.ones_like(self.E_all,dtype=tf.bool))

        self.n = n

        self.dx = 180 / self.Nl
        self.SW = SW
        self.SvarW = SvarW
        self.SH = SH
        self.SvarH = SvarH

        self.profile = 'gaussian'

    def set_seed(self,seed_con):
        np.random.seed(seed_con)
        
    def get_center_orientation(self):
        return 0.0

    def get_oriented_neurons(self,delta_ori=15,grating_orientation=None):
        Tuning_vec = self.ori_map
        if grating_orientation==None:
            grating_orientation = self.get_center_orientation()
        Tuning_vec = self.make_periodic(np.abs(Tuning_vec - grating_orientation),90)
        lb = -delta_ori
        ub = +delta_ori

        if 0 <= lb and ub <= 180:    # both within
            this_neurons, = np.where(np.logical_and(lb<Tuning_vec, Tuning_vec<ub))
            
        elif 0 > lb and ub <= 180:  # lb is negative
            true_lb = np.mod(lb,180)
            this_neurons, = np.where(np.logical_or(true_lb<Tuning_vec, Tuning_vec<ub))
            
        elif 0 <= lb and 180 < ub: # ub is bigger than 180
            true_ub = np.mod(ub,180)
            this_neurons, = np.where(np.logical_or(lb<Tuning_vec, Tuning_vec<true_ub))
            
        elif 0 > lb and 180 < ub: # all oris
            true_lb = np.mod(lb,180)
            true_ub = np.mod(ub,180)
            this_neurons, = np.where(np.logical_and(true_lb<Tuning_vec, Tuning_vec<true_ub))
        else:
            print(lb,ub)
        return this_neurons

    def get_neurons_at_given_ori_distance_to_grating(self,diff_width=5, degs_away_from_center=15,
            grating_orientation=None, signed=False):

        Tuning_vec = self.ori_map
        if grating_orientation is None:
            grating_orientation = self.get_center_orientation()
        if signed:
            Tuning_vec = self.make_periodic(Tuning_vec - grating_orientation,90)
        else:
            Tuning_vec = self.make_periodic(np.abs(Tuning_vec - grating_orientation),90)

        lb = degs_away_from_center-diff_width
        ub = degs_away_from_center+diff_width


        if 0 <= lb and ub <= 180:    # both within
            this_neurons=np.logical_and(lb<Tuning_vec, Tuning_vec<ub)

        elif 0 > lb and ub <= 180:  # lb is negative
            true_lb = np.mod(lb,180)
            this_neurons=np.logical_or(true_lb<Tuning_vec, Tuning_vec<ub)

        elif 0 <= lb and 180 < ub: # ub is bigger than 180
            true_ub = np.mod(ub,180)
            this_neurons=np.logical_or(lb<Tuning_vec, Tuning_vec<true_ub)

        elif 0 > lb and 180 < ub: # all oris
            true_lb = np.mod(lb,180)
            true_ub = np.mod(ub,180)
            this_neurons=np.logical_and(true_lb<Tuning_vec, Tuning_vec<true_ub)

        return this_neurons
        
    def make_periodic(self,vec_in,half_period):
        vec_out = np.copy(vec_in);
        vec_out[vec_out >  half_period] =  2*half_period-vec_out[vec_out >  half_period];
        vec_out[vec_out < -half_period] = -2*half_period-vec_out[vec_out < -half_period];
        return vec_out

    def make_vector(self,S,ori=None,kernel='nonnormgaussian'):
        if ori is None:
            ori = self.get_center_orientation()
        dori = self.make_periodic(np.abs(self.ori_map[:,0] - ori),90)
        if kernel == 'gaussian':
            out = np.exp(-dori**2/(2*S**2))/np.sqrt(2*np.pi)/S
        elif kernel == 'nonnormgaussian':
            out = np.exp(-dori**2/(2*S**2))
        elif kernel == 'exponential':
            out = np.exp(-np.abs(dori)/S)
        elif kernel == 'vonmisses':
            dori_rad=dori/float(90)*np.pi
            S_rad=S/float(90)*np.pi
            d_rad=np.pi/float(90)
            out = np.exp(np.cos(dori_rad)/lam_rad)/(2*np.pi*i0(1/S_rad))*d_rad
        elif kernel == 'jtheta':
            dori_rad=dori/float(90)*np.pi
            S_rad=S/float(90)*np.pi
            d_rad=np.pi/float(90)
            out = np.real(jtheta(3,dori_rad/2,np.exp(-S_rad**2/2)))/(2*np.pi)*d_rad
        elif kernel == 'nonnormjtheta':
            dori_rad=dori/float(90)*np.pi
            S_rad=S/float(90)*np.pi
            out = np.real(jtheta(3,dori_rad/2,np.exp(-S_rad**2/2)))/np.real(jtheta(3,0,np.exp(-S_rad**2/2)))
        return out

    def make_kernel(self,S,kernel='gaussian'):
        dori = self.make_periodic(np.abs(self.ori_map[:,0] - self.ori_map[:,0][:,np.newaxis]),90)
        if kernel == 'gaussian':
            out = np.exp(-dori**2/(2*S**2))/np.sqrt(2*np.pi)/S
        elif kernel == 'nonnormgaussian':
            out = np.exp(-dori**2/(2*S**2))
        elif kernel == 'exponential':
            out = np.exp(-np.abs(dori)/S)
        elif kernel == 'vonmisses':
            dori_rad=dori/float(90)*np.pi
            S_rad=S/float(90)*np.pi
            d_rad=np.pi/float(90)
            out = np.exp(np.cos(dori_rad)/lam_rad)/(2*np.pi*i0(1/S_rad))*d_rad
        elif kernel == 'jtheta':
            dori_rad=dori/float(90)*np.pi
            S_rad=S/float(90)*np.pi
            d_rad=np.pi/float(90)
            out = np.real(jtheta(3,dori_rad/2,np.exp(-S_rad**2/2)))/(2*np.pi)*d_rad
        elif kernel == 'nonnormjtheta':
            dori_rad=dori/float(90)*np.pi
            S_rad=S/float(90)*np.pi
            out = np.real(jtheta(3,dori_rad/2,np.exp(-S_rad**2/2)))/np.real(jtheta(3,0,np.exp(-S_rad**2/2)))
        return out*self.dx

    def generate_inputs(self,H,varH,ori=None,kernel='nonnormgaussian'):
        self.H = np.zeros(self.N)

        for i in range(self.n):
            if i == 0:
                i_inds = self.E_idxs
                i_all = self.E_all
                Ni = self.NE
            else:
                i_inds = self.I_idxs
                i_all = self.I_all
                Ni = self.NI

            H_mean = H[i]*self.make_vector(self.SH[i],ori,kernel)
            H_var = varH[i]*self.make_vector(self.SvarH[i],ori,kernel)

            for loc in range(self.Nloc):
                self.H[i_inds[loc]] = np.random.normal(H_mean[loc],np.sqrt(H_var[loc]),Ni)

    def generate_weights(self,W,varW,kernel='gaussian'):
        self.M = np.zeros((self.N,self.N))

        for i in range(self.n):
            if i == 0:
                i_inds = self.E_idxs
                i_all = self.E_all
                Ni = self.NE
            else:
                i_inds = self.I_idxs
                i_all = self.I_all
                Ni = self.NI
            for j in range(self.n):
                if j == 0:
                    j_inds = self.E_idxs
                    j_all = self.E_all
                    Nj = self.NE
                else:
                    j_inds = self.I_idxs
                    j_all = self.I_all
                    Nj = self.NI

                W_mean = W[i,j]*self.make_kernel(self.SW[i,j])
                W_var = varW[i,j]*self.make_kernel(self.SvarW[i,j])

                for iloc in range(self.Nloc):
                    for jloc in range(self.Nloc):
                        self.M[i_inds[iloc],j_inds[jloc]] = np.random.normal(W_mean[iloc,jloc]/Nj,
                            np.sqrt(W_var[iloc,jloc]/Nj),(Ni,Nj))

    def generate_disorder(self,W,varW,H,varH,Lam,CV_Lam):
        self.generate_weights(W,varW)
        self.M_tf = tf.convert_to_tensor(self.M,dtype=tf.float32)
        self.generate_inputs(H,varH)
        self.H_tf = tf.convert_to_tensor(self.H,dtype=tf.float32)

        sigma_l = np.sqrt(np.log(1+CV_Lam**2))
        mu_l = np.log(Lam)-sigma_l**2/2
        self.LAM = np.zeros(self.N)
        self.LAM[self.E_all] = np.random.lognormal(mu_l, sigma_l, self.NE*self.Nloc)
        self.LAM_tf = tf.convert_to_tensor(self.LAM,dtype=tf.float32)

