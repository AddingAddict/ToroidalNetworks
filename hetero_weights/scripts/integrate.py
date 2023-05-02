import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def sim_dyn(ri,T,L,M,H,LAM,E_cond,mult_tau=False):

    # This function computes the dynamics of the rate model
    if mult_tau:
        def ode_fn(t,R):
            MU = tf.linalg.matvec(M,R) + H
            MU = tf.where(E_cond,ri.tE*MU,ri.tI*MU)
            MU = MU + LAM*L
            F = tf.where(E_cond,(-R+ri.phiE(MU))/ri.tE,(-R+ri.phiI(MU))/ri.tI)
            return F
    else:
        def ode_fn(t,R):
            MU = tf.linalg.matvec(M,R) + H + LAM*L
            F = tf.where(E_cond,(-R+ri.phiE(MU))/ri.tE,(-R+ri.phiI(MU))/ri.tI)
            return F

    return tfp.math.ode.DormandPrince().solve(ode_fn,np.min(T),tf.zeros_like(H),solution_times=T)