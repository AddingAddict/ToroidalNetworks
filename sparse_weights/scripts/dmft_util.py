import numpy as np
from scipy.linalg import toeplitz
from scipy.interpolate import interp1d,RegularGridInterpolator
from scipy.integrate import quad,simpson
from mpmath import fp

sr2pi = np.sqrt(2*np.pi)

jtheta = np.vectorize(fp.jtheta, 'D')

def wrapnormdens(x,s,L=180):
    return np.real(jtheta(3,x*np.pi/L,np.exp(-(s*(2*np.pi)/L)**2/2)))/(2*np.pi)

def basesubwrapnorm(x,s,L=180):
    return (wrapnormdens(x,s,L)-wrapnormdens(L/2,s,L))/(wrapnormdens(0,s,L)-wrapnormdens(L/2,s,L))

def doub_vec(A):
    if A.ndim==1:
        return np.concatenate([A,A])
    else:
        return np.kron(np.ones(2)[...,None,:],A)

def doub_mat(A):
    if A.ndim==2:
        return np.block([[A,np.zeros_like(A)],[np.zeros_like(A),A]])
    elif A.ndim==4:
        return np.kron(np.eye(2)[...,None,:,:],A.transpose(0,2,1,3)).transpose(0,2,1,3)
    else:
        return np.kron(np.eye(2)[...,None,:,:],A)

def R(M1,M2,mu1,mu2,Sig1,Sig2,k):
    c = np.sign(k)*np.fmin(np.abs(k)/np.sqrt(Sig1*Sig2),1)
    sig1 = np.sign(c)*np.sqrt(Sig1*np.abs(c))
    sig2 = np.sqrt(Sig2*np.abs(c))
    Del1 = Sig1*(1-np.abs(c))
    Del2 = Sig2*(1-np.abs(c))
    return quad(lambda x: np.exp(-0.5*x**2)/sr2pi*\
                M1(mu1+sig1*x,Del1)*\
                M2(mu2+sig2*x,Del2),-8,8)[0]

def R_int(M1,M2,mu1,mu2,Sig1,Sig2,k,x):
    c = np.sign(k)*np.fmin(np.abs(k)/np.sqrt(Sig1*Sig2),1)
    sig1 = np.sign(c)*np.sqrt(Sig1*np.abs(c))
    sig2 = np.sqrt(Sig2*np.abs(c))
    Del1 = Sig1*(1-np.abs(c))
    Del2 = Sig2*(1-np.abs(c))
    return np.exp(-0.5*x**2)/sr2pi*\
                M1(mu1+sig1*x,Del1)*\
                M2(mu2+sig2*x,Del2)

def R_simp(M1,M2,mu1,mu2,Sig1,Sig2,k):
    xs = np.linspace(-8,8,1001)
    return simpson(R_int(M1,M2,mu1,mu2,Sig1,Sig2,k,xs),x=xs)

def each_diag(A,k=0):
    if k == 0:
        return np.einsum('...jj->...j',A)
    else:
        new_shape = np.array(A.shape[:-1])
        new_shape[-1] -= k
        out = np.zeros(new_shape)
        mult_shape = A.shape[:-2]
        for i in range(np.prod(mult_shape)):
            mult_idx = np.unravel_index(i,mult_shape)
            out[mult_idx] = np.diag(A[mult_idx],k)
        return out

def each_matmul(A,B):
    if A.ndim==2:
        return np.einsum('ijk,jk->ik',A[:,:,None],B)
    else:
        return np.einsum('ijk,jk->ik',A,B)

def grid_stat(stat,A,Tstat,dt):
    Nsav = A.shape[-1]
    Nstat = round(Tstat/dt)+1
    new_shape = np.array(A.shape)
    new_shape[-1] = Nstat
    A_ext = np.zeros(new_shape)
    if Nsav < Nstat:
        A_ext[...,:Nsav] = A
        A_ext[...,Nsav:] = A[...,-1:]
    else:
        A_ext = A[...,:Nstat]
    mult_shape = A.shape[:-1]
    new_shape = np.concatenate([new_shape,[Nstat]])
    A_mat = np.zeros(new_shape)
    for i in range(np.prod(mult_shape)):
        mult_idx = np.unravel_index(i,mult_shape)
        A_mat[mult_idx] = toeplitz(A_ext[mult_idx])
    return stat(A_mat,axis=(-1,-2))

def d2_stencil(Tsav,dt):
    Nsav = round(Tsav/dt)+1
    d2_mat = np.zeros((Nsav,Nsav))
    d2_mat[(np.arange(Nsav), np.arange(Nsav))] = -2/dt**2
    d2_mat[(np.arange(Nsav-1), np.arange(1,Nsav))] = 1/dt**2
    d2_mat[(np.arange(1,Nsav), np.arange(Nsav-1))] = 1/dt**2
    d2_mat[0,1] = 2/dt**2
    d2_mat[-1,-1] = -1/dt**2
    return d2_mat

def get_time_freq_func(f):
    N = f.shape[-1]
    new_shape = np.array(f.shape)
    new_shape[-1] += N-2
    ft = np.zeros(new_shape)
    ft[...,:N] = f
    ft[...,N:] = f[...,-1:1:-1]
    fo = np.real(np.fft.fft(ft))
    return ft,fo

def smooth_func(f,dt,fcut=17,beta=1):
    N = f.shape[-1]
    _,fo = get_time_freq_func(f)
    fo *= 1/(1 + np.exp((np.abs(np.fft.fftfreq(2*(N-1),dt)) - fcut)*beta))
    return np.real(np.fft.ifft(fo))[...,:N]
    
def get_solve_width(sa,L=180):
    widths = np.linspace(1,L*3/4,135)
    fbars = basesubwrapnorm(sa,widths,L)
    max_fbar = np.max(fbars)
    min_fbar = np.min(fbars)
    widths_vs_fbars_itp = interp1d(fbars,widths)
    def solve_widths(fbar):
        return widths_vs_fbars_itp(np.fmax(min_fbar,np.fmin(max_fbar,fbar)))
    return solve_widths
    
def get_2feat_solve_width(sa,dori=45,L=180):
    widths = np.linspace(1,L*3/4,135)
    fbars = (basesubwrapnorm(sa,widths,L) + basesubwrapnorm(sa+dori,widths,L)) /\
        (1 + basesubwrapnorm(dori,widths,L))
    max_fbar = np.max(fbars)
    min_fbar = np.min(fbars)
    widths_vs_fbars_itp = interp1d(fbars,widths)
    def solve_widths(fbar):
        return widths_vs_fbars_itp(np.fmax(min_fbar,np.fmin(max_fbar,fbar)))
    return solve_widths

def unstruct_fact(s,L=180):
    return (1/L-wrapnormdens(L/2,s,L))/(wrapnormdens(0,s,L)-wrapnormdens(L/2,s,L))

def struct_fact(x,sconv,sorig,L=180):
    return (wrapnormdens(x,sconv,L)-wrapnormdens(L/2,sorig,L))/\
        (wrapnormdens(0,sorig,L)-wrapnormdens(L/2,sorig,L))
    
def inv_overlap(xs,ss,L=180):
    overlap_mat = basesubwrapnorm(xs[None,:,None]-xs[None,None,:],ss[:,None,:],L)
    return np.linalg.inv(overlap_mat)