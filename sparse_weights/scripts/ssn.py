'''
Code collaboratively written by Alessandro Sanzeni, Agostina Palmigiano, and Tuan Nguyen
'''
try:
    import pickle5 as pickle
except:
    import pickle
import numpy as np
import torch
import torch_interpolations
from scipy.special import erf
from scipy.integrate import quad
from scipy.interpolate import interp1d,interpn

sr2 = np.sqrt(2)
sr2pi = np.sqrt(2*np.pi)
srpi = np.sqrt(np.pi)

def expval(fun,us,sigs):
    if np.isscalar(us):
        return quad(lambda z: fun(us+sigs*z)*np.exp(-z**2/2)/sr2pi,-8,8)[0]
    else:
        return [quad(lambda z: fun(us[i]+sigs[i]*z)*np.exp(-z**2/2)/sr2pi,-8,8)[0]
                for i in range(len(us))]

du = 1e-4

def d(fun,u):
    return (fun(u+du)-fun(u-du))/(2*du)

def d2(fun,u):
    return (fun(u+du)-2*fun(u)+fun(u-du))/du**2

def Mx(n,x):
    pdf = np.exp(-0.5*x**2)/sr2pi
    cdf = 0.5*(1+erf(x/sr2))
    
    if n == -4:
        return -x*(x**2-3)*pdf
    elif n == -3:
        return (x**2-1)*pdf
    elif n == -2:
        return -x*pdf
    elif n == -1:
        return pdf
    elif n == 0:
        return cdf
    elif n == 1:
        return pdf+x*cdf
    elif n == 2:
        return x*pdf+(1+x**2)*cdf
    elif n == 3:
        return (2+x**2)*pdf+x*(3+x**2)*cdf
    elif n == 4:
        return x*(5+x**2)*pdf+(3+6*x**2+x**4)*cdf
    elif n == 5:
        return (8+9*x**2+x**4)*pdf+x*(15+10*x**2+x**4)*cdf
    elif n == 6:
        return x*(3+x**2)*(11+x**2)*pdf+(15+45*x**2+15*x**4+x**6)*cdf
    elif n == 7:
        return (48+87*x**2+20*x**4+x**6)*pdf+x*(105+105*x**2+21*x**4+x**6)*cdf
    elif n == 8:
        return x*(279+185*x**2+27*x**4+x**6)*pdf+(105+420*x**2+210*x**4+28*x**6+x**8)*cdf
    else:
        raise('Power not implemented')

def Cx(n,x1,x2,c):
    if x1 > x2:
        xi = x1
        xj = x2
    else:
        xi = x2
        xj = x1
        
    if n == 0:
        A = Mx(0,xj)
    elif n == 1:
        A = xi*Mx(1,xj) + Mx(0,xj)
    elif n == 2:
        A = (1+xi**2)*Mx(2,xj) + 4*xi*Mx(1,xj) + 2*Mx(0,xj)
    elif n == 3:
        A = xi*(3+xi**2)*Mx(3,xj) + 9*(1+xi**2)*Mx(2,xj) + 18*xi*Mx(1,xj) + 6*Mx(0,xj)
    else:
        raise('Power not implemented')

    if x1 > - x2:
        pdf1 = np.exp(-0.5*x1**2)/sr2pi
        pdf2 = np.exp(-0.5*x2**2)/sr2pi
        cdf1 = 0.5*(1+erf(x1/sr2))
        cdf2 = 0.5*(1+erf(x2/sr2))
        if n == 0:
            B = cdf1+cdf2-1
        elif n == 1:
            B = (x1*x2-1)*(cdf1+cdf2-1) + x1*pdf2+x2*pdf1
        elif n == 2:
            B = (x1**2*x2**2+x1**2+x2**2-4*x1*x2+3)*(cdf1+cdf2-1) +\
                (x1*(1+x2**2)-4*x2)*pdf1 + (x2*(1+x1**2)-4*x1)*pdf2
        elif n == 3:
            B = (x1**3*x2**3+3*(x1**3*x2+x1*x2**3)-9*(x1**2*x2**2+x1**2+x2**2)+\
                    27*x1*x2-15)*(cdf1+cdf2-1) +\
                (x1**2*x2*(3+x2**2)-9*x1*(1+x2**2)+2*x2*(12+x2**2))*pdf1 +\
                (x2**2*x1*(3+x1**2)-9*x2*(1+x1**2)+2*x1*(12+x1**2))*pdf2
        else:
            raise('Power not implemented')
    else:
        B = 0

    r1 = Mx(n,x1)
    r2 = Mx(n,x2)

    Λp = A-r1*r2
    Λm = B-r1*r2

    a0 = r1*r2
    a1 = n**2*Mx(n-1,x1)*Mx(n-1,x2)
    a2 = (n*np.fmax(n-1,1))**2*Mx(n-2,x1)*Mx(n-2,x2)/2
    a3 = (n*np.fmax(n-1,1)*np.fmax(n-2,1))**2*Mx(n-3,x1)*Mx(n-3,x2)/6
    a4 = (n*np.fmax(n-1,1)*np.fmax(n-2,1)*np.fmax(n-3,1))**2*Mx(n-4,x1)*Mx(n-4,x2)/24
    a5 = (n*np.fmax(n-1,1)*np.fmax(n-2,1)*np.fmax(n-3,1)*np.fmax(n-4,1))**2*Mx(n-5,x1)*Mx(n-5,x2)/120
    a6 = (Λp+Λm)/2 - a2 - a4
    a7 = (Λp-Λm)/2 - a1 - a3 - a5

    return a0 + a1*c + a2*c**2 + a3*c**3 + a4*c**4 + a5*c**5 + a6*c**6 + a7*c**7

class SSN_refract(object):
    def __init__(self,tE=0.02,tI=0.01,trp=0.002,k=0.04,n=2.0,tht=0):
        # Parameters defined by ale
        self.tE = tE
        self.tI = tI
        self.trp = trp
        self.k = k
        self.n = n
        self.tht = tht
        
    def calc_phi(self,u,t):
        r = np.zeros_like(u);
        umtht = u - self.tht

        if np.isscalar(u):
            if(umtht>self.tht): r=self.k*umtht**self.n
        else:
            for idx in range(len(umtht)):
                if(umtht[idx]>self.tht): r[idx]=self.k*umtht[idx]**self.n
        return 1/(self.trp + 1/r)

    def calc_phi_tensor(self,u,t,out=None):
        umtht = u - self.tht
        if not out:
            out = torch.zeros_like(u)
        torch.where(umtht>0,self.k*umtht**self.n,out,out=out)
        return 1/(self.trp + 1/out)

    def phiE(self,u):
        return self.calc_phi(u,self.tE)
    def phiI(self,u):
        return self.calc_phi(u,self.tI)

    def phiE_tensor(self,u):
        return self.calc_phi_tensor(u,self.tE)
    def phiI_tensor(self,u):
        return self.calc_phi_tensor(u,self.tI)
    
class SSN(object):
    def __init__(self,tE=0.02,tI=0.01,k=0.04,n=2.0,tht=0):
        # Parameters defined by ale
        self.tE = tE
        self.tI = tI
        self.k = k
        self.n = n
        self.tht = tht
        
    def calc_phi(self,u,t):
        r = np.zeros_like(u);
        umtht = u - self.tht

        if np.isscalar(u):
            if(umtht>self.tht): r=self.k*umtht**self.n
        else:
            for idx in range(len(umtht)):
                if(umtht[idx]>self.tht): r[idx]=self.k*umtht[idx]**self.n
        return r

    def calc_phi_tensor(self,u,t,out=None):
        umtht = u - self.tht
        if not out:
            out = torch.zeros_like(u)
        torch.where(umtht>0,self.k*umtht**self.n,out,out=out)
        return out
    
    def calc_M(self,u,var):
        if np.isclose(var,0): return self.calc_phi(u,self.tE)
        
        if np.isclose(self.n,np.round(self.n)):
            return self.k*var**(self.n/2)*Mx(int(np.round(self.n)),(u-self.tht)/np.sqrt(var))
        else:
            raise('Power not implemented')
    
    def calc_C(self,u1,u2,var1,var2,cov):
        if np.isclose(cov,0):
            return self.calc_M(u1,var1)*self.calc_M(u1,var1)
        if np.isclose(var1,0) or np.isclose(var2,0):
            c = 0
        else:
            c = np.sign(cov)*min(abs(cov)/np.sqrt(var1*var2),1)
            
        x1 = u1/np.sqrt(var1)
        x2 = u2/np.sqrt(var2)
        
        if np.isclose(self.n,np.round(self.n)):
            return self.k**2*var1**(self.n/2)*var2**(self.n/2)*Cx(int(np.round(self.n)),x1,x2,c)
        else:
            raise('Power not implemented')

    def set_up_nonlinearity_w_laser(self,LLam,CV_Lam,nameout=None):
        save_file = False
        if nameout is not None:
            try:
                with open(nameout+'.pkl', 'rb') as handle:
                    out_dict = pickle.load(handle)
                self.phiL_int_E=out_dict['phiL_int_E']
                self.phiL2_int_E=out_dict['phiL2_int_E']
                print('Loading previously saved nonlinearity with laser')
                return None
            except:
                print('Calculating nonlinearity with laser')
                save_file = True

        Lvar = np.log(1+CV_Lam**2)
        Lstd = np.sqrt(Lvar)
        Lmean = np.log(LLam)-0.5*Lvar
        
        u_tab_max=5.0;
        u_tab=np.linspace(-u_tab_max/5,u_tab_max,int(10000*1.2+1))
        u_tab=np.concatenate(([-1000],u_tab))
        u_tab=np.concatenate((u_tab,[1000]))

        phiL_tab_E=u_tab*0
        phiL2_tab_E=u_tab*0

        for idx in range(len(phiL_tab_E)):
            phiL_tab_E[idx]=quad(lambda x: np.exp(-0.5*((np.log(x)-Lmean)/Lstd)**2)/(np.sqrt(2*np.pi)*Lstd*x)*\
                self.phi_int_E(u_tab[idx]+x),0,50*LLam)[0]
            phiL2_tab_E[idx]=quad(lambda x: np.exp(-0.5*((np.log(x)-Lmean)/Lstd)**2)/(np.sqrt(2*np.pi)*Lstd*x)*\
                self.phi_int_E(u_tab[idx]+x)**2,0,50*LLam)[0]

        self.phiL_int_E=interp1d(u_tab, phiL_tab_E, kind='linear', fill_value='extrapolate')
        self.phiL2_int_E=interp1d(u_tab, phiL2_tab_E, kind='linear', fill_value='extrapolate')

        if save_file:
            out_dict = {'phiL_int_E':self.phiL_int_E,
                        'phiL2_int_E':self.phiL2_int_E}
            with open(nameout+'.pkl', 'wb') as handle:
                pickle.dump(out_dict,handle)
        
    def set_up_mean_nonlinearity_w_laser(self,nameout=None):
        save_file = False
        if nameout is not None:
            try:
                with open(nameout+'.pkl', 'rb') as handle:
                    out_dict = pickle.load(handle)
                self.uL_tab=out_dict['uL_tab']
                self.sigL_tab=out_dict['sigL_tab']
                self.M_phiL_tab_E=out_dict['M_phiL_tab_E']
                self.M_phiL2_tab_E=out_dict['M_phiL2_tab_E']
                print('Loading previously saved mean nonlinearity')

                def M_phi_int_E(self,u,sig):
                    return interpn((self.u_tab,self.sig_tab), self.M_phi_tab_E, (u,sig), method='linear', fill_value=None)
                def M_phi_int_I(self,u,sig):
                    return interpn((self.u_tab,self.sig_tab), self.M_phi_tab_I, (u,sig), method='linear', fill_value=None)
                return None
            except:
                print('Calculating mean nonlinearity')
                save_file = True

        u_tab_max=1.0;
        u_tab=np.linspace(-u_tab_max/5,u_tab_max,int(1000*1.2+1))
        u_tab=np.concatenate(([-1000],u_tab))
        u_tab=np.concatenate((u_tab,[1000]))

        sig_tab_max=0.1;
        sig_tab=np.linspace(0,sig_tab_max,int(100+1))
        sig_tab=np.concatenate((sig_tab,[1]))

        M_phiL_tab_E,M_phiL2_tab_E=u_tab[:,None]*sig_tab[None,:]*0,u_tab[:,None]*sig_tab[None,:]*0;
        # phi_der_tab_E,phi_der_tab_I=u_tab*0,u_tab*0;

        for u_idx in range(len(u_tab)):
            for sig_idx in range(len(sig_tab)):
                M_phiL_tab_E[u_idx,sig_idx]=expval(self.phiL_int_E,u_tab[u_idx],sig_tab[sig_idx])
                M_phiL2_tab_E[u_idx,sig_idx]=expval(self.phiL2_int_E,u_tab[u_idx],sig_tab[sig_idx])

        self.uL_tab=u_tab
        self.sigL_tab=sig_tab
        self.M_phiL_tab_E=M_phiL_tab_E
        self.M_phiL2_tab_E=M_phiL2_tab_E

        if save_file:
            out_dict = {'uL_tab':self.u_tab,
                        'sigL_tab':self.sig_tab,
                        'M_phiL_tab_E':self.M_phiL_tab_E,
                        'M_phiL2_tab_E':self.M_phiL2_tab_E}
            with open(nameout+'.pkl', 'wb') as handle:
                pickle.dump(out_dict,handle)

    def M_phiL_int_E(self,u,sig):
        return interpn((self.uL_tab,self.sigL_tab), self.M_phiL_tab_E, (u,sig), method='linear', fill_value=None)[0]
    def M_phiL2_int_E(self,u,sig):
        return interpn((self.uL_tab,self.sigL_tab), self.M_phiL2_tab_E, (u,sig), method='linear', fill_value=None)[0]

    def phiE_tensor(self,u):
        return self.calc_phi_tensor(u,self.tE)
    def phiI_tensor(self,u):
        return self.calc_phi_tensor(u,self.tI)

    def dphiE_tensor(self,u):
        return d(self.phiE_tensor,u)
    def dphiI_tensor(self,u):
        return d(self.phiI_tensor,u)

    def phiE(self,u):
        return self.calc_phi(u,self.tE)
    def phiI(self,u):
        return self.calc_phi(u,self.tI)
    def phiLE(self,u):
        return self.phiL_int_E(u)

    def dphiE(self,u):
        return d(self.phiE,u)
    def dphiI(self,u):
        return d(self.phiI,u)
    def dphiLE(self,u):
        return d(self.phiLE,u)

    def d2phiE(self,u):
        return d2(self.phiE,u)
    def d2phiI(self,u):
        return d2(self.phiI,u)
    def d2phiLE(self,u):
        return d2(self.phiLE,u)

    def phi2E(self,u):
        return self.phi_int_E(u)**2
    def phi2I(self,u):
        return self.phi_int_I(u)**2
    def phiL2E(self,u):
        return self.phiL2_int_E(u)

    def dphi2E(self,u):
        return d(self.phi2E,u)
    def dphi2I(self,u):
        return d(self.phi2I,u)
    def dphiL2E(self,u):
        return d(self.phiL2E,u)

    def d2phi2E(self,u):
        return d2(self.phi2E,u)
    def d2phi2I(self,u):
        return d2(self.phi2I,u)
    def d2phiL2E(self,u):
        return d2(self.phiL2E,u)
    
    def FE(self,mu):
        if np.isscaler(mu):
            return self.phiE([mu],self.tE)
        return self.phiE(mu)
    def FI(self,mu):
        if np.isscaler(mu):
            return self.phiI([mu],self.tI)
        return self.phiI(mu)
    
    def ME(self,mu,Sig):
        args = np.row_stack(list(np.broadcast(mu,Sig)))
        out = np.zeros(args.shape[0])
        for i,(mu_i,Sig_i) in enumerate(args):
            out[i] = self.calc_M(mu_i,Sig_i)
        return out
    def MI(self,mu,Sig):
        args = np.row_stack(list(np.broadcast(mu,Sig)))
        out = np.zeros(args.shape[0])
        for i,(mu_i,Sig_i) in enumerate(args):
            out[i] = self.calc_M(mu_i,Sig_i)
        return out
    
    def CE(self,mu,Sig,k):
        args = np.row_stack(list(np.broadcast(mu,Sig,k)))
        out = np.zeros(args.shape[0])
        for i,(mu_i,Sig_i,k_i) in enumerate(args):
            out[i] = self.calc_C(mu_i,mu_i,Sig_i,Sig_i,k_i)
        return out
    def CI(self,mu,Sig,k):
        args = np.row_stack(list(np.broadcast(mu,Sig,k)))
        out = np.zeros(args.shape[0])
        for i,(mu_i,Sig_i,k_i) in enumerate(args):
            out[i] = self.calc_C(mu_i,mu_i,Sig_i,Sig_i,k_i)
        return out