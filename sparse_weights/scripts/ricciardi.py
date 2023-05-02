'''
Code collaboratively written by Alessandro Sanzeni, Agostina Palmigiano, and Tuan Nguyen
'''
import pickle
import numpy as np
from scipy.special import erf, erfi
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import interp1d,interpn
from mpmath import fp

sr2 = np.sqrt(2)
sr2pi = np.sqrt(2*np.pi)
srpi = np.sqrt(np.pi)

def expval(fun,us,sigs):
    if np.isscalar(us):
        return quad(lambda z: fun(us+sigs*z)*np.exp(-z**2/2)/sr2pi,-8,8)[0]
    else:
        return [quad(lambda z: fun(us[i]+sigs[i]*z)*np.exp(-z**2/2)/sr2pi,-8,8)[0]
                for i in range(len(us))]

du = 1e-3

def d(fun,u):
    return (fun(u+du)-fun(u-du))/(2*du)

def d2(fun,u):
    return (fun(u+du)-2*fun(u)+fun(u-du))/du**2
    
class Ricciardi(object):    
    def __init__(self,tE=0.02,tI=0.01,trp=0.002,tht=0.02,Vr=0.01,st=0.01):
        # Parameters defined by ale
        self.tE = tE
        self.tI = tI
        self.trp = trp
        self.tht = tht
        self.Vr = Vr
        self.st = st

    def calc_phi(self,u,t):
        min_u = (self.Vr-u)/self.st
        max_u = (self.tht-u)/self.st
        r = np.zeros_like(u);

        if np.isscalar(u):
            if(min_u>10):
                r=max_u/t/srpi*np.exp(-max_u**2)
            elif(max_u>-4):
                r=1.0/(self.trp+t*(0.5*np.pi*\
                                   (erfi(max_u)-erfi(min_u)) +
                                   max_u**2*fp.hyp2f2(1.0,1.0,1.5,2.0,max_u**2) -
                                   min_u**2*fp.hyp2f2(1.0,1.0,1.5,2.0,min_u**2)))
            else:
                r=1.0/(self.trp+t*(np.log(abs(min_u)) - np.log(abs(max_u)) +
                                   (0.25*min_u**-2 - 0.1875*min_u**-4 + 0.3125*min_u**-6 -
                                    0.8203125*min_u**-8 + 2.953125*min_u**-10) -
                                   (0.25*max_u**-2 - 0.1875*max_u**-4 + 0.3125*max_u**-6 -
                                    0.8203125*max_u**-8 + 2.953125*max_u**-10)))
        else:
            for idx in range(len(u)):
                if(min_u[idx]>10):
                    r[idx]=max_u[idx]/t/srpi*np.exp(-max_u[idx]**2)
                elif(min_u[idx]>-4):
                    r[idx]=1.0/(self.trp+t*(0.5*np.pi*\
                                       (erfi(max_u[idx]) -
                                        erfi(min_u[idx])) +
                                       max_u[idx]**2*fp.hyp2f2(1.0,1.0,1.5,2.0,
                                                                      max_u[idx]**2) -
                                       min_u[idx]**2*fp.hyp2f2(1.0,1.0,1.5,2.0,
                                                                      min_u[idx]**2)))
                else:
                    r[idx]=1.0/(self.trp+t*(np.log(abs(min_u[idx])) -
                                            np.log(abs(max_u[idx])) +
                                       (0.25*min_u[idx]**-2 - 0.1875*min_u[idx]**-4 +
                                        0.3125*min_u[idx]**-6 - 0.8203125*min_u[idx]**-8 +
                                        2.953125*min_u[idx]**-10) -
                                       (0.25*max_u[idx]**-2 - 0.1875*max_u[idx]**-4 +
                                        0.3125*max_u[idx]**-6 - 0.8203125*max_u[idx]**-8 +
                                        2.953125*max_u[idx]**-10)))
        return r
        
    def set_up_nonlinearity(self,nameout=None):
        save_file = False
        if nameout is not None:
            try:
                with open(nameout+'.pkl', 'rb') as handle:
                    out_dict = pickle.load(handle)
                self.phi_int_E=out_dict['phi_int_E']
                self.phi_int_I=out_dict['phi_int_I']
                print('Loading previously saved nonlinearity')
                return None
            except:
                print('Calculating nonlinearity')
                save_file = True

        u_tab_max=10.0;
        u_tab=np.linspace(-u_tab_max/5,u_tab_max,int(200000*1.2+1))
        u_tab=np.concatenate(([-10000],u_tab))
        u_tab=np.concatenate((u_tab,[10000]))

        phi_tab_E,phi_tab_I=u_tab*0,u_tab*0;
        # phi_der_tab_E,phi_der_tab_I=u_tab*0,u_tab*0;

        for idx in range(len(phi_tab_E)):
            phi_tab_E[idx]=self.calc_phi(u_tab[idx],self.tE)
            phi_tab_I[idx]=self.calc_phi(u_tab[idx],self.tI)

        self.phi_int_E=interp1d(u_tab, phi_tab_E, kind='linear', fill_value='extrapolate')
        self.phi_int_I=interp1d(u_tab, phi_tab_I, kind='linear', fill_value='extrapolate')

        if save_file:
            out_dict = {'phi_int_E':self.phi_int_E,
                        'phi_int_I':self.phi_int_I}
            with open(nameout+'.pkl', 'wb') as handle:
                pickle.dump(out_dict,handle)

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
        
    def set_up_mean_nonlinearity(self,nameout=None):
        save_file = False
        if nameout is not None:
            try:
                with open(nameout+'.pkl', 'rb') as handle:
                    out_dict = pickle.load(handle)
                self.u_tab=out_dict['u_tab']
                self.sig_tab=out_dict['sig_tab']
                self.M_phi_tab_E=out_dict['M_phi_tab_E']
                self.M_phi_tab_I=out_dict['M_phi_tab_I']
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

        M_phi_tab_E,M_phi_tab_I=u_tab[:,None]*sig_tab[None,:]*0,u_tab[:,None]*sig_tab[None,:]*0;
        # phi_der_tab_E,phi_der_tab_I=u_tab*0,u_tab*0;

        for u_idx in range(len(u_tab)):
            for sig_idx in range(len(sig_tab)):
                M_phi_tab_E[u_idx,sig_idx]=expval(self.phi_int_E,u_tab[u_idx],sig_tab[sig_idx])
                M_phi_tab_I[u_idx,sig_idx]=expval(self.phi_int_I,u_tab[u_idx],sig_tab[sig_idx])

        self.u_tab=u_tab
        self.sig_tab=sig_tab
        self.M_phi_tab_E=M_phi_tab_E
        self.M_phi_tab_I=M_phi_tab_I

        if save_file:
            out_dict = {'u_tab':self.u_tab,
                        'sig_tab':self.sig_tab,
                        'M_phi_tab_E':self.M_phi_tab_E,
                        'M_phi_tab_I':self.M_phi_tab_I}
            with open(nameout+'.pkl', 'wb') as handle:
                pickle.dump(out_dict,handle)

    def M_phi_int_E(self,u,sig):
        return interpn((self.u_tab,self.sig_tab), self.M_phi_tab_E, (u,sig), method='linear', fill_value=None)[0]
    def M_phi_int_I(self,u,sig):
        return interpn((self.u_tab,self.sig_tab), self.M_phi_tab_I, (u,sig), method='linear', fill_value=None)[0]
        
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

    def phiE(self,u):
        return self.phi_int_E(u)
    def phiI(self,u):
        return self.phi_int_I(u)
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
    
    