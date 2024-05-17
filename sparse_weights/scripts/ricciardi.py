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
from scipy.special import erf, erfi, dawsn
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import interp1d,interpn
from mpmath import fp

sr2 = np.sqrt(2)
sr2pi = np.sqrt(2*np.pi)
srpi = np.sqrt(np.pi)

def int_dawsni_scal(x):
    return -0.5*x**2*fp.hyp2f2(1.0,1.0,1.5,2.0,x**2)
int_dawsni = np.vectorize(int_dawsni_scal)

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
            if(min_u>3):
                r=max_u/t/srpi*np.exp(-max_u**2)
            elif(min_u>-5):
                r=1.0/(self.trp+t*(0.5*np.pi*(erfi(max_u[idx]) - erfi(min_u[idx])) -\
                    2*(int_dawsni(max_u[idx]) - int_dawsni(min_u[idx]))))
            else:
                r=1.0/(self.trp+t*(np.log(abs(min_u)) - np.log(abs(max_u)) +
                                   (0.25*min_u**-2 - 0.1875*min_u**-4 + 0.3125*min_u**-6 -
                                    0.8203125*min_u**-8 + 2.953125*min_u**-10) -
                                   (0.25*max_u**-2 - 0.1875*max_u**-4 + 0.3125*max_u**-6 -
                                    0.8203125*max_u**-8 + 2.953125*max_u**-10)))
        else:
            for idx in range(len(u)):
                if(min_u[idx]>3):
                    r[idx]=max_u[idx]/t/srpi*np.exp(-max_u[idx]**2)
                elif(min_u[idx]>-5):
                    r[idx]=1.0/(self.trp+t*(0.5*np.pi*(erfi(max_u[idx]) - erfi(min_u[idx])) -\
                        2*(int_dawsni(max_u[idx]) - int_dawsni(min_u[idx]))))
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

    def calc_phi_tensor(self,u,t,out=None):
        if not out:
            out = torch.zeros_like(u)
        min_u = (self.Vr-u)/self.st
        max_u = (self.tht-u)/self.st
        r_low = max_u/t/srpi*torch.exp(-max_u**2)
        r_mid = 1.0/(self.trp+t*(srpi*(1+(0.5641895835477563*max_u-0.07310176646978049*max_u**3+
                                        0.019916897946949282*max_u**5-0.001187484601754455*max_u**7+
                                        0.00014245755084666304*max_u**9-4.208652789675569e-6*max_u**11+
                                        2.8330406295105274e-7*max_u**13-3.2731460579579614e-9*max_u**15+
                                        1.263640520928807e-10*max_u**17)/(1.-0.12956950748735815*max_u**2+
                                        0.046412893575273534*max_u**4-0.002486221791373608*max_u**6+
                                        0.000410629108366176*max_u**8-9.781058014448444e-6*max_u**10+
                                        1.0371239952922995e-6*max_u**12-7.166099219321984e-9*max_u**14+
                                        6.85317470793816e-10*max_u**16+1.932753647574705e-12*max_u**18+
                                        4.121310879310989e-14*max_u**20))*
                                    torch.exp(max_u**2)*max_u*(654729075+252702450*max_u**2+79999920*max_u**4+20386080*max_u**6+
                                        4313760*max_u**8+784320*max_u**10+126720*max_u**12+18944*max_u**14+2816*max_u**16+512*max_u**18)/
                                    (654729075+689188500*max_u**2+364864500*max_u**4+129729600*max_u**6+34927200*max_u**8+
                                        7620480*max_u**10+1411200*max_u**12+230400*max_u**14+34560*max_u**16+5120*max_u**18+1024*max_u**20) -
                                srpi*(1+(0.5641895835477563*min_u-0.07310176646978049*min_u**3+
                                        0.019916897946949282*min_u**5-0.001187484601754455*min_u**7+
                                        0.00014245755084666304*min_u**9-4.208652789675569e-6*min_u**11+
                                        2.8330406295105274e-7*min_u**13-3.2731460579579614e-9*min_u**15+
                                        1.263640520928807e-10*min_u**17)/(1.-0.12956950748735815*min_u**2+
                                        0.046412893575273534*min_u**4-0.002486221791373608*min_u**6+
                                        0.000410629108366176*min_u**8-9.781058014448444e-6*min_u**10+
                                        1.0371239952922995e-6*min_u**12-7.166099219321984e-9*min_u**14+
                                        6.85317470793816e-10*min_u**16+1.932753647574705e-12*min_u**18+
                                        4.121310879310989e-14*min_u**20))*
                                    torch.exp(min_u**2)*min_u*(654729075+252702450*min_u**2+79999920*min_u**4+20386080*min_u**6+
                                        4313760*min_u**8+784320*min_u**10+126720*min_u**12+18944*min_u**14+2816*min_u**16+512*min_u**18)/
                                    (654729075+689188500*min_u**2+364864500*min_u**4+129729600*min_u**6+34927200*min_u**8+
                                        7620480*min_u**10+1411200*min_u**12+230400*min_u**14+34560*min_u**16+5120*min_u**18+1024*min_u**20)))
        r_hgh = 1.0/(self.trp+t*(torch.log(torch.abs(min_u)) - torch.log(torch.abs(max_u)) +
                                   (0.25*min_u**-2 - 0.1875*min_u**-4 + 0.3125*min_u**-6 -
                                    0.8203125*min_u**-8 + 2.953125*min_u**-10) -
                                   (0.25*max_u**-2 - 0.1875*max_u**-4 + 0.3125*max_u**-6 -
                                    0.8203125*max_u**-8 + 2.953125*max_u**-10)))
        torch.where(min_u>-3,r_mid,out,out=out)
        torch.where(min_u> 3,r_low,out,out=out)
        torch.where(min_u<=-3,r_hgh,out,out=out)
        return out
        
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
        
    def set_up_nonlinearity_tensor(self,nameout=None):
        save_file = False
        if nameout is not None:
            try:
                with open(nameout+'.pkl', 'rb') as handle:
                    out_dict = pickle.load(handle)
                self.phi_int_tensor_E=out_dict['phi_int_tensor_E']
                self.phi_int_tensor_I=out_dict['phi_int_tensor_I']
                print('Loading previously saved nonlinearity')
                return None
            except:
                print('Calculating nonlinearity')
                save_file = True

        if hasattr(self,"phi_int_E"):
            u_tab=self.phi_int_E.x
            phi_tab_E=self.phi_int_E.y
            phi_tab_I=self.phi_int_I.y
        else:
            u_tab_max=10.0;
            u_tab=np.linspace(-u_tab_max/5,u_tab_max,int(200000*1.2+1))
            u_tab=np.concatenate(([-10000],u_tab))
            u_tab=np.concatenate((u_tab,[10000]))

            phi_tab_E,phi_tab_I=u_tab*0,u_tab*0;
            # phi_der_tab_E,phi_der_tab_I=u_tab*0,u_tab*0;

            for idx in range(len(phi_tab_E)):
                phi_tab_E[idx]=self.calc_phi(u_tab[idx],self.tE)
                phi_tab_I[idx]=self.calc_phi(u_tab[idx],self.tI)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using",device)

        u_tab_tensor = torch.from_numpy(u_tab.astype(np.float32)).to(device)
        phi_tab_tensor_E = torch.from_numpy(phi_tab_E.astype(np.float32)).to(device)
        phi_tab_tensor_I = torch.from_numpy(phi_tab_I.astype(np.float32)).to(device)

        self.phi_int_tensor_E=torch_interpolations.RegularGridInterpolator((u_tab_tensor,), phi_tab_tensor_E)
        self.phi_int_tensor_I=torch_interpolations.RegularGridInterpolator((u_tab_tensor,), phi_tab_tensor_I)

        if save_file:
            out_dict = {'phi_int_tensor_E':self.phi_int_tensor_E,
                        'phi_int_tensor_I':self.phi_int_tensor_I}
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

    def phiE_tensor(self,u):
        # return self.calc_phi_tensor(u,self.tE)
        return self.phi_int_tensor_E(u[None,:])
    def phiI_tensor(self,u):
        # return self.calc_phi_tensor(u,self.tI)
        return self.phi_int_tensor_I(u[None,:])

    def dphiE_tensor(self,u):
        return d(self.phiE_tensor,u)
    def dphiI_tensor(self,u):
        return d(self.phiI_tensor,u)

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
    
    