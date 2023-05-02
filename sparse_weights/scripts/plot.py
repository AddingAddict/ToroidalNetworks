import numpy as np
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt

# Define plotting style
font = {'family' : 'normal', 'weight' : 'normal', 'size' : 7, 'family' : 'serif', 'serif' : ['Arial']}
mpl.rc('font', **font)
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['text.usetex'] = False
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['pdf.fonttype'] = 42

def get_gauss_fits(θ,r,guess=True):
    def halfgauss(x,A,S,B):
        return B+A*np.exp(-0.5*x**2/S**2)
        
    def HWHM(X,Y):
        if np.mean(Y) < 0:
            half_max = 0.5 * np.min(Y)
        else:
            half_max = 0.5 * np.max(Y)
        d = np.abs(Y-half_max)
        return X[np.argmin(d)]
    
    if guess:
        try:
            popt, pcov = curve_fit(halfgauss, θ, r, p0=(r[0]-np.mean(r[-5:]),
                                                        HWHM(θ,r-np.mean(r[-5:]))/np.sqrt(2*np.log(2)),
                                                        np.mean(r[-5:])))
        except:
            popt, pcov = curve_fit(halfgauss, θ, r, p0=(r[0]-np.mean(r[-5:]),10,np.mean(r[-5:])))
    else:
        popt, pcov = curve_fit(halfgauss, θ, r)
    popt[1] = np.abs(popt[1])
    
    return popt,np.sqrt(np.diag(pcov))

def plot_data_err(ax,θ,rmean,rerr,c='k',ls='-'):
    ax.plot(θ,rmean,c=c,ls=ls,alpha=0.5)
    ax.fill_between(θ,rmean+rerr,rmean-rerr,color=c,alpha=0.2)

def plot_data_fit_pts(ax,θ,r,label,c='k',ls='-',printfit=False):
    def halfgauss(x,A,S,B):
        return B+A*np.exp(-0.5*x**2/S**2)
    
    popt,perr = get_gauss_fits(θ,r,guess=False)
    
    if printfit:
        print(label,'Amp =',popt[0],'±',perr[0])
        print(label,'Wid =',popt[1],'±',perr[1])
        print(label,'Bas =',popt[2],'±',perr[2])
        print()

    quants = 30
    bin_idxs = pd.qcut(θ,quants,labels=False)
    bins = np.array([np.mean(θ[bin_idxs == i]) for i in range(quants)])
    mean = np.array([np.mean(r[bin_idxs == i]) for i in range(quants)])
    err = np.array([np.std(r[bin_idxs == i]/np.sqrt(len(r)/quants)) for i in range(quants)])

    θs = np.linspace(0,90,181)

    ax.plot(θs,halfgauss(θs,*popt),c=c,ls=ls)
    ax.plot(bins,mean,c=c,ls=ls,alpha=0.5)
    ax.fill_between(bins,mean+err,mean-err,color=c,alpha=0.2)

def plot_data_fit_line(ax,θ,r,label,c='k',ls='-',printfit=False):
    def halfgauss(x,A,S,B):
        return B+A*np.exp(-0.5*x**2/S**2)
    
    popt,perr = get_gauss_fits(θ,r,guess=True)
    
    if printfit:
        print(label,'Amp =',popt[0],'±',perr[0])
        print(label,'Wid =',popt[1],'±',perr[1])
        print(label,'Bas =',popt[2],'±',perr[2])
        print()

    θs = np.linspace(0,90,181)

    ax.plot(θs,halfgauss(θs,*popt),c=c,ls=ls)
    ax.plot(θ,r,c=c,ls=ls,alpha=0.5)
