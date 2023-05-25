import numpy as np
import torch
from scipy.special import erf, erfi
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import interp1d,interpn
from mpmath import fp

class SSN(object):
    def __init__(self,tE=0.02,tI=0.01,trp=0.002,tht=0.02,Vr=0.01,st=0.01):
        # Parameters defined by ale
        self.tE = tE
        self.tI = tI
        self.trp = trp
        self.tht = tht
        self.Vr = Vr
        self.st = st