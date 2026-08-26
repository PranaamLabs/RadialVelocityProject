import numpy as np
from .lombscargle import LombScargle
from .fit_3var import fit_3var
from .fit_6var import fit_6var
import matplotlib.pyplot as plt

class Pipeline:
    def __init__(self,t,v,dv):
        self.t= t
        self.v= v
        self.dv= dv
    
    def _omegas(self, N_freq = 3000) -> np.ndarray:
        T_span = self.t.max() - self.t.min()
        dt = np.diff(np.sort(self.t))
        f_min = 1.0 / T_span
        f_max = 1.0 / (2.0 * np.median(dt))
        freqs = np.linspace(f_min, f_max, N_freq)
        return 2 * np.pi * freqs
    
    def apply_LS(self):
        omegas = self._omegas()
        gls = LombScargle(self.t, self.v, self.dv)
        P = gls.periodogram(omegas)
        best_omega = omegas[np.argmax(P)]
        self.period = 2*np.pi/best_omega

        return self.period, np.max(P)
    
    def fitting_3var(self):
        return fit_3var([self.t,self.v,self.dv,self.t.min()], self.period)

    def fitting_6var(self):
        return fit_6var([self.t,self.v,self.dv,self.t.min()], self.period)