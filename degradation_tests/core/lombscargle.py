import numpy as np

class LombScargle:
    """
    Generalized Lomb-Scargle Periodogram (GLS).

    This implementation follows the weighted floating-mean formulation of
    Zechmeister & Kürster (2009).

    The model fitted at each angular frequency ω is

        y(t) = C + A cos(ωt) + B sin(ωt)

    where C is the floating mean and A, B are the sinusoidal amplitudes.

    The implementation uses weighted least squares with weights

        w_n = 1 / σ_n²

    and computes the improvement in fit relative to the weighted constant
    model,

        Δχ² = χ₀² - χ²(ω).

    A time offset τ is introduced to construct shifted basis functions

        c_n = cos[ω(t_n - τ)]
        s_n = sin[ω(t_n - τ)]

    chosen such that

        Σ w_n c_n s_n ≈ 0,

    reducing the correlation between the sine and cosine basis vectors and
    simplifying the Gram matrix. Although the original GLS formulation can
    be written without explicitly introducing τ, this implementation uses
    the shifted basis for numerical convenience and derivational clarity.

    The periodogram power is normalized according to Eq. (22) of
    Zechmeister & Kürster (2009),

        P(ω) = ((N - 1) / 2) · Δχ² / χ₀²

    where N is the number of observations.

    References
    ----------
    Zechmeister, M. & Kürster, M. (2009),
    'The generalised Lomb-Scargle periodogram',
    Astronomy & Astrophysics, 496, 577-584.
    """
    def __init__(self, t: list, y: list, dy: list = None):
        #Initial Stuff
        self.t = np.asarray(t, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        if len(self.t) != len(self.y):
            raise ValueError("Number of observations and time of it dont match")
        
        if dy is not None and len(dy) != len(y):
            raise ValueError("Number of observation values and uncertainities dont match")

        if dy is not None:
            self.dy = np.asarray(dy, dtype=np.float64)
        else:
            self.dy = np.ones(self.y.shape, dtype=np.float64)
        
        
        #Weights
        self.w = 1/self.dy**2
        self.W = np.sum(self.w)
        self.Y0 = np.sum(self.w*self.y)
        self.y_mean = self.Y0/self.W
        self.chi0_2 = np.sum(self.w*(self.y - self.y_mean)**2)
        self.N = self.y.size

        self.w_y = self.w * self.y

    def _compute_tau(self, omega: float) -> float:
        '''Computes tau using ZK09-(3)'''
        theta = 2*omega*self.t
        return (np.arctan2(
            np.sum(self.w* np.sin(theta)),np.sum(self.w* np.cos(theta))
            )) / (2 * omega)
    
    def _compute_shifted_basis(self, omega:float, tau:float)-> tuple[np.ndarray,np.ndarray]:
        theta = omega * (self.t - tau)
        c_n = np.cos(theta)
        s_n = np.sin(theta)
        return c_n, s_n
    
    def _basisAndCorrelations(self, c_n:np.ndarray, s_n:np.ndarray) -> tuple[float, float, float, float, float, float]:
        C_hat = np.sum(self.w * c_n)
        S_hat = np.sum(self.w * s_n)
        CC_hat = np.sum(self.w * c_n**2)
        SS_hat = np.sum(self.w * s_n**2)
        Yc_hat = np.sum(self.w_y * c_n)
        Ys_hat = np.sum(self.w_y * s_n)
        return (C_hat, S_hat, CC_hat, SS_hat, Yc_hat, Ys_hat)
    
    def _compute_reduced_gram(self, C_hat:float, S_hat:float, CC_hat:float, SS_hat:float)-> tuple[float, float, float]:
        G_11= CC_hat - C_hat**2/self.W
        G_22= SS_hat - S_hat**2/self.W
        G_12= -(C_hat* S_hat)/self.W
        return (G_11, G_12, G_22)
    
    def _compute_reduced_rhs(self, Yc_hat:float, Ys_hat:float, C_hat:float, S_hat:float)->tuple[float, float]:
        r1 = Yc_hat - self.Y0*C_hat/self.W
        r2 = Ys_hat - self.Y0*S_hat/self.W
        return r1, r2
    
    def _compute_delta_chi2(self, G_11:float, G_22:float, G_12:float, r1:float, r2:float)->float:
        detG= G_11*G_22 - G_12**2
        if detG < 1e-15:
            return 0.0
        delta_chi2 = (G_22*r1**2 + G_11*r2**2 - 2*G_12*r1*r2)/(detG)
        return delta_chi2
    
    def _normalize(self, delta_chi2:float)->float:
        '''Normalized according to ZK09-(22)'''
        factor = (self.N - 1)/2
        return factor * delta_chi2 / self.chi0_2
    
    def power(self, omega:float)->float:
        '''Computes power for single freq'''
        tau = self._compute_tau(omega)
        c, s = self._compute_shifted_basis(omega, tau)
        C_hat, S_hat, CC_hat, SS_hat, Yc_hat, Ys_hat = self._basisAndCorrelations(c, s)
        G_11, G_12, G_22 = self._compute_reduced_gram(C_hat, S_hat, CC_hat, SS_hat)
        r1, r2 = self._compute_reduced_rhs(Yc_hat, Ys_hat, C_hat, S_hat)
        dchi2 = self._compute_delta_chi2(G_11, G_22, G_12, r1, r2)
        return self._normalize(dchi2)
    
    def periodogram(self, omegas:list)->np.ndarray:
        powers = np.zeros(len(omegas))
        for i, omega in enumerate(omegas):
            powers[i] = self.power(omega)
        return powers