"""
v = γ + K(cos(ν + ω) + e cos ω)
is the general equation of radial velocity. Here v = v(P, e, ω, T_p, K, γ)
By a few simplifications we can linearize this equation
v = γ + A(cosν + e) + Bsinν
where A = Kcosω and B = -Ksinω. Here v = v(P, e, T_p, A, B, γ)
this we can solve with linear regression to get best A, B, γ.
P we know from LombScargle Periodogram, leaving us with just need of e and T_p
These two parameters we can use some optimization tool to predict best values.
This file will contain the Class to handle the linear regression for given t_i, v_i, dv_i, e, T_p, P
and optimization to find right e and T_p
"""

import numpy as np

class KeplerianModel:
    def __init__(self, t, v, dv, e, T_p, P):
        self.t= np.array(t)
        self.v= np.array(v)
        self.dv= np.array(dv)
        self.e= e
        self.T_p = T_p
        self.P = P
        self.w = 1/self.dv**2
        self.W = np.sum(self.w)
        self.WV = np.sum(self.w * self.v)


    def mean_anomaly(self):
        '''M=(2*pi*(t-T_p)/P)
        Use: M = self.mean_anomaly()'''
        return 2*np.pi*(self.t - self.T_p) / self.P
    
    @staticmethod
    def eccentric_anomaly(M, e, max_iterations=200, tolerance=1e-10):
        # Map M into the standard [-pi, pi] domain
        M = (M + np.pi) % (2 * np.pi) - np.pi
        
        # Better initial guess for high eccentricity orbits
        E = M + e * np.sin(M) if e < 0.8 else M
        
        for _ in range(max_iterations):
            f = E - e * np.sin(E) - M
            df = 1.0 - e * np.cos(E)
            
            # Avoid zero division if e approaches or exceeds 1 due to optimizer steps
            if np.abs(df) < 1e-12:
                df = 1e-12
                
            delta = f / df
            E -= delta
            
            if np.abs(delta) < tolerance:
                return E
                
        # If it fails to converge, return a fallback instead of crashing the pipeline
        return E
    
    def true_anomaly(self, E:np.ndarray):
        '''
        tan(ν/2) = sqrt((1+e)/(1-e))*tan(E/2)
        Use: nu = self.true_anomaly(E)
        '''
        num= np.sqrt(1+self.e)*np.sin(E/2)
        den= np.sqrt(1-self.e)*np.cos(E/2)
        nu = 2*np.arctan2(num,den)
        return nu
    
    def solve(self,nu:np.ndarray):
        '''AX=b
        returns A,b
        define X= cos(nu) + e
        Y = sin(nu)
        Use: self.design_matrix(nu)
        returns ndarray: gamma, A, B'''
        X = np.cos(nu) + self.e
        Y = np.sin(nu)
        WX = np.sum(self.w * X)
        WY = np.sum(self.w * Y)
        WXX = np.sum(self.w * X**2)
        WYY = np.sum(self.w * Y**2)
        WXY = np.sum(self.w * X * Y)
        WXV = np.sum(self.w * X * self.v)
        WYV = np.sum(self.w * Y * self.v)

        self.M = [[self.W, WX, WY],
                [WX, WXX, WXY],
                [WY, WXY, WYY]]
        self.b= [self.WV, WXV, WYV]
        return np.linalg.solve(self.M,self.b)
    
    def errors(self):
        '''M is the Fischer Information Matrix, the diagonal elements of its inverse are the squares of errors.
        The 1,2 element of the same is Covarianve of A and B
        returns dγ, dA, dB and CovAB'''
        C = np.linalg.inv(self.M)
        errors = np.sqrt(np.diag(C))
        cov = C[1][2]
        return np.append(errors,cov)
    
    def radial_velocity(self, nu, gamma, A, B):
        '''Calculates model v using:
        v = γ + A(cosν + e) + Bsinν'''
        return gamma + A*(np.cos(nu) + self.e) + B*np.sin(nu)
        
    def chi_squared(self, v_model):
        return np.sum(self.w * (self.v - v_model)**2)
    