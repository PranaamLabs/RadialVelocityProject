from .keplerian_fit import KeplerianModel
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt


def chi2_6var(params, t, v, dv):
    P, e, T_p, K, omega, gamma = params
    
    # 1. Kinematics
    M = 2 * np.pi * (t - T_p) / P
    E = np.array([KeplerianModel.eccentric_anomaly(m, e) for m in M])
    
    num = np.sqrt(1 + e) * np.sin(E / 2)
    den = np.sqrt(1 - e) * np.cos(E / 2)
    nu = 2 * np.arctan2(num, den)
    
    # 2. Pure nonlinear RV equation
    v_model = gamma + K * (np.cos(nu + omega) + e * np.cos(omega))
    
    # 3. Chi-squared
    w = 1 / dv**2
    return np.sum(w * (v - v_model)**2)

def fit_6var(params:list, P_guess):
    t, v, dv, t_min = params
    
    # Dynamic Boundaries for the global search
    v_ptp = np.ptp(v) # Peak-to-peak velocity (max minus min)
    
    bounds = [
        (P_guess * 0.8, P_guess * 1.2),  # P: 10% search window around your LS guess
        (0.0, 0.95),                     # e: 0 to highly eccentric
        (-P_guess, P_guess),                  # T_p: must be within one period
        (1.0, v_ptp),                    # K: up to max data amplitude
        (0.0, 2 * np.pi),                # omega: 0 to 2pi
        (np.min(v), np.max(v))           # gamma: bound within data limits
    ]
    
    # print("Running 6-variable Differential Evolution...")
    DE_result = differential_evolution(
        chi2_6var, 
        args=(t, v, dv), 
        bounds=bounds,
        maxiter=2000,
        popsize=20, # slightly larger population for 6 variables
        disp=False
    )

    # print("\nRunning 6-variable BFGS local refinement...")
    local_result = minimize(
        chi2_6var,
        x0=DE_result.x,
        args=(t, v, dv),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 2000}
    )

    # Extract all 6 variables
    P, e, T_p_relative, K, omega, gamma = local_result.x
    T_p_absolute = t_min + T_p_relative
    
    # The Hessian gives ALL errors automatically now!
    inv = local_result.hess_inv.todense()
    errors = np.sqrt(np.abs(np.diag(inv))) # abs() protects against numerical instability
    dP, de, dT_p, dK, dw, dg = errors

    return {
        'P': P,
        'e': e,
        'T_p': T_p_absolute,
        'K': K,
        'omega': omega,
        'gamma': gamma,
        'DE_result': DE_result,
        'local_result': local_result
    }