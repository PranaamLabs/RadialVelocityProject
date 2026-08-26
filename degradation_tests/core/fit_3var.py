from .keplerian_fit import KeplerianModel
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
from math import sqrt

def chi2_3var(params, t,v,dv):
    P, e, T_p = params
    model = KeplerianModel(t,v,dv,e,T_p,P)
    M = model.mean_anomaly()
    E = np.array([model.eccentric_anomaly(m,e) for m in M])
    nu = model.true_anomaly(E)
    gamma, A, B = model.solve(nu)
    v_model = model.radial_velocity(nu, gamma, A, B)
    return model.chi_squared(v_model)

def fit_3var(params:list, P_guess):
    t, v, dv, t_min= params
    bounds = [
        (P_guess * 0.9, P_guess * 1.1), 
        (1e-3, 0.95), 
        (-P_guess, P_guess)
    ]
    DE_result = differential_evolution(
        chi2_3var, 
        args=(t, v, dv), 
        bounds=bounds,
        maxiter=200,
        popsize=10,
        tol=1e-4,
        disp=False,
        polish=False
    )

    local_result = minimize(
        chi2_3var,
        x0=DE_result.x,
        args=(t, v, dv),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000}
    )

    P_best, e_best, T_p_relative = local_result.x
    T_p_absolute = t_min + T_p_relative
    inv = local_result.hess_inv.todense()
    dP = sqrt(inv[0][0])
    de = sqrt(inv[1][1])
    dT_p = sqrt(inv[2][2])

    #final model
    model = KeplerianModel(t, v, dv, e_best, T_p_relative, P_best)
    M = model.mean_anomaly()
    E = np.array([model.eccentric_anomaly(m, e_best) for m in M])
    nu = model.true_anomaly(E)
    gamma, A, B = model.solve(nu)
    v_model = model.radial_velocity(nu, gamma, A, B)
    #errors
    dg, dA, dB, covAB = model.errors()

    K = np.sqrt(A**2 + B**2)
    omega = np.arctan2(-B, A)
    dK = sqrt((A*dA/K)**2 + (B*dB/K)**2 + ((2*A*B)/(K**2)*covAB))
    dw = sqrt(((B*dA)/(K**2))**2 + ((A*dB)/K**2)**2 - 2*((A*B)/(K**4))*covAB)

    return {
        'P': P_best,
        'e': e_best,
        'T_p': T_p_absolute,
        'K': K,
        'omega': omega,
        'gamma': gamma,
        'DE_result': DE_result,
        'local_result': local_result
    }