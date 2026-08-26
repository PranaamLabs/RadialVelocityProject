import numpy as np

def solve_kepler(M, e, tol=1e-12, max_iter=100):
    """Simple Newton-Raphson to generate the true RV curve."""
    M = (M + np.pi) % (2 * np.pi) - np.pi
    E = M.copy()
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        df = 1.0 - e * np.cos(E)
        delta = f / df
        E -= delta
        if np.all(np.abs(delta) < tol):
            break
    return E

def synthetic_data(P, e, T_p, K, omega, gamma, N=60, baseline_days=100, noise_base=0.5):
    """
    Creates synthetic radial velocity data with unequal gaps and noise.
    """
    # 1. Unequal time gaps (random uniform distribution sorted)
    t = np.sort(np.random.uniform(0, baseline_days, N))
    
    # 2. True kinematics
    M = 2 * np.pi * (t - T_p) / P
    E = solve_kepler(M, e)
    
    num = np.sqrt(1 + e) * np.sin(E / 2)
    den = np.sqrt(1 - e) * np.cos(E / 2)
    nu = 2 * np.arctan2(num, den)
    
    v_true = gamma + K * (np.cos(nu + omega) + e * np.cos(omega))
    
    # 3. Apply Noise
    # Vary the uncertainty slightly point-to-point
    dv = np.random.uniform(noise_base * 0.8, noise_base * 1.2, N)
    # Scatter the true velocities according to a Gaussian distribution
    v_obs = v_true + np.random.normal(0, dv)

    return t, v_obs, dv

def seasonal_mask(t, v, dv, gap_width_days):
    '''Cuts out a seasonal block of time from the middle of the observation baseline'''
    mid_time = (t.max() - t.min()) / 2.0
    half_gap = gap_width_days / 2.0
    
    mask = (t < (mid_time - half_gap)) | (t > (mid_time + half_gap))
    
    return t[mask], v[mask], dv[mask]

def lunar_mask(t, v, dv, blackout_days):
    mask = t%29.5 > blackout_days
    return t[mask], v[mask], dv[mask]

def phase_mask(t, v, dv, P_true, T_p_true, gap_fraction):
    '''Removes a symmetric chunk of data centered around periastron (phase 0.0)'''
    phases = ((t - T_p_true) / P_true) % 1.0
    
    # Example: If gap_fraction is 0.2, we remove 0.9 to 1.0, and 0.0 to 0.1.
    half_gap = gap_fraction / 2.0
    mask = (phases > half_gap) & (phases < (1.0 - half_gap))
    
    return t[mask], v[mask], dv[mask]

