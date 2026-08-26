from keplerian_fit import KeplerianModel
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt

def get_data(filename):
    df = pd.read_csv(filename)
    v = df["Radial Velocity (m/s)"].to_numpy()
    dv = df["Radial Velocity Uncertainty (m/s)"].to_numpy()
    t_min = df["HJD (days)"].min()
    df["Days"] = df["HJD (days)"] - t_min
    t = df["Days"].to_numpy()
    return t, v, dv, t_min

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

def plot_fit_6var(t, v, dv, P, e, T_p, K, omega, gamma, title_suffix=""):
    # True phase folding: Phase 0 is now exactly at Periastron
    phases = ((t - T_p) / P) % 1.0
    
    phase_smooth = np.linspace(0, 1, 500)
    # Shift t_smooth so the model matches the phase-folded data
    t_smooth = phase_smooth * P + T_p 
    
    M = 2 * np.pi * (t_smooth - T_p) / P
    E = np.array([KeplerianModel.eccentric_anomaly(m, e) for m in M])
    
    num = np.sqrt(1 + e) * np.sin(E / 2)
    den = np.sqrt(1 - e) * np.cos(E / 2)
    nu = 2 * np.arctan2(num, den)
    
    v_model = gamma + K * (np.cos(nu + omega) + e * np.cos(omega))
    
    plt.figure(figsize=(8, 5))
    plt.errorbar(phases, v, yerr=dv, fmt='ko', ecolor='gray', capsize=2, label='Data', zorder=1)
    plt.errorbar(phases + 1.0, v, yerr=dv, fmt='ko', alpha=0.3, zorder=1)
    plt.plot(phase_smooth, v_model, 'r-', linewidth=2, label='Keplerian Fit', zorder=2)
    plt.plot(phase_smooth + 1.0, v_model, 'r-', linewidth=2, alpha=0.3, zorder=2)
    
    plt.xlabel(r'Orbital Phase ($\phi$) - Periastron at $\phi=0$')
    plt.ylabel(r'Radial Velocity (m/s)')
    plt.title(f'RV Fit - {title_suffix} ($P = {P:.4f}$ d, $e = {e:.4f}$)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, 2)
    plt.legend()
    plt.show()

def fit_general(filename, P_guess):
    t, v, dv, t_min = get_data(filename)
    
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
    
    print("Running 6-variable Differential Evolution...")
    DE_result = differential_evolution(
        chi2_6var, 
        args=(t, v, dv), 
        bounds=bounds,
        maxiter=2000,
        popsize=20, # slightly larger population for 6 variables
        disp=False
    )

    print("\nRunning 6-variable BFGS local refinement...")
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

    print(f"\nFinal orbital parameters:")
    print(f"P = {P:.4f} +/- {dP:.4f} days")
    print(f"e = {e:.6f} +/- {de:.6f}")
    print(f"T_p = {T_p_absolute:.4f} +/- {dT_p:.4f} days")
    print(f"K = {K:.2f} +/- {dK:.2f} m/s")
    print(f"omega = {omega:.3f} +/- {dw:.3f} rad ({(np.degrees(omega)%360):.1f} +/- {np.degrees(dw):.1f}°)")
    print(f"gamma = {gamma:.2f} +/- {dg:.2f} m/s")
    print(f"\nFinal chi2: {local_result.fun:.2f}")

    dataset_name = filename.split('/')[-1].split('.')[0].capitalize()
    plot_fit_6var(t, v, dv, P, e, T_p_relative, K, omega, gamma, title_suffix=dataset_name)

    return local_result

#fit_general("data.csv", 21.1668)