from keplerian_fit import KeplerianModel
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
from math import sqrt

def get_data(filename):
    df = pd.read_csv(filename)
    v = df["Radial Velocity (m/s)"].to_numpy()
    dv = df["Radial Velocity Uncertainty (m/s)"].to_numpy()
    t_min = df["HJD (days)"].min()
    df["Days"]= df["HJD (days)"] - t_min
    t = df["Days"].to_numpy()
    return t,v,dv,t_min

def chi2(params, t,v,dv,P):
    e, T_p = params
    model = KeplerianModel(t,v,dv,e,T_p,P)
    M = model.mean_anomaly()
    E = np.array([model.eccentric_anomaly(m,e) for m in M])
    nu = model.true_anomaly(E)
    gamma, A, B = model.solve(nu)
    v_model = model.radial_velocity(nu, gamma, A, B)
    return model.chi_squared(v_model)

def plot_fit(t, v, dv, P, e, T_p, gamma, A, B, title_suffix=""):
    # 1. Phase-fold observed data
    phases = (t / P) % 1.0
    
    # 2. Generate continuous phase and time array for model
    phase_smooth = np.linspace(0, 1, 500)
    t_smooth = phase_smooth * P
    
    # 3. Compute continuous model kinematics
    M = 2 * np.pi * (t_smooth - T_p) / P
    E = np.array([KeplerianModel.eccentric_anomaly(m, e) for m in M])
    
    num = np.sqrt(1 + e) * np.sin(E / 2)
    den = np.sqrt(1 - e) * np.cos(E / 2)
    nu = 2 * np.arctan2(num, den)
    
    v_model = gamma + A * (np.cos(nu) + e) + B * np.sin(nu)
    
    # 4. Plot over 2 periods [0, 2]
    plt.figure(figsize=(8, 5))
    
    # Data
    plt.errorbar(phases, v, yerr=dv, fmt='ko', ecolor='gray', capsize=2, label='Data', zorder=1)
    plt.errorbar(phases + 1.0, v, yerr=dv, fmt='ko', alpha=0.3, zorder=1)
    
    # Model
    plt.plot(phase_smooth, v_model, 'r-', linewidth=2, label='Keplerian Fit', zorder=2)
    plt.plot(phase_smooth + 1.0, v_model, 'r-', linewidth=2, alpha=0.3, zorder=2)
    
    plt.xlabel(r'Orbital Phase ($\phi$)')
    plt.ylabel(r'Radial Velocity (m/s)')
    plt.title(f'RV Fit - {title_suffix} ($P = {P:.4f}$ d, $e = {e:.4f}$)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, 2)
    plt.legend()
    plt.show()

def fit(filename, P):
    t, v, dv, t_min= get_data(filename)
    DE_result = differential_evolution(
        chi2, 
        args=(t, v, dv, P), 
        bounds=[(1e-3, 0.95), (0, P)],
        maxiter=1000,
        popsize=15,
        disp=True
    )
    print(f"DE result: e={DE_result.x[0]:.4f}, T_p={DE_result.x[1]:.4f}")
    print(f"DE chi2: {DE_result.fun:.2f}")

    local_result = minimize(
        chi2,
        x0=DE_result.x,  # Start from DE's best
        args=(t, v, dv, P),
        method='BFGS',
        options={'maxiter': 1000}
    )
    print(f"Local result: e={local_result.x[0]:.6f}, T_p={local_result.x[1]:.6f}")
    print(f"Local chi2: {local_result.fun:.2f}")

    e_best, T_p_best = local_result.x
    inv = local_result.hess_inv
    de = sqrt(inv[0][0])
    dT_p = sqrt(inv[1][1])

    #final model
    model = KeplerianModel(t, v, dv, e_best, T_p_best, P)
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

    print(f"\nFinal orbital parameters:")
    print(f"P = {P:.4f} days")
    print(f"e = {e_best:.6f} +/- {de:.6f}")
    print(f"T_p = {t_min + T_p_best:.4f} +/- {dT_p:.4f} days")
    print(f"K = {K:.2f} +/- {dK:.2f} m/s")
    print(f"omega = {omega:.3f} +/- {dw:.3f} rad ({np.degrees(omega):.1f} +/- {np.degrees(dw):.1f}°)")
    print(f"gamma = {gamma:.2f} +/- {dg:.2f} m/s")
    print(f"A = {A:.2f} +/- {dA:.2f}, B = {B:.2f} +/- {dB:.2f}")

    dataset_name = filename.split('/')[-1].split('.')[0].capitalize()
    plot_fit(t, v, dv, P, e_best, T_p_best, gamma, A, B, title_suffix=dataset_name)

    return {
        'P': P,
        'e': e_best,
        'T_p': T_p_best,
        'K': K,
        'omega': omega,
        'gamma': gamma,
        'DE_result': DE_result,
        'local_result': local_result
    }

print("-------------Butler-------------")
fit("../data/csv/butler.csv", 4.2302)
print("-------------HowardFulton-------------")
fit("../data/csv/howardfulton.csv", 4.2308)