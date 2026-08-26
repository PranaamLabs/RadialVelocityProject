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

def chi2_3var(params, t,v,dv):
    P, e, T_p = params
    model = KeplerianModel(t,v,dv,e,T_p,P)
    M = model.mean_anomaly()
    E = np.array([model.eccentric_anomaly(m,e) for m in M])
    nu = model.true_anomaly(E)
    gamma, A, B = model.solve(nu)
    v_model = model.radial_velocity(nu, gamma, A, B)
    return model.chi_squared(v_model)

def numerical_hessian(params, t, v, dv, rel_step=1e-4):
    """
    Numerically computes the Hessian matrix of chi² at the optimum
    using central finite differences.
    """
    H = np.zeros((3, 3))

    chi0 = chi2_3var(params, t, v, dv)

    # Diagonal terms
    for i in range(3):
        p_plus = params.copy()
        step_i = rel_step * max(abs(params[i]), 1.0)
        p_plus[i] += step_i

        p_minus = params.copy()
        p_minus[i] -= step_i

        chi_plus = chi2_3var(p_plus, t, v, dv)
        chi_minus = chi2_3var(p_minus, t, v, dv)

        H[i, i] = (chi_plus - 2 * chi0 + chi_minus) / step_i**2

    # Off-diagonal terms
    for i in range(3):
        for j in range(i + 1, 3):
            step_i = rel_step * max(abs(params[i]), 1.0)
            step_j = rel_step * max(abs(params[j]), 1.0)

            p_pp = params.copy()
            p_pp[i] += step_i
            p_pp[j] += step_j

            p_pm = params.copy()
            p_pm[i] += step_i
            p_pm[j] -= step_j

            p_mp = params.copy()
            p_mp[i] -= step_i
            p_mp[j] += step_j

            p_mm = params.copy()
            p_mm[i] -= step_i
            p_mm[j] -= step_j

            chi_pp = chi2_3var(p_pp, t, v, dv)
            chi_pm = chi2_3var(p_pm, t, v, dv)
            chi_mp = chi2_3var(p_mp, t, v, dv)
            chi_mm = chi2_3var(p_mm, t, v, dv)

            H[i, j] = (chi_pp - chi_pm - chi_mp + chi_mm) / (4 * step_i * step_j)
            H[j, i] = H[i, j]

    return H

def plot_fit(t, v, dv, P, e, T_p, gamma, A, B, dP, de, title_suffix=""):
    phases = ((t - T_p) / P) % 1.0
    
    phase_smooth = np.linspace(0, 1, 500)
    t_smooth = phase_smooth * P + T_p
    
    M = 2 * np.pi * (t_smooth - T_p) / P
    E = np.array([KeplerianModel.eccentric_anomaly(m, e) for m in M])
    
    num = np.sqrt(1 + e) * np.sin(E / 2)
    den = np.sqrt(1 - e) * np.cos(E / 2)
    nu = 2 * np.arctan2(num, den)
    
    v_model = gamma + A * (np.cos(nu) + e) + B * np.sin(nu)
    
    plt.figure(figsize=(8, 5))
    plt.errorbar(phases, v, yerr=dv, fmt='ko', ecolor='gray', capsize=2, label='Data', zorder=1)
    plt.errorbar(phases + 1.0, v, yerr=dv, fmt='ko', alpha=0.3, zorder=1)
    plt.plot(phase_smooth, v_model, 'r-', linewidth=2, label='Keplerian Fit', zorder=2)
    plt.plot(phase_smooth + 1.0, v_model, 'r-', linewidth=2, alpha=0.3, zorder=2)
    
    plt.xlabel(r'Orbital Phase ($\phi$) - Periastron at $\phi=0$')
    plt.ylabel(r'Radial Velocity (m/s)')
    plt.title(f'RV Fit - {title_suffix} ($P = {P:.4f} ± {dP:.6f}$ d, $e = {e:.4f} ± {de:.3f}$)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, 2)
    plt.legend()
    plt.show()

def fit_3var(filename, P_guess):
    t, v, dv, t_min= get_data(filename)
    bounds = [
        (P_guess * 0.9, P_guess * 1.1), 
        (1e-3, 0.95), 
        (-P_guess, P_guess)
    ]
    DE_result = differential_evolution(
        chi2_3var, 
        args=(t, v, dv), 
        bounds=bounds,
        maxiter=1000,
        popsize=15,
        disp=False
    )
    print(f"Local result: P={DE_result.x[0]:.8f} e={DE_result.x[1]:.6f}, T_p={DE_result.x[2]:.6f}")
    print(f"DE chi2: {DE_result.fun:.2f}")

    local_result = minimize(
        chi2_3var,
        x0=DE_result.x,
        args=(t, v, dv),
        method='BFGS',
        #bounds=bounds,
        options={'maxiter': 1000}
    )
    print(f"Local result: P={local_result.x[0]:.8f} e={local_result.x[1]:.6f}, T_p={local_result.x[2]:.6f}")
    print(f"Local chi2: {local_result.fun:.2f}")
    #print(f'nit: {local_result.nit}')
    #print(local_result.message)
    #print(local_result.hess_inv)

    P_best, e_best, T_p_relative = local_result.x
    T_p_absolute = t_min + T_p_relative
    H = numerical_hessian(local_result.x, t, v, dv)
    #print(f"Hessian:{H}")

    cov = 2 * np.linalg.inv(H)
    #print(f"Cov: {cov}")
    #print(f"Eigenvals: {np.linalg.eigvals(H)}")

    dP = np.sqrt(cov[0, 0])
    de = np.sqrt(cov[1, 1])
    dT_p = np.sqrt(cov[2, 2])

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

    print(f"\nFinal orbital parameters:")
    print(f"P = {P_best:.8f} +/- {dP:.8f} days")
    print(f"e = {e_best:.6f} +/- {de:.6f}")
    print(f"T_p = {T_p_absolute:.4f} +/- {dT_p:.4f} days")
    print(f"K = {K:.2f} +/- {dK:.2f} m/s")
    print(f"omega = {omega:.3f} +/- {dw:.3f} rad ({np.degrees(omega):.1f} +/- {np.degrees(dw):.1f}°)")
    print(f"gamma = {gamma:.2f} +/- {dg:.2f} m/s")
    print(f"A = {A:.2f} +/- {dA:.2f}, B = {B:.2f} +/- {dB:.2f}")

    dataset_name = filename.split('/')[-1].split('.')[0].capitalize()
    plot_fit(t, v, dv, P_best, e_best, T_p_relative, gamma, A, B, dP, de, title_suffix=dataset_name)

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

#fit("data.csv", 21.1668)
if __name__=="__main__":
    print(f"{"-"*10}Butler data{"-"*10}")
    fit_3var("data_reconstruction/data/csv/butler.csv", 4.2302)
    print(f"{"-"*10}HowardFulton data{"-"*10}")
    fit_3var("data_reconstruction/data/csv/howardfulton.csv", 4.2308)