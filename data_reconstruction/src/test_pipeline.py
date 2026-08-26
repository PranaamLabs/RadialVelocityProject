import numpy as np
import pandas as pd
from lombscargle import LombScargle
import fit_3var
import fit_6var
import matplotlib.pyplot as plt

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

def synthetic_data(filename, P, e, T_p, K, omega, gamma, N=60, baseline_days=100, noise_base=0.5):
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
    
    # 4. Save to CSV
    df = pd.DataFrame({
        "HJD (days)": t,
        "Radial Velocity (m/s)": v_obs,
        "Radial Velocity Uncertainty (m/s)": dv
    })
    
    df.to_csv(filename, index=False)
    print(f"Synthetic data saved to {filename}")
    print(f"True Params: P={P}, e={e}, T_p={T_p}, K={K}, omega={omega}, gamma={gamma}")
    return True

class Pipeline:
    def __init__(self,csvfile):
        self.csvfile = csvfile

    def get_data(self):
        self.df = pd.read_csv(self.csvfile)
        self.y = self.df["Radial Velocity (m/s)"].to_numpy()
        self.dy = self.df["Radial Velocity Uncertainty (m/s)"].to_numpy()
        self.df["Days"]= self.df["HJD (days)"] - self.df["HJD (days)"].min()
        self.t = self.df["Days"].to_numpy()
        return True
    
    def _omegas(self, oversampling=50) -> np.ndarray:
        T_span = self.t.max() - self.t.min()
        dt = np.diff(np.sort(self.t))
        f_min = 1.0 / T_span
        f_max = 1.0 / (2.0 * np.median(dt))
        # Better yet, just enforce a high resolution grid:
        N_freq = 10000 
        freqs = np.linspace(f_min, f_max, N_freq)
        return 2 * np.pi * freqs
    
    def apply_LS(self):
        omegas = self._omegas()
        gls = LombScargle(self.t, self.y, self.dy)
        P = gls.periodogram(omegas)
        best_omega = omegas[np.argmax(P)]
        self.period = 2*np.pi/best_omega
        print(f"Period: {self.period:.4f} days")

        plt.plot(omegas, P)
        plt.title("LombScargle Periodogram")
        plt.xlabel(r'Angular Frequency ($\omega$) [rad/d]')
        plt.ylabel('Power')
        plt.axvline(
            best_omega,
            color='red',
            linestyle='--',
            alpha=0.7,
            label=f'Peak: {best_omega:.4f} rad/d'
        )
        plt.legend()
        plt.show()

        return True
    
    def fitting_3var(self):
        fit_3var.fit_3var(self.csvfile, self.period)

    def fitting_6var(self):
        fit_6var.fit_general(self.csvfile, self.period)

if __name__ == "__main__":
    # 1. Define true parameters for the synthetic system
    test_file = "synthetic_rv_data.csv"
    true_P = 21.1668
    true_e = 0.35
    true_Tp = 12.0
    true_K = 45.0
    true_omega = 1.2  # radians
    true_gamma = 5.0
    
    print("--- Generating Synthetic Data (Near-Zero Noise) ---")
    synthetic_data(
        filename=test_file,
        P=true_P, 
        e=true_e, 
        T_p=true_Tp, 
        K=true_K, 
        omega=true_omega, 
        gamma=true_gamma,
        N=80,               
        baseline_days=150,  
        noise_base=6
    )
    
    print("\n--- Initializing RV Pipeline ---")
    pl = Pipeline(test_file)
    pl.get_data()
    
    # 2. Run Generalized Lomb-Scargle to find the period
    print("\n[Step 1] Running Lomb-Scargle Periodogram...")
    pl.apply_LS()
    
    # 3. Run the partially linearized 3-variable optimizer
    print("\n[Step 2] Running Linearized 3-Variable Fit (Fast)...")
    pl.fitting_3var()
    
    # 4. Run the full 6-variable non-linear optimizer
    #print("\n[Step 3] Running Full 6-Variable Non-Linear Fit (Robust)...")
    #pl.fitting_6var()