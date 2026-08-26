import os

# Limit threads for joblib parallel processing efficiency
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from degradation_tests.core.test_pipeline import Pipeline
from cadence.synthetic_data import synthetic_data

# --------------------------------------------------------
# CONSTANTS: The Edge of the Information Floor
# --------------------------------------------------------

TRUE_PARAMS = {
    'P': 21.1668,
    'e': 0.65,
    'T_p': 12.0,
    'K': 45.0,
    'omega': 1.2,
    'gamma': 5.0,
    'N': 12,              # The critical threshold: High variance, likely to have errors
    'baseline_days': 200, 
    'noise_base': 0.5
}

# Number of Monte Carlo iterations to test
N_MC_VALUES = np.arange(10, 101, 10) 
OUTER_REPEATS = 20 # How many times to repeat the MC batch to find its stability

csv_name = "mc_convergence_ssl.csv"

# --------------------------------------------------------
# FUNCTIONS
# --------------------------------------------------------

def run_single_fit():
    t, v, dv = synthetic_data(**TRUE_PARAMS)
    try:
        pl = Pipeline(t, v, dv)
        pl.apply_LS()
        results = pl.fitting_3var()
        return results['P'], results['K'], results['e']
    except Exception:
        # If the 12 points bunch up and crash the matrix, return NaNs
        return np.nan, np.nan, np.nan

def run_repeat(n_mc):
    """Runs a batch of n_mc fits and returns the median of P, K, and e."""
    results = [run_single_fit() for _ in range(n_mc)]
    results = np.array(results)
    
    # Use nanmedian to ignore the occasional matrix crashes
    P_med = np.nanmedian(results[:, 0])
    K_med = np.nanmedian(results[:, 1])
    e_med = np.nanmedian(results[:, 2])
    
    return P_med, K_med, e_med

# --------------------------------------------------------
# EXECUTION & LOGGING
# --------------------------------------------------------

if __name__ == "__main__":
    with open(csv_name, "w") as f:
        f.write("N_MC,P_mean,P_sem,K_mean,K_sem,e_mean,e_sem\n")

    print(f"Running Convergence Test at N = {TRUE_PARAMS['N']} points...")

    for N_MC in N_MC_VALUES:
        print(f"Testing N_MC = {N_MC}...")
        
        # Run OUTER_REPEATS batches in parallel
        results = Parallel(n_jobs=-1)(
            delayed(run_repeat)(N_MC)
            for _ in range(OUTER_REPEATS)
        )
        
        # Extract the medians from the parallel runs
        median_P_list = np.asarray([r[0] for r in results])
        median_K_list = np.asarray([r[1] for r in results])
        median_e_list = np.asarray([r[2] for r in results])

        # Calculate means and standard errors of those medians
        P_mean = np.nanmean(median_P_list)
        P_sem  = np.nanstd(median_P_list, ddof=1) / np.sqrt(OUTER_REPEATS)
        
        K_mean = np.nanmean(median_K_list)
        K_sem  = np.nanstd(median_K_list, ddof=1) / np.sqrt(OUTER_REPEATS)
        
        e_mean = np.nanmean(median_e_list)
        e_sem  = np.nanstd(median_e_list, ddof=1) / np.sqrt(OUTER_REPEATS)

        with open(csv_name, "a") as f:
            f.write(f"{N_MC},{P_mean:.6f},{P_sem:.6f},{K_mean:.6f},{K_sem:.6f},{e_mean:.6f},{e_sem:.6f}\n")

    print("\nFinished. Generating plots...")

    # --------------------------------------------------------
    # PLOTTING
    # --------------------------------------------------------

    df = pd.read_csv(csv_name)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Period Plot
    axs[0].errorbar(df["N_MC"], df["P_mean"], yerr=df["P_sem"], fmt='o-', capsize=4, color='red')
    axs[0].axhline(TRUE_PARAMS['P'], color='k', linestyle='--', label=f"True P = {TRUE_PARAMS['P']}")
    axs[0].set_title("Period Convergence")
    axs[0].set_xlabel("Number of MC Iterations")
    axs[0].set_ylabel("Median P ± SEM")
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # 2. Amplitude Plot
    axs[1].errorbar(df["N_MC"], df["K_mean"], yerr=df["K_sem"], fmt='s-', capsize=4, color='purple')
    axs[1].axhline(TRUE_PARAMS['K'], color='k', linestyle='--', label=f"True K = {TRUE_PARAMS['K']}")
    axs[1].set_title("Amplitude (K) Convergence")
    axs[1].set_xlabel("Number of MC Iterations")
    axs[1].set_ylabel("Median K ± SEM")
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.6)

    # 3. Eccentricity Plot
    axs[2].errorbar(df["N_MC"], df["e_mean"], yerr=df["e_sem"], fmt='^-', capsize=4, color='green')
    axs[2].axhline(TRUE_PARAMS['e'], color='k', linestyle='--', label=f"True e = {TRUE_PARAMS['e']}")
    axs[2].set_title("Eccentricity (e) Convergence")
    axs[2].set_xlabel("Number of MC Iterations")
    axs[2].set_ylabel("Median e ± SEM")
    axs[2].legend()
    axs[2].grid(True, linestyle='--', alpha=0.6)

    plt.suptitle(f"Monte Carlo Convergence Test (Info Floor limit: N={TRUE_PARAMS['N']})", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mc_convergence_ssl.png', dpi=300)
    plt.show()