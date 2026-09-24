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
from degradation_tests.cadence.synthetic_data import synthetic_data

# --------------------------------------------------------
# CONSTANTS: The Edge of the Information Floor
# --------------------------------------------------------

TRUE_PARAMS = {
    'P': 21.1668,
    'T_p': 12.0,
    'K': 45.0,
    'omega': 1.2,
    'gamma': 5.0,
    'baseline_days': 200, 
    'noise_base': 0.5
}

# N, e - 3 cases
baseline = (80, 0.25) 
mod = (50, 0.5)
extreme = (20, 0.85)
cases = [baseline, mod, extreme]

# Number of Monte Carlo iterations to test
N_MC_VALUES = np.arange(10, 101, 10) 
OUTER_REPEATS = 20 # How many times to repeat the MC batch to find its stability

csv_name = "mc_convergence_ssl.csv"

# --------------------------------------------------------
# FUNCTIONS
# --------------------------------------------------------

def run_single_fit(params):
    t, v, dv = synthetic_data(**params)
    try:
        pl = Pipeline(t, v, dv)
        pl.apply_LS()
        results = pl.fitting_3var()
        return results['P'], results['K'], results['e']
    except Exception:
        # If the points bunch up and crash the matrix, return NaNs
        return np.nan, np.nan, np.nan

def run_repeat(params, n_mc):
    """Runs a batch of n_mc fits and returns the median of P, K, and e."""
    results = [run_single_fit(params) for _ in range(n_mc)]
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
        f.write("Regime_N,Regime_e,N_MC,P_mean,P_sem,K_mean,K_sem,e_mean,e_sem\n")

    print("Running Multi-Regime Convergence Test...")
    
    # 1. Generate all required execution arguments upfront
    tasks = []
    for N_val, e_val in cases:
        TP = TRUE_PARAMS.copy()
        TP["N"] = N_val
        TP["e"] = e_val
        for N_MC in N_MC_VALUES:
            for _ in range(OUTER_REPEATS):
                tasks.append((TP, N_MC, N_val, e_val))
                
    print(f"Total parallel batches to run: {len(tasks)}")

    # 2. Run ONE continuous parallel pool
    # This prevents worker starvation; as soon as a core is free, it grabs the next task.
    raw_results = Parallel(n_jobs=-2, batch_size='auto')(
        delayed(run_repeat)(t[0], t[1]) for t in tasks
    )

    # 3. Aggregate and write to CSV
    # Rebuild the DataFrame from the flattened results
    records = []
    for i, (TP, N_MC, N_val, e_val) in enumerate(tasks):
        P_med, K_med, e_med = raw_results[i]
        records.append([N_val, e_val, N_MC, P_med, K_med, e_med])
        
    df_raw = pd.DataFrame(records, columns=['Regime_N', 'Regime_e', 'N_MC', 'P_med', 'K_med', 'e_med'])
    
    # Calculate mean and SEM across the 20 repeats
    summary = df_raw.groupby(['Regime_N', 'Regime_e', 'N_MC']).agg(
        P_mean=('P_med', 'mean'),
        P_sem=('P_med', lambda x: np.std(x, ddof=1) / np.sqrt(OUTER_REPEATS)),
        K_mean=('K_med', 'mean'),
        K_sem=('K_med', lambda x: np.std(x, ddof=1) / np.sqrt(OUTER_REPEATS)),
        e_mean=('e_med', 'mean'),
        e_sem=('e_med', lambda x: np.std(x, ddof=1) / np.sqrt(OUTER_REPEATS))
    ).reset_index()

    summary.to_csv(csv_name, index=False)
    print("\nFinished. Generating plots...")

    # --------------------------------------------------------
    # PLOTTING
    # --------------------------------------------------------

    df = pd.read_csv(csv_name)

    # Calculate Normalized Errors for combined plotting
    df['P_err_pct'] = (np.abs(df['P_mean'] - TRUE_PARAMS['P']) / TRUE_PARAMS['P']) * 100
    df['K_err_pct'] = (np.abs(df['K_mean'] - TRUE_PARAMS['K']) / TRUE_PARAMS['K']) * 100
    df['e_err_abs'] = np.abs(df['e_mean'] - df['Regime_e'])
    
    # Scale SEM to match the error metrics
    df['P_sem_pct'] = (df['P_sem'] / TRUE_PARAMS['P']) * 100
    df['K_sem_pct'] = (df['K_sem'] / TRUE_PARAMS['K']) * 100

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = ['blue', 'orange', 'red']
    markers = ['o', 's', '^']

    # Group by the specific (N, e) regime and plot
    for idx, (name, group) in enumerate(df.groupby(['Regime_N', 'Regime_e'])):
        n_val, e_val = name
        label = f"N={n_val}, e={e_val}"
        c, m = colors[idx], markers[idx]
        
        axs[0].errorbar(group["N_MC"], group["P_err_pct"], yerr=group["P_sem_pct"], fmt=f'{m}-', capsize=4, color=c, label=label)
        axs[1].errorbar(group["N_MC"], group["K_err_pct"], yerr=group["K_sem_pct"], fmt=f'{m}-', capsize=4, color=c, label=label)
        axs[2].errorbar(group["N_MC"], group["e_err_abs"], yerr=group["e_sem"], fmt=f'{m}-', capsize=4, color=c, label=label)

    # Formatting Plot 1: Period
    axs[0].set_title("Period Convergence (% Error)")
    axs[0].set_ylabel("Median P Error (%) ± SEM")

    # Formatting Plot 2: Amplitude
    axs[1].set_title("Amplitude (K) Convergence (% Error)")
    axs[1].set_ylabel("Median K Error (%) ± SEM")

    # Formatting Plot 3: Eccentricity
    axs[2].set_title("Eccentricity (e) Convergence (Absolute Error)")
    axs[2].set_ylabel("Median e Absolute Error ± SEM")

    for ax in axs:
        ax.set_xlabel("Number of MC Iterations")
        ax.axhline(0, color='k', linestyle='--', alpha=0.7, label='Perfect Fit')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle("Monte Carlo Convergence Across Difficulty Regimes", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mc_convergence_ssl.png', dpi=300)
    plt.show()