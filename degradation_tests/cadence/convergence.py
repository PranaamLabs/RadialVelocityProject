import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from degradation_tests.core.test_pipeline import Pipeline
from degradation_tests.cadence.synthetic_data import synthetic_data, phase_mask

# --------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------

TRUE_PARAMS = {
    'P': 21.1668,
    'e': 0.6,
    'T_p': 12.0,
    'K': 45.0,
    'omega': 1.2,
    'gamma': 5.0,
    'N': 80,
    'baseline_days': 200,
    'noise_base': 1.0
}

PHASE_GAP = 0.30

N_VALUES = np.arange(10, 81, 10)

OUTER_REPEATS = 20

csv_name = "mc_convergence_for_cad.csv"

# --------------------------------------------------------
# CSV
# --------------------------------------------------------

with open(csv_name, "w") as f:
    f.write("N_MC,K_mean,K_sem,e_mean,e_sem\n")

# --------------------------------------------------------
# MAIN LOOP
# --------------------------------------------------------

def run_repeat(N_MC):

    K_errors = []
    e_errors = []

    for _ in range(N_MC):

        t, v, dv = synthetic_data(**TRUE_PARAMS)

        t, v, dv = phase_mask(
            t,
            v,
            dv,
            TRUE_PARAMS['P'],
            TRUE_PARAMS['T_p'],
            PHASE_GAP
        )

        if len(t) < 10:
            continue

        pl = Pipeline(t, v, dv)

        pl.apply_LS()
        fit = pl.fitting_3var()

        K_errors.append(abs(fit["K"] - TRUE_PARAMS["K"]))
        e_errors.append(abs(fit["e"] - TRUE_PARAMS["e"]))

    return np.median(K_errors), np.median(e_errors)

for N_MC in N_VALUES:

    print(f"\nTesting N = {N_MC}")

    median_K_list = []
    median_e_list = []

    # Repeat the WHOLE experiment many times
    results = Parallel(
        n_jobs=8,
        backend='loky',
        prefer='processes'
    )(
        delayed(run_repeat)(N_MC)
        for _ in range(OUTER_REPEATS)
    )

    median_K_list = np.asarray([r[0] for r in results])
    median_e_list = np.asarray([r[1] for r in results])

    K_mean = np.mean(median_K_list)
    K_std  = np.std(median_K_list, ddof=1)
    K_sem = K_std / np.sqrt(OUTER_REPEATS)

    e_mean = np.mean(median_e_list)
    e_std  = np.std(median_e_list, ddof=1)
    e_sem = e_std / np.sqrt(OUTER_REPEATS)

    with open(csv_name, "a") as f:
        f.write(
            f"{N_MC},{K_mean:.6f},{K_sem:.6f},{e_mean:.6f},{e_sem:.6f}\n"
        )

print("\nFinished.")

# --------------------------------------------------------
# Plot
# --------------------------------------------------------

df = pd.read_csv(csv_name)

fig, ax = plt.subplots(1,2,figsize=(13,5))

ax[0].errorbar(
    df["N_MC"],
    df["K_mean"],
    yerr=df["K_sem"],
    marker='o',
    capsize=4,
    linewidth=2
)

ax[0].set_title("Monte Carlo Convergence (K)")
ax[0].set_xlabel("Monte Carlo Realizations")
ax[0].set_ylabel("Median |Kfit - Ktrue| (m/s)")
ax[0].grid(alpha=0.3)

ax[1].errorbar(
    df["N_MC"],
    df["e_mean"],
    yerr=df["e_sem"],
    marker='s',
    capsize=4,
    linewidth=2
)

ax[1].set_title("Monte Carlo Convergence (e)")
ax[1].set_xlabel("Monte Carlo Realizations")
ax[1].set_ylabel("Median |efit - etrue|")
ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("mc_convergence_for_cad.png", dpi=300)
plt.show()