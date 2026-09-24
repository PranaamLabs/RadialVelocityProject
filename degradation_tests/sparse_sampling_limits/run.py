import numpy as np
from degradation_tests.cadence.synthetic_data import synthetic_data
from degradation_tests.core.test_pipeline import Pipeline
from joblib import Parallel, delayed
import os

csv_filename = 'sparse_sampling.csv'

with open(csv_filename, 'w') as f:
    f.write("test_type,N,baseline_days,true_P,fit_P,LS_power,true_K,fit_K,true_e,fit_e,Status\n")

TRUE_PARAMS = {
    'P': 21.1668, 'e': 0.65, 'T_p': 12.0,
    'K': 45.0, 'omega': 1.2, 'gamma': 5.0,
    'baseline_days': 200, 'noise_base': 0.5
    }
ITERATIONS_PER_STEP = 60  # From graph analysis. Higher numbers provide less accuracy for computation time spent.

observations = [100, 80, 60, 40, 30, 20, 15, 10, 8, 7, 6, 5]
e_space = np.linspace(0,0.95,20)

def _process_sparsity(N, current_params):
    current_params['N'] = N
    t, v, dv = synthetic_data(**current_params)
    try:
        pl = Pipeline(t, v, dv)
        fit_P, ls_power = pl.apply_LS()
        results = pl.fitting_3var()
        return (f"Sparsity,{N},{current_params['P']:.4f},{fit_P:.4f},{ls_power:.2f},"
                f"{current_params['K']:.2f},{results['K']:.2f},"
                f"{current_params['e']:.3f},{results['e']:.3f}\n")
    except Exception:
        return None

if __name__ == "__main__":
    os.makedirs('sparsity_CSVs', exist_ok=True)

    for e in e_space:
        csv_filename = f'sparsity_CSVs/sparsity_e={e*100:.0f}.csv'
        if os.path.exists(csv_filename):
            continue
            
        current_params = TRUE_PARAMS.copy()
        current_params['e'] = e

        with open(csv_filename, 'w') as f:
            f.write("test_type,N,true_P,fit_P,LS_power,true_K,fit_K,true_e,fit_e\n")
            
        print(f"Processing e={e:.3f}...")
        tasks = [N for N in observations for _ in range(ITERATIONS_PER_STEP)]
        results = Parallel(n_jobs=-2)(delayed(_process_sparsity)(N, current_params) for N in tasks)
        
        with open(csv_filename, 'a') as f:
            for res in results:
                if res: f.write(res)