import numpy as np
import os
from joblib import Parallel, delayed
from degradation_tests.cadence.synthetic_data import synthetic_data
from degradation_tests.core import test_pipeline

ITERS_PER_STEP = 50 #i'll just assume it to be close to this rather than running a convergence test for this, its simply time consuming.

e_space = np.linspace(0, 0.95, 20)
noise_base_space = np.linspace(0.5, 58.5, 30)

TRUE_PARAMS = {
    'P': 21.1668, 'e': 0, 'T_p': 12.0,
    'K': 45.0, 'omega': 1.2, 'gamma': 5.0,
    'baseline_days': 200, 'noise_base': 0,
    'N': 80
}

def single_run(e, n, current_params):
    current_params['e'] = e
    current_params['noise_base'] = n
    
    t, v, dv = synthetic_data(**current_params)
    
    try:
        pl = test_pipeline.Pipeline(t, v, dv)
        fit_P, _ = pl.apply_LS()
        results = pl.fitting_3var()
        
        return f"{e},{n},{results['P']},{results['K']},{results['T_p']},{results['e']},{results['omega']},{results['gamma']},{results['dP']},{results['dK']},{results['dT_p']},{results['de']},{results['dw']},{results['dg']}"

    except Exception:
        return None

if __name__ == '__main__':
    save_dir = 'error_analysis_CSVs'
    os.makedirs(save_dir, exist_ok=True)
     
    for e in e_space:
        csv_filename = f'{save_dir}/error_e={e*100:.0f}.csv'
        if os.path.exists(csv_filename):
            continue
            
        current_params = TRUE_PARAMS.copy()

        with open(csv_filename, 'w') as f:
            f.write('e,noise_base,fit_P,fit_K,fit_T_p,fit_e,fit_w,fit_g,dP,dK,dT_p,de,dw,dg\n')

        print(f"Processing e={e:.3f}...")
        tasks = [N for N in noise_base_space for _ in range(ITERS_PER_STEP)]
        results = Parallel(n_jobs=-2)(delayed(single_run)(e, N, current_params) for N in tasks)
        
        with open(csv_filename, 'a') as f:
            for res in results:
                if res: 
                    f.write(res + '\n')