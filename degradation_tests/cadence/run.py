import numpy as np
from joblib import Parallel, delayed
from degradation_tests.core.test_pipeline import Pipeline
from degradation_tests.cadence.synthetic_data import synthetic_data, seasonal_mask, lunar_mask, phase_mask

# 1. Define Constant True Universe
TRUE_PARAMS = {
    'P': 21.1668, 'e': 0.85, 'T_p': 12.0, 
    'K': 45.0, 'omega': 1.2, 'gamma': 5.0,
    'N': 80, 'baseline_days': 200, 'noise_base': 1.0
}
ITERATIONS_PER_STEP = 50
CORES = 8
csv_filename = "cadence_results_e085.csv"

# --- Independent Worker Functions ---

def _process_seasonal(gap_width, current_params):
    t, v, dv = synthetic_data(**current_params)
    t_m, v_m, dv_m = seasonal_mask(t, v, dv, gap_width)
    
    if len(t_m) < 10: 
        return None
        
    pl = Pipeline(t_m, v_m, dv_m)
    fit_P, ls_power = pl.apply_LS()
    results = pl.fitting_3var()
    
    return (f"Seasonal,{gap_width:.2f},{len(t_m)},"
            f"{current_params['P']:.4f},{results['P']:.4f},{ls_power:.2f},"
            f"{current_params['K']:.2f},{results['K']:.2f},"
            f"{current_params['e']:.3f},{results['e']:.3f}\n")

def _process_lunar(blackout_days, current_params):
    t, v, dv = synthetic_data(**current_params)
    t_m, v_m, dv_m = lunar_mask(t, v, dv, blackout_days)

    if len(t_m) < 10: 
        return None
        
    pl = Pipeline(t_m, v_m, dv_m)
    fit_P, ls_power = pl.apply_LS()
    results = pl.fitting_3var()

    return (f"Lunar,{blackout_days:.2f},{len(t_m)},"
            f"{current_params['P']:.4f},{results['P']:.4f},{ls_power:.2f},"
            f"{current_params['K']:.2f},{results['K']:.2f},"
            f"{current_params['e']:.3f},{results['e']:.3f}\n")

def _process_phase(gap_fraction, current_params):
    t, v, dv = synthetic_data(**current_params)
    t_m, v_m, dv_m = phase_mask(t, v, dv, current_params['P'], current_params['T_p'], gap_fraction)

    if len(t_m) < 10: 
        return None
        
    pl = Pipeline(t_m, v_m, dv_m)
    fit_P, ls_power = pl.apply_LS()
    results = pl.fitting_3var()

    return (f"Phase,{gap_fraction:.2f},{len(t_m)},"
            f"{current_params['P']:.4f},{results['P']:.4f},{ls_power:.2f},"
            f"{current_params['K']:.2f},{results['K']:.2f},"
            f"{current_params['e']:.3f},{results['e']:.3f}\n")

# --- Retained API for External Imports ---

def seasonal_run(csv_filename, current_params):
    tasks = [gw for gw in np.linspace(0, 150, 15) for _ in range(ITERATIONS_PER_STEP)]
    results = Parallel(n_jobs=CORES)(delayed(_process_seasonal)(gw, current_params) for gw in tasks)
    
    with open(csv_filename, 'a') as f:
        for res in results:
            if res is not None:
                f.write(res)

def lunar_run(csv_filename, current_params):
    tasks = [bd for bd in np.linspace(0, 14, 15) for _ in range(ITERATIONS_PER_STEP)]
    results = Parallel(n_jobs=CORES)(delayed(_process_lunar)(bd, current_params) for bd in tasks)
    
    with open(csv_filename, 'a') as f:
        for res in results:
            if res is not None:
                f.write(res)

def phase_run(csv_filename, current_params):
    tasks = [gf for gf in np.linspace(0, 0.5, 11) for _ in range(ITERATIONS_PER_STEP)]
    results = Parallel(n_jobs=CORES)(delayed(_process_phase)(gf, current_params) for gf in tasks)
    
    with open(csv_filename, 'a') as f:
        for res in results:
            if res is not None:
                f.write(res)

# Optional standalone execution block
if __name__=="__main__":
    with open(csv_filename, 'w') as f:
        f.write("test_type,independent_var,N_left,true_P,fit_P,LS_power,true_K,fit_K,true_e,fit_e\n")

    print("Running Seasonal Gap Tests...")
    seasonal_run(csv_filename, TRUE_PARAMS)

    print("Running Lunar Gap Tests...")
    lunar_run(csv_filename, TRUE_PARAMS)

    print("Running Phase Starvation Tests...")
    phase_run(csv_filename, TRUE_PARAMS)

    print("All tests complete.")