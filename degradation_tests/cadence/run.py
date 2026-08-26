import numpy as np
from degradation_tests.core.test_pipeline import Pipeline
from degradation_tests.cadence.synthetic_data import synthetic_data, seasonal_mask, lunar_mask, phase_mask

# 1. Define Constant True Universe
TRUE_PARAMS = {
    'P': 21.1668, 'e': 0.85, 'T_p': 12.0, 
    'K': 45.0, 'omega': 1.2, 'gamma': 5.0,
    'N': 80, 'baseline_days': 200, 'noise_base': 1.0
}
ITERATIONS_PER_STEP = 50 # Monte Carlo smoothing

# 2. Setup CSV File
csv_filename = "cadence_results_e085.csv"

def seasonal_run(csv_filename):
    for gap_width in np.linspace(0, 150, 15): # Sweep from 0 to 150 days
            for _ in range(ITERATIONS_PER_STEP):
                # A. Generate & Mask
                t, v, dv = synthetic_data(**TRUE_PARAMS)
                t_m, v_m, dv_m = seasonal_mask(t, v, dv, gap_width)
                
                if len(t_m) < 10: 
                    continue # Skip if we deleted too much data for matrix math
                    
                # B. Run Pipeline
                pl = Pipeline(t_m, v_m, dv_m)
                fit_P, ls_power = pl.apply_LS()
                results = pl.fitting_3var()
                
                # C. Save Data Immediately
                with open(csv_filename, 'a') as f:
                    f.write(f"Seasonal,{gap_width:.2f},{len(t_m)},"
                            f"{TRUE_PARAMS['P']:.4f},{results['P']:.4f},{ls_power:.2f},"
                            f"{TRUE_PARAMS['K']:.2f},{results['K']:.2f},"
                            f"{TRUE_PARAMS['e']:.3f},{results['e']:.3f}\n")

def lunar_run(csv_filename):
    for blackout_days in np.linspace(0,14,15):
            for _ in range(ITERATIONS_PER_STEP):
                t, v, dv = synthetic_data(**TRUE_PARAMS)
                t_m, v_m, dv_m = lunar_mask(t, v, dv, blackout_days)
    
                if len(t_m) < 10: 
                    continue # Skip if we deleted too much data for matrix math
                    
                pl = Pipeline(t_m, v_m, dv_m)
                fit_P, ls_power = pl.apply_LS()
                results = pl.fitting_3var()
    
                with open(csv_filename, 'a') as f:
                    f.write(f"Lunar,{blackout_days:.2f},{len(t_m)},"
                            f"{TRUE_PARAMS['P']:.4f},{results['P']:.4f},{ls_power:.2f},"
                            f"{TRUE_PARAMS['K']:.2f},{results['K']:.2f},"
                            f"{TRUE_PARAMS['e']:.3f},{results['e']:.3f}\n")

def phase_run(csv_filename):
     for gap_fraction in np.linspace(0,0.5,11):
             for _ in range(ITERATIONS_PER_STEP):
                 t, v, dv = synthetic_data(**TRUE_PARAMS)
                 t_m, v_m, dv_m = phase_mask(t, v, dv, TRUE_PARAMS['P'], TRUE_PARAMS['T_p'], gap_fraction)
     
                 if len(t_m) < 10: 
                     continue # Skip if we deleted too much data for matrix math
                     
                 pl = Pipeline(t_m, v_m, dv_m)
                 fit_P, ls_power = pl.apply_LS()
                 results = pl.fitting_3var()
     
                 with open(csv_filename, 'a') as f:
                     f.write(f"Phase,{gap_fraction:.2f},{len(t_m)},"
                             f"{TRUE_PARAMS['P']:.4f},{results['P']:.4f},{ls_power:.2f},"
                             f"{TRUE_PARAMS['K']:.2f},{results['K']:.2f},"
                             f"{TRUE_PARAMS['e']:.3f},{results['e']:.3f}\n")

if __name__=="__main__":
    with open(csv_filename, 'w') as f:
        f.write("test_type,independent_var,N_left,true_P,fit_P,LS_power,true_K,fit_K,true_e,fit_e\n")

    # 3. Test 1: Seasonal Gap Loop
    print("Running Seasonal Gap Tests...")
    seasonal_run(csv_filename)

    print("Running Lunar Gap Tests...")
    lunar_run(csv_filename)

    print("Running Phase Starvation Tests...")
    phase_run(csv_filename)

    print("All tests complete. Data saved to CSV.")