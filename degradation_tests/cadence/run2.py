from . import run
import numpy as np
import os

TP = run.TRUE_PARAMS

e_space = np.linspace(0,0.95,20)
os.makedirs('cadence_CSVs', exist_ok=True)

for e in e_space:
    csv_filename = f'cadence_CSVs/cadence_tests_e={e*100:.0f}.csv'

    if os.path.exists(csv_filename):
        print(f"Skipping e={e:.3f}: File already exists.")
        continue
        
    print(f"Processing e={e:.3f}...")

    current_params = TP.copy()
    current_params['e'] = e
    
    with open(csv_filename, 'a') as f:
        f.write("test_type,independent_var,N_left,true_P,fit_P,LS_power,true_K,fit_K,true_e,fit_e\n")

    print("seasonal")
    run.seasonal_run(csv_filename, current_params)
    print("lunar")
    run.lunar_run(csv_filename, current_params)
    print("phase")
    run.phase_run(csv_filename, current_params)
    print(f"done for {e:.3f}")