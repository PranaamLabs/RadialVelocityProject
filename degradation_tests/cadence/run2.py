import run
import numpy as np
import os

TP = run.TRUE_PARAMS

e_space = np.linspace(0,0.95,20)
os.mkdir('cadence_CSVs')

for e in e_space:
    TP['e'] = e
    csv_filename = f'cadence_CSVs/cadence_tests_e={e*100:.2f}.csv'
    
    with open(csv_filename, 'w') as f:
        f.write("test_type,independent_var,N_left,true_P,fit_P,LS_power,true_K,fit_K,true_e,fit_e\n")
        
    run.seasonal_run(csv_filename)
    run.lunar_run(csv_filename)
    run.phase_run(csv_filename)