import numpy as np
from cadence.synthetic_data import synthetic_data
from degradation_tests.core.test_pipeline import Pipeline
from joblib import Parallel, delayed

csv_filename = 'sparse_sampling.csv'

with open(csv_filename, 'w') as f:
    f.write("test_type,N,baseline_days,true_P,fit_P,LS_power,true_K,fit_K,true_e,fit_e,Status\n")

TRUE_PARAMS = {
    'P': 21.1668, 'e': 0.65, 'T_p': 12.0,
    'K': 45.0, 'omega': 1.2, 'gamma': 5.0,
    'N': 6, 'baseline_days': 200, 'noise_base': 0.5
}
ITERATIONS_PER_STEP = 70  # From graph analysis. Higher numbers provide less accuracy for computation time spent.

observations = [100, 80, 60, 40, 30, 20, 15, 10, 8, 7, 6, 5]
baseline_days_array = np.linspace(200, 5, 40)


def run_single_trial(params, test_type):
    t, v, dv = synthetic_data(**params)

    try:
        pl = Pipeline(t, v, dv)
        fit_P, ls_power = pl.apply_LS()
        results = pl.fitting_3var()
        status = "Success"
    except Exception:
        fit_P, ls_power = np.nan, np.nan
        results = {'P': np.nan, 'K': np.nan, 'e': np.nan}
        status = "Failure"

    return (
        f"{test_type},{params['N']},{params['baseline_days']},"
        f"{params['P']:.4f},{fit_P:.4f},{ls_power:.2f},"
        f"{params['K']:.2f},{results['K']:.2f},"
        f"{params['e']:.3f},{results['e']:.3f},{status}\n"
    )


# Test 1 ------
def observational_test(observations, test_type='Observational'):
    '''For a baseline of 200 days, we see how many observations are required for accurate results'''
    print("RUNNING TEST 1 (N-Stepdown)...")
    TRUE_PARAMS['baseline_days'] = 200  # Reset baseline to default

    for N in observations:
        TRUE_PARAMS['N'] = N

        rows = Parallel(n_jobs=10)(
            delayed(run_single_trial)(TRUE_PARAMS.copy(), test_type)
            for _ in range(ITERATIONS_PER_STEP)
        )

        with open(csv_filename, 'a') as f:
            f.writelines(rows)


# Test 2 ------
def base_line_test(baseline_days_array, test_type='Baseline_days'):
    '''we have 5 orbital parameters to fit, we need atleast 6 observations for the accurate results.
    We study how much these 6 observations can be spread apart on "baseline_days" '''
    print('RUNNING TEST 2 (Baseline Pinch)...')
    TRUE_PARAMS['N'] = 6

    for d in baseline_days_array:
        TRUE_PARAMS['baseline_days'] = d

        rows = Parallel(n_jobs=10)(
            delayed(run_single_trial)(TRUE_PARAMS.copy(), test_type)
            for _ in range(ITERATIONS_PER_STEP)
        )

        with open(csv_filename, 'a') as f:
            f.writelines(rows)


if __name__ == "__main__":
    observational_test(observations)
    base_line_test(baseline_days_array)
    print("All Information Floor tests completed.")