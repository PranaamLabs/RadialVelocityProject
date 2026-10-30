# Radial Velocity Signal Analysis

**A project exploring radial velocity signal reconstruction and parameter recovery under different observational conditions.**

This repository contains my work on reconstructing exoplanet radial velocity signals and studying how observational limitations affect the recovery of orbital parameters.

The project is currently divided into two parts:

1. **Data Reconstruction** – Building a pipeline to analyse radial velocity observations and recover orbital parameters.
2. **Degradation Studies** – Using synthetic data and Monte Carlo simulations to investigate how different observational conditions affect parameter recovery.

The repository has accumulated quite a few experiments and intermediate files along the way, so here's a walkthrough of what is where and what I've done so far.

*Last updated: 30 October 2026*

---

## 1. Data Reconstruction

All the main reconstruction code is in `data_reconstruction/src/`, with the datasets in `data_reconstruction/data/`.

### 1.1 Datasets

I started with two radial velocity datasets of **51 Pegasi b**:

- **HF** – Howard & Fulton (2016)
- **B** – Butler et al. (2006)

The original data and processed versions are stored in `data_reconstruction/data/`.

### 1.2 Lomb–Scargle Periodogram

The first step was identifying the dominant periodicity in the observations.

I implemented a Generalized Lomb–Scargle (GLS) periodogram from scratch in `lombscargle.py`, following the approach described in Zechmeister & Kürster (2009).

I then compared my implementation against `astropy.timeseries.lombscargle()` to check its numerical consistency. The comparison plots and tests are in:

`archive/experiments/reconstruction/lombscargle/`

Applying the implementation to both datasets gave:

- HF: approximately 4.2308 days
- B: approximately 4.2302 days

Both are close to the known orbital period of approximately 4.2308 days.

The Butler dataset also shows additional peaks at higher frequencies, which are likely associated with harmonics and aliases rather than the fundamental orbital period.

### 1.3 Phase Folding

I also implemented phase folding in `phase_fold.py`.

This was useful for getting an initial visual understanding of the periodicity and the approximately sinusoidal appearance of the observations.

The corresponding plots are in:

`archive/experiments/reconstruction/phase_folding/`

### 1.4 Keplerian Fitting

Once I had an estimate of the period, the next step was fitting a Keplerian radial velocity model:

$
v(t)=\gamma+K[\cos(\nu+\omega)+e\cos\omega]
$

where the six parameters to be recovered are:

- \(P\) – Orbital period
- \(e\) – Eccentricity
- \(K\) – Radial velocity semi-amplitude
- \(T_p\) – Time of periastron passage
- \(\gamma\) – Systemic radial velocity
- \(\omega\) – Argument of periastron

I explored two approaches to fitting this model.

**Approach 1: Direct six-parameter optimization**

Implemented in `fit_6var.py`.

Here, I treat the model as a nonlinear optimization problem and use:

- `scipy.optimize.differential_evolution` for global optimization.
- `scipy.optimize.minimize` for local refinement.

The period estimate obtained from the Lomb–Scargle periodogram is used to constrain the optimization bounds.

The uncertainty estimation in this approach is not entirely satisfactory yet, and I intend to revisit it.

**Approach 2: Separating linear and nonlinear parameters**

Implemented in `fit_3var.py`.

I came across the approach discussed in Wright (2009), where the radial velocity equation can be rearranged to separate three linear parameters from the nonlinear ones.

Expanding the model gives:

$
v(t)=\gamma+
K\cos\omega(\cos\nu+e)
-K\sin\omega\sin\nu
$

For fixed \(P\), \(e\), and \(T_p\), this can be written as a linear model in:

$
\gamma,\quad K\cos\omega,\quad -K\sin\omega
$

This allows those three parameters to be obtained through linear regression, while the remaining three parameters (\(P,e,T_p\)) are optimized nonlinearly.

I also implemented Hessian-based uncertainty estimation for the nonlinear parameters using central finite differences, based on a local quadratic approximation to the objective function.

`keplerian_fit.py` contains the underlying calculations for a given set of \(P,e,T_p\), including the mean, eccentric and true anomalies, and the linear coefficients.

The original approach in Wright (2009) deals with multiple planets; I adapted the formulation for a single-planet model.

### 1.5 Reconstruction Results

I applied the linear–nonlinear fitting approach to both the HF and B datasets.

The resulting fits and parameter estimates are stored here:

- Plots: `archive/experiments/reconstruction/keplerian_fit/`
- Fitted parameters: `data_reconstruction/data/fit_data.txt`

---

## 2. Degradation Studies

The second part of the project investigates how different observational limitations affect the recovery of orbital parameters.

The main experimental code is in `degradation_tests/`, with the shared fitting routines in `degradation_tests/core/`.

### 2.1 General Methodology

Most of these experiments follow a similar Monte Carlo-based parameter-recovery procedure.

The general workflow is:

**1. Monte Carlo convergence tests**

Before running the actual experiments, I first investigated how many Monte Carlo realizations would be sufficient for reasonably stable results.

I ran convergence tests under baseline conditions, varying the number of realizations \(N\) and observing how the estimated quantities behaved.

I didn't want to arbitrarily choose a very large \(N\), since a complete experiment can take more than six hours on my laptop. Computational cost had to be considered too.

**2. Synthetic data generation and parameter variation**

For each parameter combination, I generate synthetic radial velocity observations using known input orbital parameters and Gaussian noise.

I then introduce the observational degradation being studied, vary the relevant parameters, and repeat the fitting procedure \(N\) times.

The fitted parameters from each realization are saved to CSV files.

**3. Analysis and visualization**

For each parameter combination, I calculate the mean of the recovered quantities across the Monte Carlo realizations.

Since two parameters are varied at a time, the results can be represented using heat maps:

- X and Y axes represent the two varied parameters.
- The colour represents an error metric or another quantity of interest.

This makes it easier to identify how different combinations of observational conditions affect parameter recovery.

### 2.2 Cadence Testing

Relevant code: `degradation_tests/cadence/`

Main scripts:

- `run.py`
- `convergence.py`
- `heatmaps_plot.py`

I explored three different ways of removing observations to simulate limitations in observational cadence.

**A. Seasonal Masking**

This removes a continuous block of observations from the middle of the observation baseline.

The idea is to simulate situations where observations are unavailable for an extended period, perhaps because an observatory is temporarily unavailable or occupied.

Parameters varied:

- Gap width (days)
- Eccentricity

Quantity studied: Log-normal period error.

The current implementation removes a single block around the middle of the observation baseline. It could be extended to allow multiple gaps or gaps at different locations.

**B. Lunar Masking**

This removes a small chunk of observations periodically, once every approximately 29.5 days.

The idea is to simulate observational restrictions around full moon, when some observations may be avoided.

Parameters varied:

- Lunar blackout duration (days per month)
- Eccentricity

Quantity studied: Peak Lomb–Scargle power at the main frequency.

**C. Phase Starvation**

This is the main cadence experiment.

For eccentric orbits, observations near periastron are particularly important for constraining the shape of the radial velocity curve and recovering parameters such as eccentricity and semi-amplitude.

To investigate this, I remove observations symmetrically around periastron, with half the removed phase interval on either side of phase 0.

Parameters varied:

- Fraction of orbital phase removed near periastron
- Eccentricity

Quantities studied:

- Absolute error in \(K\)
- Absolute error in \(e\)

The Monte Carlo convergence analysis for these experiments is stored in:

`archive/experiments/convergence_tests_mc/cadence/`

The results suggest diminishing returns relative to computational cost beyond approximately \(N=50\).

### 2.3 Sparse Sampling Analysis

Relevant code: `degradation_tests/sparse_sampling_limits/`

Main scripts:

- `run.py`
- `plot.py`
- `convergence.py`

Here, I investigate how the number of observations affects the recovery of orbital parameters.

I vary the number of observations while keeping the observation baseline fixed at 200 days.

Eccentricity is varied as well, allowing me to investigate how sampling requirements depend on orbital eccentricity.

Quantities studied:

- Mean Lomb–Scargle power
- Mean absolute error in \(K\)
- Mean absolute error in \(e\)

The corresponding Monte Carlo convergence plot is in:

`archive/experiments/convergence_tests_mc/sparse_sampling/`

The convergence behaviour appears to plateau around \(N=60\).

### 2.4 Error Accuracy Analysis

Relevant code: `degradation_tests/error_accuracy_analysis/`

Main scripts:

- `run_gaussian_noise.py`
- `plot.py`

This experiment investigates how accurately the fitting procedure estimates parameter uncertainties.

For nonlinear fitting, the uncertainty estimates rely on a local quadratic (second-order Taylor) approximation to the objective function, using the Hessian.

I wanted to compare these estimated errors against the actual variation in recovered parameters across Monte Carlo realizations.

For this, I generate synthetic observations with varying Gaussian noise levels and eccentricities, perform repeated fits, and compare:

- The uncertainty estimated by the fitting procedure.
- The empirical uncertainty obtained from the Monte Carlo realizations.

The relative discrepancy is represented as:

$
\frac{|\sigma_{\mathrm{MC}}-\sigma_{\mathrm{fit}}|}
{\sigma_{\mathrm{fit}}}
$

where \(\sigma_{\mathrm{MC}}\) is the Monte Carlo-based uncertainty and \(\sigma_{\mathrm{fit}}\) is the uncertainty estimated from the fit.

I used \(N=50\) realizations for this experiment as a practical computational choice, rather than performing a separate convergence study.

---

## 3. Repository Structure

Here's a breakdown of the current directory structure.

```text
├── archive/
│   ├── experiments/
│   │   ├── cadence_testing/
│   │   ├── convergence_tests_mc/
│   │   ├── error_accuracy_analysis/
│   │   ├── reconstruction/
│   │   │   ├── keplerian_fit/
│   │   │   ├── lombscargle/
│   │   │   └── phase_folding/
│   │   ├── sparse_sampling_limits/
│   │   └── test_fitting_pipeline/
│   ├── logs/
│   └── notes/
│       └── Goldstein CM/
│
├── data_reconstruction/
│   ├── data/
│   │   └── csv/
│   └── src/
│       ├── apply_ls.py
│       ├── compare_ls.py
│       ├── dataset_metadata.py
│       ├── fit.py
│       ├── fit_3var.py
│       ├── fit_6var.py
│       ├── image_plot.py
│       ├── keplerian_fit.py
│       ├── lombscargle.py
│       ├── phase_fold.py
│       ├── savetocsv.py
│       └── test_pipeline.py
│
├── degradation_tests/
│   ├── cadence/
│   │   ├── convergence.py
│   │   ├── heatmaps_plot.py
│   │   ├── run.py
│   │   ├── run2.py
│   │   ├── single_csv_plot.py
│   │   └── synthetic_data.py
│   ├── core/
│   │   ├── fit_3var.py
│   │   ├── fit_6var.py
│   │   ├── keplerian_fit.py
│   │   ├── lombscargle.py
│   │   └── test_pipeline.py
│   ├── error_accuracy_analysis/
│   │   ├── plot.py
│   │   └── run_gaussian_noise.py
│   └── sparse_sampling_limits/
│       ├── convergence.py
│       ├── plot.py
│       └── run.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

A few notes on the structure:

- `data_reconstruction/` contains the original reconstruction pipeline.
- `degradation_tests/core/` contains the fitting routines used by the degradation experiments.
- `degradation_tests/` contains the individual experiment implementations and plotting scripts.
- `archive/` contains intermediate results, plots, logs and working notes from earlier stages of the project.

---

## 4. Current Status

So far, I've:

- Implemented and cross-validated a Generalized Lomb–Scargle periodogram.
- Reconstructed the orbital period from two independent 51 Pegasi b datasets.
- Implemented two approaches to Keplerian parameter fitting.
- Applied the fitting pipeline to both datasets.
- Conducted cadence degradation experiments involving seasonal gaps, lunar masking and periastron phase starvation.
- Investigated the effects of sparse temporal sampling.
- Started investigating the accuracy of Hessian-based uncertainty estimates against Monte Carlo results.

There is still work to do, particularly in uncertainty estimation and further analysis of the experimental results.

**This README describes the work completed so far, rather than a finished or finalized pipeline.**