# Radial Velocity Signal Analysis Project

**The repo might seem to be a bit unorganised(I did my best), therefore I'm writing this MD file so that you get an understanding of what this repo has.**

**This file has been updated upto 9-9-26**

---

The Project has two parts : Data Reconstruction and Degradation Tests

---

## 1. Data Reconstruction
- The codes of this section are in data_reconstruction/src
- Here, I've basically downloaded two RV datasets of 51 Pegasi b - *(Howard & Fulton, 2016)* and *(Butler et al, 2006)* and saved as csv in data_reconstruction/data/csv. I'll refer to them as *HF* and *B*.
- src/lombscargle.py has the code for Generalised LombScargle Periodogram, which does exactly the same job(verified - archive/experiments/lombscargle/* ) as `astropy.timeseries.lombscargle()`. The implementation is similar to *Zechmeister & Kürster (2009)*.
  - When applied both datasets- HF gave Period of 4.2308 days and B gave that of 4.2302 days which is close enough to the value online ($\approx$ 4.2308 days). (refer data/butler_data.txt and data/howardandfulton_data.txt)
  - The additional peaks in the Butler plot at higher $\omega$ are likely harmonics/aliases rather than the fundamental period.
- phase_fold.py gave a first impression of the sinusoidal nature of the data to me. (archive/experiments/reconstruction/phase_folding/*).

- To fit this graph according to the equation $v = γ + K(cos(ν + ω) + e cos ω)$, I need 6 variables P, e, K, $T_p$, $\gamma$, and $\omega$. Since this is a non linear equation i can directly do non linear optimization (a global optimization `scipy.optimize.differential evolution` followed by local refinement `scipy.optimize.minimize`) in fit_6var.py which uses P from LombScargle to find bounds for the two scipy functions. 
  - The error propagation in fit_6var.py is currently not fully satisfactory, and I plan to revisit this. The implementation primarily relies on SciPy's optimization routines, whereas the approach in fit_3var.py allowed me to work more explicitly with the underlying parameter structure and uncertainty estimation.
  - I came across *(Wright, 2009)* which suggested decoupling of the radial velocity equation so we get 3 variables ($\gamma$, K and $\omega$) through linear regression and the other 3 (P, e and $T_p$) from non linear optimization just like above - but i calculated the Hessian Matrix using central finite distances to find the errors for the latter 3 properly. (refer fit_3var.py). Although the paper had the math for multiple planets, I simplified it for single planet model(refer notes/). keplerian_fit.py takes t, v, dv and finds mean anomaly, eccentric anomaly, true anomaly, $\gamma$, $K \cos \omega$ and $-K \sin \omega$ for given P, e and $T_p$

- I applied the linear - non linear fitting to HF and B datasets and plotted the graphs. Plots are in archive/reconstruction/keplerian_fit/ and parameters found in data/fit_data.txt

---

## 2. Degradation Tests
TODO