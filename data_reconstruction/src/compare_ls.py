import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.timeseries import LombScargle as AstropyLS
from lombscargle import LombScargle as CustomLS

# 1. Load Data
df1 = pd.read_csv("../data/csv/butler.csv")
df2 = pd.read_csv("../data/csv/howardfulton.csv")

t1 = (df1["HJD (days)"] - df1["HJD (days)"].min()).to_list()
y1 = df1["Radial Velocity (m/s)"].to_list()
dy1 = df1["Radial Velocity Uncertainty (m/s)"].to_list()

t2 = (df2["HJD (days)"] - df2["HJD (days)"].min()).to_list()
y2 = df2["Radial Velocity (m/s)"].to_list()
dy2 = df2["Radial Velocity Uncertainty (m/s)"].to_list()

# 2. Define Frequency Grid
def omegas(t: list, oversampling=5) -> tuple[np.ndarray, np.ndarray]:
    t_arr = np.array(t)
    T_span = t_arr.max() - t_arr.min()
    dt = np.diff(np.sort(t_arr))
    f_min = 1.0 / T_span
    f_max = 1.0 / (2.0 * np.median(dt))
    N_freq = int(oversampling * (f_max - f_min) * T_span)
    freqs = np.linspace(f_min, f_max, N_freq)
    return 2 * np.pi * freqs, freqs

omegas1, freqs1 = omegas(t1)
omegas2, freqs2 = omegas(t2)

# 3. Calculate Periodograms
# Butler Dataset
custom_ls1 = CustomLS(t1, y1, dy1)
P1_custom = custom_ls1.periodogram(omegas1)

astro_ls1 = AstropyLS(t1, y1, dy1)
# Scale Astropy power to match custom ZK09 Eq 22 normalization: (N-1)/2
P1_astro = astro_ls1.power(freqs1) * ((len(t1) - 1) / 2) 

# Howard & Fulton Dataset
custom_ls2 = CustomLS(t2, y2, dy2)
P2_custom = custom_ls2.periodogram(omegas2)

astro_ls2 = AstropyLS(t2, y2, dy2)
P2_astro = astro_ls2.power(freqs2) * ((len(t2) - 1) / 2)

# 4. Find Best Fit Parameters
def extract_best(omegas_arr, power_arr):
    idx = np.argmax(power_arr)
    best_omega = omegas_arr[idx]
    period = 2 * np.pi / best_omega
    return best_omega, period

b1_omega_c, b1_period_c = extract_best(omegas1, P1_custom)
b1_omega_a, b1_period_a = extract_best(omegas1, P1_astro)

b2_omega_c, b2_period_c = extract_best(omegas2, P2_custom)
b2_omega_a, b2_period_a = extract_best(omegas2, P2_astro)

# 5. Print Results
print("-" * 50)
print("BUTLER DATASET")
print(f"Custom  - Best Omega: {b1_omega_c:.6f} rad/d | Period: {b1_period_c:.4f} days")
print(f"Astropy - Best Omega: {b1_omega_a:.6f} rad/d | Period: {b1_period_a:.4f} days")
print("-" * 50)
print("HOWARD & FULTON DATASET")
print(f"Custom  - Best Omega: {b2_omega_c:.6f} rad/d | Period: {b2_period_c:.4f} days")
print(f"Astropy - Best Omega: {b2_omega_a:.6f} rad/d | Period: {b2_period_a:.4f} days")
print("-" * 50)

# 6. Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Top Plot: Butler
ax1.plot(omegas1, P1_custom, color='C0', linewidth=3, label='Custom LS', alpha=0.6)
ax1.plot(omegas1, P1_astro, color='red', linestyle='--', linewidth=1.5, label='Astropy LS')
ax1.axvline(b1_omega_c, color='black', linestyle=':', alpha=0.7, label=f'Peak: {b1_omega_c:.4f} rad/d')
ax1.set_title('Butler Dataset Comparison')
ax1.set_xlabel(r'Angular Frequency ($\omega$) [rad/d]')
ax1.set_ylabel('Power (ZK09 Norm)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Bottom Plot: Howard & Fulton
ax2.plot(omegas2, P2_custom, color='C1', linewidth=3, label='Custom LS', alpha=0.6)
ax2.plot(omegas2, P2_astro, color='red', linestyle='--', linewidth=1.5, label='Astropy LS')
ax2.axvline(b2_omega_c, color='black', linestyle=':', alpha=0.7, label=f'Peak: {b2_omega_c:.4f} rad/d')
ax2.set_title('Howard & Fulton Dataset Comparison')
ax2.set_xlabel(r'Angular Frequency ($\omega$) [rad/d]')
ax2.set_ylabel('Power (ZK09 Norm)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()