import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lombscargle import LombScargle

df1= pd.read_csv("../data/csv/butler.csv")
df2= pd.read_csv("../data/csv/howardfulton.csv")

y1 = df1["Radial Velocity (m/s)"].to_list()
y2 = df2["Radial Velocity (m/s)"].to_list()

dy1 = df1["Radial Velocity Uncertainty (m/s)"].to_list()
dy2 = df2["Radial Velocity Uncertainty (m/s)"].to_list()

df1["Days"]= df1["HJD (days)"] - df1["HJD (days)"].min()
df2["Days"]= df2["HJD (days)"] - df2["HJD (days)"].min()
t1 = df1["Days"].to_list()
t2 = df2["Days"].to_list()

def omegas(t:list, oversampling=5) -> np.ndarray:
    T_span = t.max() - t.min()
    dt = np.diff(np.sort(t))
    f_min = 1.0 / T_span
    f_max = 1.0 / (2.0 * np.median(dt))
    N_freq = int(oversampling * (f_max - f_min) * T_span)
    freqs = np.linspace(f_min, f_max, N_freq)
    return 2 * np.pi * freqs

omegas1 = omegas(df1["Days"])
omegas2 = omegas(df2["Days"])

gls1 = LombScargle(t1, y1, dy1)
gls2 = LombScargle(t2, y2, dy2)

P1 = gls1.periodogram(omegas1)
P2 = gls2.periodogram(omegas2)

best_omega1 = omegas1[np.argmax(P1)]
best_omega2 = omegas2[np.argmax(P2)]
period1 = 2*np.pi/best_omega1
period2 = 2*np.pi/best_omega2

def update_data(filename, omega, period):
    with open(filename, 'a') as data:
        data.write("\nLombScargle:\n")
        data.write(f"Best Omega: {omega:.6f} rad/d\n")
        data.write(f"Period: {period:.4f} days\n")

update_data('../data/butler_data.txt', best_omega1, period1)
update_data('../data/howardfulton_data.txt', best_omega2, period2)

print(f"Butler Dataset - Best Omega: {best_omega1:.6f} rad/d (Period: {period1:.4f} days)")
print(f"Howard & Fulton Dataset - Best Omega: {best_omega2:.6f} rad/d (Period: {period2:.4f} days)")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Top Plot: Butler Dataset
ax1.plot(omegas1, P1, color='C0', linewidth=1.5)
ax1.axvline(best_omega1, color='red', linestyle='--', alpha=0.7, label=f'Peak: {best_omega1:.4f} rad/d')
ax1.set_title('Generalized Lomb-Scargle: Butler Dataset')
ax1.set_xlabel(r'Angular Frequency ($\omega$) [rad/d]')
ax1.set_ylabel('Power')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Bottom Plot: Howard & Fulton Dataset
ax2.plot(omegas2, P2, color='C1', linewidth=1.5)
ax2.axvline(best_omega2, color='red', linestyle='--', alpha=0.7, label=f'Peak: {best_omega2:.4f} rad/d')
ax2.set_title('Generalized Lomb-Scargle: Howard & Fulton Dataset')
ax2.set_xlabel(r'Angular Frequency ($\omega$) [rad/d]')
ax2.set_ylabel('Power')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()