import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .run import TRUE_PARAMS

# 1. Load Data
df = pd.read_csv('cadence_results_e035.csv')

# Calculate Absolute Period Error
df['P_error'] = np.abs(df['fit_P'] - df['true_P'])

# Set up the 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(hspace=0.3, wspace=0.2)

# --- Graph 1: The Aliasing Cliff (Seasonal) ---
seasonal = df[df['test_type'] == 'Seasonal']
sns.lineplot(data=seasonal, x='independent_var', y='P_error', ax=axs[0, 0], 
             estimator=np.median, errorbar=('pi',68), color='red', marker='o')
axs[0, 0].set_title('1. (Seasonal Gap vs Period Error)')
axs[0, 0].set_xlabel('Gap Width (Days)')
axs[0, 0].set_ylabel('Absolute Period Error (Days)')
axs[0, 0].set_yscale('log') # Log scale because the alias error jump is usually massive

# --- Graph 2: The Power Bleed (Lunar) ---
lunar = df[df['test_type'] == 'Lunar']
sns.lineplot(data=lunar, x='independent_var', y='LS_power', ax=axs[0, 1], 
             estimator=np.median, errorbar=('pi',68), color='blue', marker='s')
axs[0, 1].set_title('2. (Lunar Blackout vs LS Power)')
axs[0, 1].set_xlabel('Lunar Blackout (Days per Month)')
axs[0, 1].set_ylabel('Lomb-Scargle Peak Power')

# --- Graph 3: Amplitude Collapse (Phase) ---
phase = df[df['test_type'] == 'Phase']
sns.lineplot(data=phase, x='independent_var', y='fit_K', ax=axs[1, 0], 
             estimator=np.median, errorbar=('pi',68), color='purple', marker='^')
axs[1, 0].axhline(y=TRUE_PARAMS["K"], color='k', linestyle='--', label=f'True K ({TRUE_PARAMS['K']})')
axs[1, 0].set_title('3. (Phase Starvation vs K)')
axs[1, 0].set_xlabel('Fraction of Phase Deleted near Periastron')
axs[1, 0].set_ylabel('Fitted Semi-Amplitude K (m/s)')
axs[1, 0].legend()

# --- Graph 4: Orbit Circularization (Phase) ---
sns.lineplot(data=phase, x='independent_var', y='fit_e', ax=axs[1, 1], 
             estimator=np.median, errorbar=('pi',68), color='green', marker='D')
axs[1, 1].axhline(y=TRUE_PARAMS["e"], color='k', linestyle='--', label=f'True e ({TRUE_PARAMS['e']})')
axs[1, 1].set_title('4. (Phase Starvation vs e)')
axs[1, 1].set_xlabel('Fraction of Phase Deleted near Periastron')
axs[1, 1].set_ylabel('Fitted Eccentricity (e)')
axs[1, 1].legend()

plt.suptitle("Cadence Degradation Studies", fontsize=16, fontweight='bold')
plt.savefig('cadence_degradation_e035.png', dpi=300)
plt.show()