import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
df = pd.read_csv('sparse_sampling.csv')

# Calculate Success Rates
df['Success_Int'] = (df['Status'] == 'Success').astype(int)
base_success_rate = df[df['test_type'] == 'Baseline_days'].groupby('baseline_days')['Success_Int'].mean() * 100

# Filter out failures and make a strict copy for the parameter plots
obs_success = df[(df['test_type'] == 'Observational') & (df['Status'] == 'Success')].copy()
base_success = df[(df['test_type'] == 'Baseline_days') & (df['Status'] == 'Success')].copy()

# 2. Set up the 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(16, 8))
plt.subplots_adjust(hspace=0.4, wspace=0.2)

# --- Graph 1: The Nyquist Fade (N-Stepdown) ---
sns.lineplot(data=obs_success, x='N', y='LS_power', ax=axs[0, 0], 
             estimator=np.median, errorbar='sd', color='blue', marker='o')
axs[0, 0].set_title('1. The Nyquist Fade (N vs Lomb-Scargle Power)')
axs[0, 0].set_xlabel('Number of Observations (N)')
axs[0, 0].set_ylabel('LS Peak Power')
axs[0, 0].invert_xaxis() # Read from left to right (degrading)

# --- Graph 2: The Sparsity Explosion (N-Stepdown) ---
sns.lineplot(data=obs_success, x='N', y='fit_K', ax=axs[0, 1], 
             estimator=np.median, errorbar='sd', color='purple', marker='^')
axs[0, 1].axhline(y=45.0, color='k', linestyle='--', label='True K (45.0)')
axs[0, 1].set_title('2. Sparsity Explosion (N vs Fitted Amplitude)')
axs[0, 1].set_xlabel('Number of Observations (N)')
axs[0, 1].set_ylabel('Fitted Semi-Amplitude K (m/s)')
axs[0, 1].legend()
axs[0, 1].invert_xaxis()

# --- Graph 3: The Window Limit (Baseline Pinch) ---
# Calculate Absolute Period Error safely
base_success['P_error'] = np.abs(base_success['fit_P'] - 21.1668)

# THE FIX: Mathematically force any absolute 0s to be slightly positive (1e-5)
# This completely bypasses the symlog bug and safely uses standard log
base_success['P_error'] = np.clip(base_success['P_error'], 1e-5, None)

sns.lineplot(data=base_success, x='baseline_days', y='P_error', ax=axs[1, 0], 
             estimator=np.median, errorbar='sd', color='darkorange', marker='s')

axs[1, 0].set_title('3. The Window Limit (Baseline vs Period Error)')
axs[1, 0].set_xlabel('Observation Baseline (Days)')
axs[1, 0].set_ylabel('Absolute Period Error (Days)')
axs[1, 0].set_yscale('log') # Back to standard log!
axs[1, 0].invert_xaxis()

# --- Graph 4: Orbit Unconstrained (Baseline Pinch) ---
sns.lineplot(data=base_success, x='baseline_days', y='fit_e', ax=axs[1, 1], 
             estimator=np.median, errorbar='sd', color='green', marker='D')
axs[1, 1].axhline(y=0.65, color='k', linestyle='--', label='True e (0.65)')
axs[1, 1].set_title('4. Loss of Constraints (Baseline vs Fitted Eccentricity)')
axs[1, 1].set_xlabel('Observation Baseline (Days)')
axs[1, 1].set_ylabel('Fitted Eccentricity (e)')
axs[1, 1].legend()
axs[1, 1].invert_xaxis()

plt.suptitle("Sparse Sampling Limits (Matrix & Sparsity Degradation)", fontsize=16, fontweight='bold')
plt.savefig('sparse_sampling.png', dpi=300, bbox_inches='tight')
plt.show()