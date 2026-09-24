import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

all_csvs = glob.glob("sparsity_CSVs/*.csv")
df = pd.concat([pd.read_csv(i) for i in all_csvs])

# Calculate Errors
df['abs_P_error'] = np.abs(df['fit_P'] - df['true_P'])
df['abs_K_error'] = np.abs(df['fit_K'] - df['true_K'])

# Build 2D Matrices
matrix_P = df.pivot_table(index='true_e', columns='N', values='abs_P_error', aggfunc='median')
matrix_K = df.pivot_table(index='true_e', columns='N', values='abs_K_error', aggfunc='mean')
matrix_LS = df.pivot_table(index='true_e', columns='N', values='LS_power', aggfunc='median')

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: LS Power Map
X, Y = matrix_LS.columns, matrix_LS.index
im1 = axs[0].pcolormesh(X, Y, matrix_LS.values, shading='auto', cmap='viridis')
axs[0].invert_xaxis()  # Read left-to-right as degradation
axs[0].set_title('Median LS Peak Power')
axs[0].set_xlabel('Number of Observations (N)')
axs[0].set_ylabel('True Eccentricity (e)')
fig.colorbar(im1, ax=axs[0], label='Lomb-Scargle Power')

# Plot 2: K Error Map
im2 = axs[1].pcolormesh(X, Y, matrix_K.values, shading='auto', cmap='coolwarm')
axs[1].invert_xaxis()
axs[1].set_title('Mean Absolute K Error')
axs[1].set_xlabel('Number of Observations (N)')
fig.colorbar(im2, ax=axs[1], label='K Error (m/s)')

# Plot 3: Period Error Map (Log Scale for alias jumps)
im3 = axs[2].pcolormesh(X, Y, matrix_P.values, shading='auto', 
                        norm=colors.LogNorm(vmin=np.nanmin(matrix_P.values[matrix_P.values>0])), 
                        cmap='magma')
axs[2].invert_xaxis()
axs[2].set_title('Median Absolute Period Error')
axs[2].set_xlabel('Number of Observations (N)')
fig.colorbar(im3, ax=axs[2], label='Period Error (Days)')

plt.suptitle("Sparsity Degradation limits: $N$ vs $e$", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()