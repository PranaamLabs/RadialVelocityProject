import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

all_csvs = glob.glob("sparsity_CSVs/*.csv")
df = pd.concat([pd.read_csv(i) for i in all_csvs])

# Calculate Errors
df['rel_P_error'] = np.abs(df['fit_P'] - df['true_P'])/df['true_P']
df['rel_K_error'] = np.abs(df['fit_K'] - df['true_K'])/df['true_K']
df['rel_e_error'] = np.abs(df['fit_e'] - df['true_e'])/df['true_e']
 
# Build 2D Matrices
matrix_P = df.pivot_table(index='true_e', columns='N', values='rel_P_error', aggfunc='median')
matrix_K = df.pivot_table(index='true_e', columns='N', values='rel_K_error', aggfunc='mean')
matrix_LS = df.pivot_table(index='true_e', columns='N', values='LS_power', aggfunc='median')
matrix_e = df.pivot_table(index='true_e', columns='N', values='rel_e_error', aggfunc='mean')

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
axs[1].set_title('Mean Relative K Error')
axs[1].set_xlabel('Number of Observations (N)')
fig.colorbar(im2, ax=axs[1], label='K Error (m/s)')

# Plot 3: Period Error Map (Log Scale for alias jumps)
im3 = axs[2].pcolormesh(X, Y, matrix_e.values, shading='auto', 
                        norm=colors.LogNorm(vmin=np.nanmin(matrix_P.values[matrix_e.values>0])), 
                        cmap='magma')
axs[2].invert_xaxis()
axs[2].set_title('Mean Relative Eccentricity Error')
axs[2].set_xlabel('Number of Observations (N)')
fig.colorbar(im3, ax=axs[2], label='Eccentricity Error (Days)')

plt.suptitle("Sparsity Degradation limits: $N$ vs $e$", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()