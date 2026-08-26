import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

all_csvs = glob.glob("cadence_CSVs/*.csv")
df = pd.DataFrame()
for i in all_csvs:
    df = pd.concat([df,pd.read_csv(i)])

grouped = df.groupby(['test_type', 'true_e', 'independent_var'])[['fit_P', 'LS_power','fit_k', 'fit_e']].agg(['median', 'std'])

# 1. Calculate your absolute errors across the whole dataframe
df['abs_P_error'] = np.abs(df['fit_P'] - df['true_P'])
df['abs_K_error'] = np.abs(df['fit_K'] - df['true_K'])
df['abs_e_error'] = np.abs(df['fit_e'] - df['true_e'])

# 2. Split the dataframe by test type
df_seasonal = df[df['test_type'] == 'Seasonal']
df_lunar = df[df['test_type'] == 'Lunar']
df_phase = df[df['test_type'] == 'Phase']

# 3. Build the 2D Heatmap Matrices using pivot_table
# Top Left: Seasonal Gap vs Period Error (Median)
matrix_seasonal_P = df_seasonal.pivot_table(
    index='true_e', 
    columns='independent_var', 
    values='abs_P_error', 
    aggfunc='median'
)

# Top Right: Lunar Blackout vs LS Power (Median)
matrix_lunar_LS = df_lunar.pivot_table(
    index='true_e', 
    columns='independent_var', 
    values='LS_power', 
    aggfunc='median'
)

# Bottom Left: Phase Starvation vs K Error (Mean Absolute Error)
matrix_phase_K = df_phase.pivot_table(
    index='true_e', 
    columns='independent_var', 
    values='abs_K_error', 
    aggfunc='mean'
)

# Bottom Right: Phase Starvation vs e Error (Mean Absolute Error)
matrix_phase_e = df_phase.pivot_table(
    index='true_e', 
    columns='independent_var', 
    values='abs_e_error', 
    aggfunc='mean'
)

# Create the 2x2 subplot grid
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Cadence Degradation Studies (Heatmaps)', fontsize=18, fontweight='bold', y=0.98)

# ---------------------------------------------------------
# 1. Top Left: Seasonal Gap vs Period Error
# ---------------------------------------------------------
X1 = matrix_seasonal_P.columns
Y1 = matrix_seasonal_P.index
Z1 = matrix_seasonal_P.values

# Using LogNorm because period errors range from tiny fractions to massive alias jumps
im1 = axs[0, 0].pcolormesh(X1, Y1, Z1, shading='auto', 
                           norm=colors.LogNorm(vmin=np.nanmin(Z1[Z1>0]), vmax=np.nanmax(Z1)), 
                           cmap='magma')
axs[0, 0].set_title('1. Seasonal Gap vs Median Absolute Period Error')
axs[0, 0].set_xlabel('Gap Width (Days)')
axs[0, 0].set_ylabel('True Eccentricity ($e$)')
fig.colorbar(im1, ax=axs[0, 0], label='Period Error (Days)')

# ---------------------------------------------------------
# 2. Top Right: Lunar Blackout vs LS Power
# ---------------------------------------------------------
X2 = matrix_lunar_LS.columns
Y2 = matrix_lunar_LS.index
Z2 = matrix_lunar_LS.values

# Sequential colormap to show fading signal power
im2 = axs[0, 1].pcolormesh(X2, Y2, Z2, shading='auto', cmap='viridis')
axs[0, 1].set_title('2. Lunar Blackout vs Median LS Peak Power')
axs[0, 1].set_xlabel('Lunar Blackout (Days per Month)')
axs[0, 1].set_ylabel('True Eccentricity ($e$)')
fig.colorbar(im2, ax=axs[0, 1], label='Lomb-Scargle Power')

# ---------------------------------------------------------
# 3. Bottom Left: Phase Starvation vs K Error
# ---------------------------------------------------------
X3 = matrix_phase_K.columns
Y3 = matrix_phase_K.index
Z3 = matrix_phase_K.values

# Divergent/hot colormap to show where K-variance explodes
im3 = axs[1, 0].pcolormesh(X3, Y3, Z3, shading='auto', cmap='coolwarm')
axs[1, 0].set_title('3. Phase Starvation vs Mean Absolute K Error')
axs[1, 0].set_xlabel('Fraction of Phase Deleted near Periastron')
axs[1, 0].set_ylabel('True Eccentricity ($e$)')
fig.colorbar(im3, ax=axs[1, 0], label='K Error (m/s)')

# ---------------------------------------------------------
# 4. Bottom Right: Phase Starvation vs e Error
# ---------------------------------------------------------
X4 = matrix_phase_e.columns
Y4 = matrix_phase_e.index
Z4 = matrix_phase_e.values

# Divergent colormap to show where e-constraints fail
im4 = axs[1, 1].pcolormesh(X4, Y4, Z4, shading='auto', cmap='coolwarm')
axs[1, 1].set_title('4. Phase Starvation vs Mean Absolute $e$ Error')
axs[1, 1].set_xlabel('Fraction of Phase Deleted near Periastron')
axs[1, 1].set_ylabel('True Eccentricity ($e$)')
fig.colorbar(im4, ax=axs[1, 1], label='Absolute $e$ Error')

# ---------------------------------------------------------
# Final Formatting
# ---------------------------------------------------------
plt.tight_layout()
fig.subplots_adjust(top=0.92) # Give the suptitle some breathing room

# Save the figure in high resolution
plt.savefig('cadence_degradation_heatmaps.png', dpi=300, bbox_inches='tight')
plt.show()