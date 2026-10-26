import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

all_csvs = glob.glob("error_analysis_CSVs/*.csv")
df = pd.concat([pd.read_csv(i) for i in all_csvs])

TRUE_PARAMS = {
    'P': 21.1668, 'T_p': 12.0,
    'K': 45.0, 'omega': 1.2, 'gamma': 5.0,
    'baseline_days': 200, 'N': 80
}

# 1 & 2: Group and Aggregate
std_df = df.groupby(['e','noise_base'])[['fit_P', 'fit_T_p', 'fit_K', 'fit_e', 'fit_g', 'fit_w']].std()
mean_df = df.groupby(['e','noise_base'])[['dP', 'dT_p', 'dK', 'de', 'dg', 'dw']].mean()

# 3: Calculate Discrepancy Metric
def metric(x):
    return abs(std_df[f'fit_{x}'] - mean_df[f'd{x}']) / std_df[f'fit_{x}']

metrics = {
    'Period (P)': metric('P'),
    'Amplitude (K)': metric('K'),
    'Eccentricity (e)': metric('e'),
    'Time of Periapsis (T_p)': metric('T_p'),
    'Argument of Periastron (\u03C9)': metric('w'),
    'Systemic Velocity (\u03B3)': metric('g')
}

# 4 & 5: Unstack to Matrices and Plot
fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
axes = axes.flatten()

for ax, (title, metric_series) in zip(axes, metrics.items()):
    # Unstack the series into a 2D matrix for the heatmap
    matrix = metric_series.unstack(level='noise_base')
    
    # Extract axes arrays for pcolormesh
    e_vals = matrix.index.values
    noise_vals = matrix.columns.values
    
    # Plot using LogNorm to prevent extreme Hessian failures from washing out the color scale
    pcm = ax.pcolormesh(
        noise_vals, e_vals, matrix.values, 
        cmap='viridis', 
        norm=colors.LogNorm(vmin=0.01, vmax=10.0), 
        shading='nearest'
    )
    
    ax.set_title(f'Hessian vs MC Error:\n{title}')
    ax.set_xlabel('Noise Base (m/s)')
    ax.set_ylabel('Eccentricity (e)')
    
    # Add a colorbar to each subplot
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label('Fractional Error')

fig.suptitle('Hessian Error Approximation Breakdown', fontsize=16, fontweight='bold')
plt.savefig('hessian_error_heatmaps.png', dpi=300)
plt.show()