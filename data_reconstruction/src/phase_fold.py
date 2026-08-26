import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_phase_folded_rv(file_path, period, title_suffix=""):
    df = pd.read_csv(file_path)
    
    y = df["Radial Velocity (m/s)"]
    dy = df["Radial Velocity Uncertainty (m/s)"]
    
    days = df["HJD (days)"] - df["HJD (days)"].min()
    phases = (days / period) % 1.0

    plt.figure(figsize=(8, 5))
    plt.errorbar(phases, y, yerr=dy, fmt='o', color='black', 
                 ecolor='gray', capsize=2, label='Data')
    plt.errorbar(phases + 1.0, y, yerr=dy, fmt='o', color='gray', alpha=0.5)
    
    plt.xlabel('Orbital Phase ($\phi$)')
    plt.ylabel('Radial Velocity (m/s)')
    plt.title(f'Phase Folded RV Data - {title_suffix} (P = {period:.4f} days)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, 2)
    plt.show()

# Execution
plot_phase_folded_rv("../data/csv/butler.csv", 4.2302, title_suffix="Butler")
plot_phase_folded_rv("../data/csv/howardfulton.csv", 4.2308, title_suffix="Howard-Fulton")