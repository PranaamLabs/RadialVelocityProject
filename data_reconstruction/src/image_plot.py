import matplotlib.pyplot as plt
import pandas as pd

#Import data
df1= pd.read_csv("../data/csv/butler.csv")
df2= pd.read_csv("../data/csv/howardfulton.csv")

df1["Days"]= df1["HJD (days)"] - df1["HJD (days)"].min()
df2["Days"]= df2["HJD (days)"] - df2["HJD (days)"].min()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


#Scatter Plot with uncertainities
ax1.errorbar(x=df1["Days"],
             y=df1["Radial Velocity (m/s)"], 
             yerr=df1["Radial Velocity Uncertainty (m/s)"],
             fmt='o', 
             ecolor='red',
             capsize=5,
             label="Days vs Radial Velocity",
             markersize=4,
             alpha=0.8,
             elinewidth=1
             )

ax1.set_title("First RV Time Series Plot (Butler)")
ax1.set_xlabel("Days")
ax1.set_ylabel("Radial Velocity (m/s)")
ax1.grid(True, linestyle=':')

ax2.errorbar(x=df2["Days"],
             y=df2["Radial Velocity (m/s)"], 
             yerr=df2["Radial Velocity Uncertainty (m/s)"],
             fmt='o', 
             ecolor='red',
             capsize=5,
             label="Days vs Radial Velocity",
             markersize=4,
             alpha=0.8,
             elinewidth=1
             )

ax2.set_title("First RV Time Series Plot (Howard & Fulton)")
ax2.set_xlabel("Days")
ax2.set_ylabel("Radial Velocity (m/s)")
ax2.grid(True, linestyle=':')


plt.tight_layout()
plt.show()