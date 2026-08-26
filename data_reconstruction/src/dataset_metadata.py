import pandas as pd
from pathlib import Path

class Inspection:
    def __init__(self, csvfile):
        self.csvfile = csvfile
        self.name= str(Path(self.csvfile).stem)
        self.parent = str(Path(self.csvfile).parent.parent)
    
    def inspect(self):
        #1. Read CSV
        df= pd.read_csv(self.csvfile)

        #2. Find observation count
        count= df["HJD (days)"].count()

        #3. Find baseline duration
        total_time= df["HJD (days)"].max() - df["HJD (days)"].min()

        #4. Find Mean Uncertainty and standard deviation of velocities
        mean_u= df["Radial Velocity Uncertainty (m/s)"].mean()
        std_dev= df["Radial Velocity (m/s)"].std()

        #5. Find min and max of Radial velocity in dataset
        min_rv= df["Radial Velocity (m/s)"].min()
        max_rv= df["Radial Velocity (m/s)"].max()

        with open(f"{self.parent}/{self.name}_data.txt", "w") as f:
            f.writelines([f"{self.name} Data's Statistical Inspection\n", 
                         f"Count= {count} days\n",
                         f"Baseline Duration = {total_time} days\n",
                         f"Mean Uncertainty = {mean_u} m/s\n",
                         f"SD of Velocities = {std_dev} m/s\n",
                         f"Minimum RV = {min_rv} m/s\n",
                         f"Maximum RV = {max_rv} m/s\n"
                         ])

if __name__ == "__main__":
    butler = Inspection("../data/csv/butler.csv")
    butler.inspect()

    howard_fulton = Inspection("../data/csv/howardfulton.csv")
    howard_fulton.inspect()
