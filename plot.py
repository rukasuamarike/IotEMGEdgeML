import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

# Replace 'your_file.csv' with the actual path to your CSV file
# python plot.py fingertap
filename = f"{sys.argv[1]}.csv"
# Read the CSV data using pandas
try:
    df = pd.read_csv(filename, header=0)  # Assuming headers are present (row 0)
except FileNotFoundError:
    print(f"Error: File '{filename}' not found!")
    exit()

# Extract data (assuming columns represent EMG signals)
time = np.linspace(0, df.shape[0] - 1, df.shape[0])
emg1 = df["emg1"]  # EMG signal 1 data
emg2 = df["emg2"]  # EMG signal 2 data
emg3 = df["emg3"]  # Add more as needed (assuming 4th column)

# Create the plot
plt.figure(figsize=(10, 6))  # Adjust figure size as desired

# Plot multiple EMG signals on the same plot
plt.plot(time, emg1, label="EMG 1", marker="o")
plt.plot(time, emg2, label="EMG 2", marker="s")
plt.plot(time, emg3, label="EMG 3", marker="^")  # Add more lines for additional signals

# Customize plot elements
plt.xlabel("Time (seconds)")  # Assuming time is in seconds (adjust label if different)
plt.ylabel(
    "EMG Signal (mV)"
)  # Assuming EMG signals are in millivolts (adjust label if different)
plt.title(f"EMG Signals over Time from '{filename}'")
plt.legend()
plt.grid(True)
# Display the plot
plt.show()
