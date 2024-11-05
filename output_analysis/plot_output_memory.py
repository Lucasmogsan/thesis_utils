# Python script to load data from "loopy.csv" and plot GPU memory usage over time

import pandas as pd
import matplotlib.pyplot as plt

# Load data from "loopy.csv"
data = pd.read_csv("output_analysis/loopy_room0_gpu_mem.csv")

print(data.head())

# Extract relative time and memory percentage columns
data['Time'] = data['Relative Time (Process)'].astype(float)
data['Memory'] = data['mapper_20241101_193125 - system/gpu.0.memoryAllocated'].astype(float)

# Convert memory percentage values to GB based on total GPU memory (24.567 GB)
total_gpu_memory_gb = 24.567
data['Memory'] = data['Memory'] * total_gpu_memory_gb / 100

# Convert to numpy array for easier indexing
data_time = data['Time'].to_numpy()
data_memory = data['Memory'].to_numpy()


# Plotting the GPU memory allocation over time
plt.figure(figsize=(10, 6))
plt.plot(data_time, data_memory, 'b-', label="GPU Memory Allocated (GB)")

# Adding labels and title
plt.xlabel('Relative Time, Process (s)', fontsize=12)
plt.ylabel('Memory Allocated (GB)', fontsize=12)
plt.title('GPU Memory Allocation Over Time', fontsize=14)
plt.grid(True)
plt.legend()

plt.ylim(0, 24)

plt.tight_layout()
plt.show()

