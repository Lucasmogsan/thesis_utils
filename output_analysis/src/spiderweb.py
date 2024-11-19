import matplotlib.pyplot as plt
import numpy as np

# Data - PSNR, SSIM, LPIPS, ATE, FPS, Memory
data = {
    # "ORB-SLAM3": [0, 0, 1, 4.9153, 30, 0],
    # "RTAB-Map": [0, 0, 1, 10.4523, 30, 0],
    "Loopy-SLAM": [31.8905, 0.9223, 0.1844, 10.0288, 0.3705, 15.00],
    "Photo-SLAM": [32.357, 0.9118, 0.1382, 4.4235, 25.6, 2.95],
    "GS-ICP-SLAM": [35.6434, 0.9623, 0.0648, 7.9092, 29.94, 5.09],
}

PSNR_max = 40
PSNR_min = 20
SSIM_max = 1
SSIM_min = 0
LPIPS_max = 1
LPIPS_min = 0
ATE_max = 15
ATE_min = 0
FPS_max = 30
FPS_min = 0
Memory_max = 24
Memory_min = 0

# Scale all the values to be between 0 and 1 relative to the values above
for label in data.keys():
    data[label][0] = (data[label][0] - PSNR_min) / (PSNR_max - PSNR_min)
    data[label][1] = (data[label][1] - SSIM_min) / (SSIM_max - SSIM_min)
    data[label][2] = (data[label][2] - LPIPS_min) / (LPIPS_max - LPIPS_min)
    data[label][3] = (data[label][3] - ATE_min) / (ATE_max - ATE_min)
    data[label][4] = (data[label][4] - FPS_min) / (FPS_max - FPS_min)
    data[label][5] = (data[label][5] - Memory_min) / (Memory_max - Memory_min)

# Take the inverse of LPIPS, ATE, Memory so lower values are better
for label in data.keys():
    data[label][2] = 1 - data[label][2]
    data[label][3] = 1 - data[label][3]
    data[label][5] = 1 - data[label][5]

# Extract the metrics and labels from the data dictionary
labels = list(data.keys())
metrics = list(data.values())

# Create a spiderweb plot
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Set the number of metrics and angles
num_metrics = len(metrics[0])
angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
angles += angles[:1]

# Plot each data point
for label, metric in zip(labels, metrics):
    values = metric + metric[:1]
    ax.plot(angles, values, label=label)

# Set the labels for each metric
ax.set_xticks(angles[:-1])
ax.set_xticklabels(["PSNR", "SSIM", "LPIPS", "ATE", "FPS", "Memory"], fontsize=12)

# Add a legend with font size 12
ax.legend(loc="upper right", fontsize=12)

# Show the plot
plt.show()
