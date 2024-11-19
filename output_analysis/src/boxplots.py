import matplotlib.pyplot as plt

# Data
data = {
    "ORB-SLAM3": [3.4830, 3.2030, 2.2530],
    "RTAB-Map": [8.6900, 6.7600, 3.89],
    "Loopy-SLAM": [5.5137, 4.7034, 5.7876],
    "Photo-SLAM": [2.1310, 2.0640, 2.3730],
    "GS-ICP-SLAM": [6.6178, 2.6388, 7.0551],
}

# Boxplot
plt.figure(figsize=(10, 6))
plt.boxplot(data.values(), labels=data.keys(), patch_artist=True, notch=False, showmeans=True)
plt.ylabel("ATE RMSE", fontsize=18)
plt.grid(axis='y', linestyle='--', alpha=0.7)
# make the labels larger
plt.xticks(fontsize=18)

# Show plot
plt.show()