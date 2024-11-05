import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

dataset = "TUM"
scene = "freiburg1_room"

# Load the data
pred_pose_path = f"/home/aut/thesis_utils/output_analysis/comparison/{dataset}/{scene}/RTABMAP/rtabmap_poses_3.txt"
gt_pose_path = f"/home/aut/thesis_utils/output_analysis/comparison/{dataset}/{scene}/RTABMAP/poses_gt_3.txt"
pred = np.loadtxt(pred_pose_path, comments='#')
gt = np.loadtxt(gt_pose_path, comments='#')

# Extract timestamps and positions
pred_timestamps, pred_positions = pred[:, 0], pred[:, 1:4]
gt_timestamps, gt_positions = gt[:, 0], gt[:, 1:4]

# Interpolate ground truth positions to match predicted timestamps
gt_positions_interp = np.array([
    np.interp(pred_timestamps, gt_timestamps, gt_positions[:, i]) for i in range(3)
]).T


# Calculate Euclidean error for ATE in meters
errors = np.linalg.norm(pred_positions - gt_positions_interp, axis=1) * 100
ate_rmse = np.sqrt(np.mean(errors**2))

# Plotting the positional error over time
plt.figure(figsize=(10, 6))
plt.plot(pred_timestamps, errors * 100, label='Positional Error (cm)')
plt.xlabel("Timestamp")
plt.ylabel("Error (cm)")
plt.title("ATE Positional Error between Predicted and Ground Truth")
plt.legend()
plt.show()

print(f"ATE RMSE: {ate_rmse:.3f} cm")
