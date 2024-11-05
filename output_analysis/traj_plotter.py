import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

dataset = "TUM"
scene = "freiburg3_long_office_household"

# Load ground truth poses
gt_pose = f"/home/aut/thesis_utils/output_analysis/comparison/{dataset}/{scene}/gt/groundtruth_matched.txt"
gt_txt = np.loadtxt(gt_pose)
gt_t = gt_txt[:, 1:4]  # position
gt_R = gt_txt[:, 4:] # orientation
origin_t = gt_t[0]
origin_R = gt_R[0]
origin_R_mat = R.from_quat(origin_R).as_matrix()

# Load predicted poses
#pred_pose_gsicp = f"/home/aut/thesis_utils/output_analysis/comparison/{dataset}/{scene}/GS_ICP_SLAM/20241025_133454/poses.txt"
#gt_pose_gsicp = f"/home/aut/thesis_utils/output_analysis/comparison/{dataset}/{scene}/GS_ICP_SLAM/20241025_133454/gt_poses.txt"
pred_pose_gsicp = f"/home/aut/thesis_utils/output_analysis/comparison/{dataset}/{scene}/GS_ICP_SLAM/med_20241104_145334/poses.txt"
gt_pose_gsicp = f"/home/aut/thesis_utils/output_analysis/comparison/{dataset}/{scene}/GS_ICP_SLAM/med_20241104_145334/gt_poses.txt"
#pred_pose_gsicp = f"/home/aut/thesis_utils/output_analysis/comparison/{dataset}/{scene}/GS_ICP_SLAM/bad_20241104_144416/poses.txt"
#gt_pose_gsicp = f"/home/aut/thesis_utils/output_analysis/comparison/{dataset}/{scene}/GS_ICP_SLAM/bad_20241104_144416/gt_poses.txt"
pred_gsicp_txt = np.loadtxt(pred_pose_gsicp)
gt_gsicp_txt = np.loadtxt(gt_pose_gsicp)

pred_pose_loopy = f"/home/aut/thesis_utils/output_analysis/comparison/{dataset}/{scene}/LoopySLAM/20241028_151425/Loopy_TUM_RGBD_freiburg3_office_predicted.txt"
gt_pose_loopy = f"/home/aut/thesis_utils/output_analysis/comparison/{dataset}/{scene}/LoopySLAM/20241028_151425/Loopy_TUM_RGBD_freiburg3_office_gt.txt"
pred_loopy_txt = np.loadtxt(pred_pose_loopy)
gt_loopy_txt = np.loadtxt(gt_pose_loopy)

pred_pose_photo = f"/home/aut/thesis_utils/output_analysis/comparison/{dataset}/{scene}/Photo-SLAM/photo_rgbd_dataset_freiburg3_long_office_household/CameraTrajectory_TUM.txt"
gt_pose_photo = f"/home/aut/thesis_utils/output_analysis/comparison/{dataset}/{scene}/Photo-SLAM/photo_rgbd_dataset_freiburg3_long_office_household/groundtruth_matched_photo.txt"
pred_photo_txt = np.loadtxt(pred_pose_photo)
gt_photo_txt = np.loadtxt(gt_pose_photo)


pred_pose_rtabmap = f"/home/aut/thesis_utils/output_analysis/comparison/{dataset}/{scene}/RTABMAP/rtabmap_poses_3.txt"
gt_pose_rtabmap = f"/home/aut/thesis_utils/output_analysis/comparison/{dataset}/{scene}/RTABMAP/poses_gt_3.txt"
pred_rtabmap_txt = np.loadtxt(pred_pose_rtabmap)
gt_rtabmap_txt = np.loadtxt(gt_pose_rtabmap)

pred_pose_orbslam = f"/home/aut/thesis_utils/output_analysis/comparison/{dataset}/{scene}/ORBSLAM3/CameraTrajectoryTUMFR3.txt"
gt_pose_orbslam = f"/home/aut/thesis_utils/output_analysis/comparison/{dataset}/{scene}/ORBSLAM3/groundtruth_matched_orb.txt"
pred_orbslam_txt = np.loadtxt(pred_pose_orbslam)
gt_orbslam_txt = np.loadtxt(gt_pose_orbslam)



## PREPARE DATA FOR PLOTTING ##
# GSICP
pred_gsicp = pred_gsicp_txt[:, [1, 2, 3]]  # Extract tx, ty, tz
gt_gsicp = gt_gsicp_txt[:, [0, 1, 2]]  # Extract tx, ty, tz
# Loopy-SLAM
pred_loopy_init = pred_loopy_txt[1:, 1:4]  # Extract tx, ty, tz
pred_loopy = np.zeros_like(pred_loopy_init)
for i in range(len(pred_loopy_init)):   # Transform the predicted poses to match the ground truth
    pred_loopy[i] = np.dot(origin_R_mat, pred_loopy_init[i]) + origin_t
gt_loopy_init = gt_loopy_txt[1:, 1:4]  # Extract tx, ty, tz
gt_loopy = np.zeros_like(gt_loopy_init)
for i in range(len(gt_loopy_init)):   # Transform the predicted poses to match the ground truth
    gt_loopy[i] = np.dot(origin_R_mat, gt_loopy_init[i]) + origin_t

# Photo-SLAM
pred_photo_init = pred_photo_txt[:, 1:4]  # Extract tx, ty, tz
pred_photo = np.zeros_like(pred_photo_init)
for i in range(len(pred_photo_init)):   # Transform the predicted poses to match the ground truth
    pred_photo[i] = np.dot(origin_R_mat, pred_photo_init[i]) + origin_t
gt_photo = gt_photo_txt[:, 1:4]  # Extract tx, ty, tz

# RTABMAP
pred_rtabmap_txt = pred_rtabmap_txt[1:]  # Remove the first row (header)
pred_rtabmap = pred_rtabmap_txt[:, 1:4]  # Extract tx, ty, tz
gt_rtabmap = gt_rtabmap_txt[1:, 1:4]  # Extract tx, ty, tz

# ORBSLAM
pred_orbslam_init = pred_orbslam_txt[:, 1:4]  # Extract tx, ty, tz
pred_orbslam = np.zeros_like(pred_orbslam_init)
for i in range(len(pred_orbslam_init)):   # Transform the predicted poses to match the ground truth
    pred_orbslam[i] = (np.dot(origin_R_mat, pred_orbslam_init[i]) + origin_t)
gt_orbslam = gt_orbslam_txt[:, 1:4]  # Extract tx, ty, tz


## Plotting the path ##
plt.figure(figsize=(10, 6))
plt.plot(gt_t[:, 0], gt_t[:, 1], 'b-', label="Ground Truth")

plt.plot(gt_gsicp[:, 0], gt_gsicp[:, 1], 'y-', label="GS-ICP-SLAM GT")
#plt.plot(gt_loopy[:, 0], gt_loopy[:, 1], 'c:', label="Loopy-SLAM GT")
#plt.plot(gt_rtabmap[:, 0], gt_rtabmap[:, 1], 'y-.', label="RTABMAP GT")


plt.plot(pred_gsicp[:, 0], pred_gsicp[:, 1], 'r--', label="GS-ICP-SLAM")
plt.plot(pred_loopy[:, 0], pred_loopy[:, 1], 'g-.', label="Loopy-SLAM")
plt.plot(pred_photo[:, 0], pred_photo[:, 1], 'c:', label="Photo-SLAM")
plt.plot(pred_rtabmap[:, 0], pred_rtabmap[:, 1], 'm:', label="RTABMAP")
plt.plot(pred_orbslam[:, 0], pred_orbslam[:, 1], 'k:', label="ORBSLAM3")

## Mark the start and end points ##
plt.plot(gt_t[0, 0], gt_t[0, 1], 'bo', markersize=10, label="GT Start")
plt.plot(gt_t[-1, 0], gt_t[-1, 1], 'b*', markersize=10, label="GT End")
plt.plot(pred_gsicp[0, 0], pred_gsicp[0, 1], 'ro', markersize=12, label="GS-ICP Start")
plt.plot(pred_gsicp[-1, 0], pred_gsicp[-1, 1], 'r*', markersize=12, label="GS-ICP End")
plt.plot(pred_loopy[0, 0], pred_loopy[0, 1], 'go', markersize=12, label="Loopy Start")
plt.plot(pred_loopy[-1, 0], pred_loopy[-1, 1], 'g*', markersize=12, label="Loopy End")
plt.plot(pred_photo[0, 0], pred_photo[0, 1], 'co', markersize=12, label="Photo Start")
plt.plot(pred_photo[-1, 0], pred_photo[-1, 1], 'c*', markersize=12, label="Photo End")
plt.plot(pred_rtabmap[0, 0], pred_rtabmap[0, 1], 'mo', markersize=12, label="RTABMAP Start")
plt.plot(pred_rtabmap[-1, 0], pred_rtabmap[-1, 1], 'm*', markersize=12, label="RTABMAP End")
plt.plot(pred_orbslam[0, 0], pred_orbslam[0, 1], 'ko', markersize=12, label="ORBSLAM3 Start")
plt.plot(pred_orbslam[-1, 0], pred_orbslam[-1, 1], 'k*', markersize=12, label="ORBSLAM3 End")

# Add labels and title
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Trajectory Comparison: Predicted vs Ground Truth")
plt.show()







# Calculate Absolute Trajectory Error (ATE)
err_gsicp = np.linalg.norm(pred_gsicp - gt_gsicp, axis=1) * 100
err_loopy = np.linalg.norm(pred_loopy - gt_loopy, axis=1) * 100
err_rtabmap = np.linalg.norm(pred_rtabmap - gt_rtabmap, axis=1) * 100
err_photo = np.linalg.norm(pred_photo - gt_photo, axis=1) * 100
err_orbslam = np.linalg.norm(pred_orbslam - gt_orbslam, axis=1) * 100





# Length of the trajectory
len_traj = np.max([len(err_gsicp), len(err_loopy), len(err_rtabmap), len(err_photo), len(err_orbslam)])
print(f"Length of the trajectory: {len_traj}")

# Create a uniform x-axis for the resampled data
x_uniform = np.linspace(0, 1, len_traj)

# Resample each error array to the same length
x_gsicp = np.linspace(0, 1, len(err_gsicp))
x_loopy = np.linspace(0, 1, len(err_loopy))
x_rtabmap = np.linspace(0, 1, len(err_rtabmap))
x_photo = np.linspace(0, 1, len(err_photo))
x_orbslam = np.linspace(0, 1, len(err_orbslam))


err_gsicp_resampled = np.interp(x_uniform, x_gsicp, err_gsicp)
err_loopy_resampled = np.interp(x_uniform, x_loopy, err_loopy)
err_rtabmap_resampled = np.interp(x_uniform, x_rtabmap, err_rtabmap)
err_photo_resampled = np.interp(x_uniform, x_photo, err_photo)
err_orbslam_resampled = np.interp(x_uniform, x_orbslam, err_orbslam)

# Plotting the resampled error arrays
plt.figure(figsize=(10, 6))
plt.plot(err_gsicp_resampled, 'r-', label="GS-ICP-SLAM")
plt.plot(err_loopy_resampled, 'g-', label="Loopy-SLAM")
plt.plot(err_rtabmap_resampled, 'm-', label="RTABMAP")
plt.plot(err_photo_resampled, 'c-', label="Photo-SLAM")
plt.plot(err_orbslam_resampled, 'k-', label="ORBSLAM3")
plt.xlabel("Frame Index (Resampled)")
plt.ylabel("Error (cm)")
plt.legend()
plt.title("Positional Error (x,y,z) between Predicted and Ground Truth")
plt.show()



# Calculate ATE, RMSE
ate_gsicp = np.mean(err_gsicp)
ate_loopy = np.mean(err_loopy)
ate_rtabmap = np.mean(err_rtabmap)
ate_photo = np.mean(err_photo)
ate_orbslam = np.mean(err_orbslam)

ate_rmse_gsicp = np.sqrt(np.mean(err_gsicp**2))
ate_rmse_loopy = np.sqrt(np.mean(err_loopy**2))
ate_rmse_rtabmap = np.sqrt(np.mean(err_rtabmap**2))
ate_rmse_photo = np.sqrt(np.mean(err_photo**2))
ate_rmse_orbslam = np.sqrt(np.mean(err_orbslam**2))

# Output results
print(f"ATE GS-ICP-SLAM: {ate_gsicp:.2f} cm, RMSE: {ate_rmse_gsicp:.2f} cm")
print(f"ATE Loopy-SLAM: {ate_loopy:.2f} cm, RMSE: {ate_rmse_loopy:.2f} cm")
print(f"ATE Photo-SLAM: {ate_photo:.2f} cm, RMSE: {ate_rmse_photo:.2f} cm")
print(f"ATE RTABMAP: {ate_rtabmap:.2f} cm, RMSE: {ate_rmse_rtabmap:.2f} cm")
print(f"ATE ORBSLAM3: {ate_orbslam:.2f} cm, RMSE: {ate_rmse_orbslam:.2f} cm")
