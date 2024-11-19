import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

dataset = "TUM"
scene = "freiburg1_desk"

# Load ground truth poses
gt_pose = f"../../output_analysis/comparison/{dataset}/{scene}/gt/groundtruth.txt"
gt_txt = np.loadtxt(gt_pose)
gt_t = gt_txt[:, 1:4]  # position
gt_R = gt_txt[:, 4:] # orientation
origin_t = gt_t[0]
origin_R = gt_R[0]
origin_R_mat = R.from_quat(origin_R).as_matrix()

# Load predicted poses
pred_pose_photo = f"../../output_analysis/comparison/{dataset}/{scene}/Photo-SLAM/fr1_run_2.txt"
gt_pose_photo = f"../../output_analysis/comparison/{dataset}/{scene}/Photo-SLAM/gt_matched_Photo-SLAM_2.txt"
pred_photo_txt = np.loadtxt(pred_pose_photo)
gt_photo_txt = np.loadtxt(gt_pose_photo)
o_t_photo = gt_photo_txt[:, 1:4][2]
o_r_photo = R.from_quat(gt_photo_txt[:, 4:][0]).as_matrix()



pred_pose_orbslam = f"../../output_analysis/comparison/{dataset}/{scene}/ORBSLAM3/fr1_run_2.txt"
gt_pose_orbslam = f"../../output_analysis/comparison/{dataset}/{scene}/ORBSLAM3/gt_matched_orb_2.txt"
pred_orbslam_txt = np.loadtxt(pred_pose_orbslam)
gt_orbslam_txt = np.loadtxt(gt_pose_orbslam)
o_t_orb = gt_orbslam_txt[:, 1:4][2]
o_r_orb = R.from_quat(gt_orbslam_txt[:, 4:][0]).as_matrix()



## PREPARE DATA FOR PLOTTING ##

# Photo-SLAM
pred_photo_init = pred_photo_txt[:, 1:4]  # Extract tx, ty, tz
pred_photo = np.zeros_like(pred_photo_init)
for i in range(len(pred_photo_init)):   # Transform the predicted poses to match the ground truth
    pred_photo[i] = np.dot(o_r_photo, pred_photo_init[i]) + o_t_photo
gt_photo = gt_photo_txt[:, 1:4]  # Extract tx, ty, tz

# ORBSLAM
pred_orbslam_init = pred_orbslam_txt[:, 1:4]  # Extract tx, ty, tz
pred_orbslam = np.zeros_like(pred_orbslam_init)
for i in range(len(pred_orbslam_init)):   # Transform the predicted poses to match the ground truth
    pred_orbslam[i] = (np.dot(o_r_orb, pred_orbslam_init[i]) + o_t_orb)
gt_orbslam = gt_orbslam_txt[:, 1:4]  # Extract tx, ty, tz


## Plotting the path ##
plt.figure(figsize=(10, 6))
plt.plot(gt_t[:, 0], gt_t[:, 1], '-', label="Ground Truth", color="blue")

#plt.plot(gt_gsicp[:, 0], gt_gsicp[:, 1], 'y-', label="GS-ICP-SLAM GT")
#plt.plot(gt_loopy[:, 0], gt_loopy[:, 1], 'c:', label="Loopy-SLAM GT")
#plt.plot(gt_rtabmap[:, 0], gt_rtabmap[:, 1], 'y-.', label="RTABMAP GT")


plt.plot(pred_photo[:, 0], pred_photo[:, 1], '-', label="Photo-SLAM", color="cyan")
plt.plot(pred_orbslam[:, 0], pred_orbslam[:, 1], '-', label="ORBSLAM3", color="orange")

## Mark the start and end points ##
plt.plot(gt_t[0, 0], gt_t[0, 1], 'o', markersize=10, label="GT Start", color="blue")
plt.plot(gt_t[-1, 0], gt_t[-1, 1], '*', markersize=10, label="GT End", color="blue")

plt.plot(pred_photo[0, 0], pred_photo[0, 1], 'o', markersize=12, label="Photo Start", color="cyan")
plt.plot(pred_photo[-1, 0], pred_photo[-1, 1], '*', markersize=12, color="cyan")

plt.plot(pred_orbslam[0, 0], pred_orbslam[0, 1], 'o', markersize=12, label="ORB-SLAM3 Start", color="orange")
plt.plot(pred_orbslam[-1, 0], pred_orbslam[-1, 1], '*', markersize=12, color="orange")

# Add labels and title
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend()
plt.title("Trajectory Comparison: Predicted vs Ground Truth")
plt.show()