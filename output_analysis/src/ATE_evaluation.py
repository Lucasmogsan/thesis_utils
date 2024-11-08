import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

dataset = "TUM"
scene = "freiburg1_desk"

gt_real_path = f"../../download_benchmark_data/dataset/{dataset}/rgbd_dataset_{scene}/groundtruth.txt"
gt_txt = np.loadtxt(gt_real_path)
gt_real_pos = gt_txt[:, 1:4]  # position

align_origin = True
visualize = True


# Load the data
for i in range(0, 3):
    pred_pose_path = f"../../output_analysis/comparison/{dataset}/{scene}/ORBSLAM3/fr1_run_{i}.txt"
    gt_pose_path = f"../../output_analysis/comparison/{dataset}/{scene}/ORBSLAM3/gt_matched_orb_{i}.txt"
    
    pred = np.loadtxt(pred_pose_path, comments='#')
    gt = np.loadtxt(gt_pose_path, comments='#')
    
    # Extract timestamps and positions
    pred_timestamps, pred_positions = pred[:, 0], pred[:, 1:4]
    gt_timestamps, gt_positions = gt[:, 0], gt[:, 1:4]
    
    origin_t = gt[0, 1:4]  # position
    origin_R = gt[0, 4:] # orientation
    origin_R_mat = R.from_quat(origin_R).as_matrix()
    
    if align_origin:
        pred_pos_new_origin = np.zeros_like(pred_positions)
        for i in range(len(pred_positions)):   # Transform the predicted poses to match the ground truth
            pred_pos_new_origin[i] = np.dot(origin_R_mat, pred_positions[i]) + origin_t
            

    # Interpolate ground truth positions to match predicted timestamps
    gt_positions_interp = np.array([
        np.interp(pred_timestamps, gt_timestamps, gt_positions[:, i]) for i in range(3)
    ]).T


    # Calculate Euclidean error for ATE in centimeters
    errors = np.linalg.norm(pred_pos_new_origin - gt_positions, axis=1) * 100
    ate_rmse = np.sqrt(np.mean(errors**2))
    
    # plot the two trajectories (only X and Y)
    if visualize == True:
        plt.figure(figsize=(10, 6))
        plt.plot(gt_real_pos[:, 0], gt_real_pos[:, 1], label='Ground Truth Trajectory', color='blue')
        plt.plot(pred_pos_new_origin[:, 0], pred_pos_new_origin[:, 1], label='Predicted Trajectory', color='red')
        plt.plot(gt_real_pos[0, 0], gt_real_pos[0, 1], 'o', color='blue')
        plt.plot(pred_pos_new_origin[0, 0], pred_pos_new_origin[0, 1], 'o', color='red')
        
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title("Predicted vs Ground Truth Trajectory")
        plt.legend()
        plt.show()

        # Plotting the positional error over time
        # plt.figure(figsize=(10, 6))
        # plt.plot(pred_timestamps, errors, label='Positional Error (cm)')
        # plt.xlabel("Timestamp")
        # plt.ylabel("Error (cm)")
        # plt.title("ATE Positional Error between Predicted and Ground Truth")
        # plt.legend()
        # plt.show()

    print(f"ATE RMSE: {ate_rmse:.3f} cm")
