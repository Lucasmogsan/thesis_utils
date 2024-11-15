import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

dataset = "TUM"
scene = "freiburg1_desk"
method = "Photo-SLAM"

align_origin = True
visualize = True


# Load ground truth poses
if dataset == "TUM":
    scene_dataset = f"rgbd_dataset_{scene}"
    fr_num = scene.split("_")[0][-1]    # Extract the number from the scene name
    pred_run = f"fr{fr_num}_run"
    gt_name = "groundtruth"
elif dataset == "Replica":
    scene_dataset = scene
    pred_run = scene
    gt_name = "traj"
    

gt_real_path = f"../../download_benchmark_data/dataset/{dataset}/{scene_dataset}/{gt_name}.txt"
gt_txt = np.loadtxt(gt_real_path)
gt_real_pos = gt_txt[:, 1:4]  # position

if dataset == "Replica":
    # Convert the Replica poses (in homogeneous matrix format) to the same format as TUM (tx ty tz qx qy qz qw)
    # Initialize arrays for translation and quaternion components
    gt_translation = gt_txt[:, [3, 7, 11]]  # Extract the translation vector (tx, ty, tz) from each matrix
    gt_quaternion = np.zeros((len(gt_txt), 4))  # Initialize an array for quaternions (qx, qy, qz, qw)
    for i in range(len(gt_txt)):
        # Extract the 3x3 rotation matrix from the 4x4 homogeneous matrix (first 9 values in row-major order)
        matrix_4x4 = np.array(gt_txt[i]).reshape(4, 4)
        R_mat = matrix_4x4[:3, :3]
        # Convert rotation matrix to quaternion (in the form of [qx, qy, qz, qw])
        R_quat = R.from_matrix(R_mat).as_quat()
        # Store the quaternion in the array
        gt_quaternion[i] = R_quat
    # Concatenate the translation (tx, ty, tz) with the quaternion (qx, qy, qz, qw)
    gt_txt = np.concatenate((gt_translation, gt_quaternion), axis=1)
    gt_real_pos = gt_txt[:, 0:3]  # position


# Load the data
for i in range(0, 3):
    pred_pose_path = f"../../output_analysis/comparison/{dataset}/{scene}/{method}/{pred_run}_{i}.txt"
    pred = np.loadtxt(pred_pose_path, comments='#')
    pred_timestamps, pred_positions = pred[:, 0], pred[:, 1:4]
    
    if dataset == "TUM":
        gt_pose_path = f"../../output_analysis/comparison/{dataset}/{scene}/{method}/gt_matched_{method}_{i}.txt"
        gt = np.loadtxt(gt_pose_path, comments='#')
        gt_timestamps, gt_positions = gt[:, 0], gt[:, 1:4]
        
        origin_t = gt[0, 1:4]  # position
        origin_R = gt[0, 4:] # orientation (quaternion)

        # Interpolate ground truth positions to match predicted timestamps
        gt_positions_interp = np.array([
            np.interp(pred_timestamps, gt_timestamps, gt_positions[:, i]) for i in range(3)
        ]).T


    elif dataset == "Replica":
        gt_positions = gt_translation
        origin_t = gt_translation[0, :]  # position
        origin_R = gt_quaternion[0, :] # orientation (quaternion)
    
    
    origin_R_mat = R.from_quat(origin_R).as_matrix()
    
    # Extract timestamps and positions
    if align_origin:
        pred_pos_new_origin = np.zeros_like(pred_positions)
        for i in range(len(pred_positions)):   # Transform the predicted poses to match the ground truth
            pred_pos_new_origin[i] = np.dot(origin_R_mat, pred_positions[i]) + origin_t
    
    
    



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
