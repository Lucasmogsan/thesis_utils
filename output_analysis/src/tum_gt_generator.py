import numpy as np

# Function to find the closest ground truth timestamp for a given depth timestamp
def find_closest_pose(depth_timestamp, gt_poses):
    time_diffs = np.abs(gt_poses[:, 0] - depth_timestamp)
    closest_idx = np.argmin(time_diffs)
    return gt_poses[closest_idx]


dataset = "TUM"
scene = "freiburg1_room"
method = "Photo-SLAM"
print(f"Generating ground truth poses for {scene} comparing to {method} predictions")

if dataset == "TUM":
    scene_dataset = f"rgbd_dataset_{scene}"

# Paths to the ground truth trajectory and depth file
gt_traj_path = f"../../download_benchmark_data/dataset/TUM/{scene_dataset}/groundtruth.txt"


for i in range(0, 3):
    traj_path = f"../../output_analysis/comparison/TUM/{scene}/{method}/fr1_run_{i}.txt"    # Path to the predicted trajectory

    # Load ground truth poses
    gt_poses = []
    with open(gt_traj_path, 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue
            values = line.strip().split()
            # Extract timestamp and pose data from ground truth
            timestamp = float(values[0])
            tx, ty, tz = map(float, values[1:4])
            qx, qy, qz, qw = map(float, values[4:])
            gt_poses.append((timestamp, tx, ty, tz, qx, qy, qz, qw))

    # Convert ground truth list to numpy array for easier indexing
    gt_poses = np.array(gt_poses)

    # Load depth file timestamps
    depth_timestamps = []
    with open(traj_path, 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue
            values = line.strip().split()
            timestamp = float(values[0])
            filename = values[1]
            depth_timestamps.append((timestamp, filename))



    # Match depth timestamps to ground truth poses and save results
    matched_poses = []

    for depth_timestamp, _ in depth_timestamps:
        closest_pose = find_closest_pose(depth_timestamp, gt_poses)
        matched_poses.append(closest_pose)

    # save to file but name it matched_poses.txt
    output_file = gt_traj_path.replace("groundtruth.txt", f"gt_matched_{method}_{i}.txt")
    print(f"Saving matched poses to {output_file}")

    with open(output_file, 'w') as f:
        # f.write("# timestamp tx ty tz qx qy qz qw\n")
        for match in matched_poses:
            f.write(f"{match[0]} {match[1]} {match[2]} {match[3]} {match[4]} {match[5]} {match[6]} {match[7]}\n")