import numpy as np

scene = "rgbd_dataset_freiburg3_long_office_household"

# Paths to the ground truth trajectory and depth file
gt_traj_path = f"/home/aut/thesis_utils/download_benchmark_data/dataset/TUM/{scene}/groundtruth.txt"
gt_depth_path = f"/home/aut/thesis_utils/download_benchmark_data/dataset/TUM/{scene}/depth.txt"
gt_rgb_path = f"/home/aut/thesis_utils/download_benchmark_data/dataset/TUM/{scene}/rgb.txt"

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
with open(gt_depth_path, 'r') as f:
    for line in f:
        if line.startswith("#"):
            continue
        values = line.strip().split()
        timestamp = float(values[0])
        filename = values[1]
        depth_timestamps.append((timestamp, filename))

# Function to find the closest ground truth timestamp for a given depth timestamp
def find_closest_pose(depth_timestamp, gt_poses):
    time_diffs = np.abs(gt_poses[:, 0] - depth_timestamp)
    closest_idx = np.argmin(time_diffs)
    return gt_poses[closest_idx]

# Match depth timestamps to ground truth poses and save results
matched_poses = []

for depth_timestamp, _ in depth_timestamps:
    closest_pose = find_closest_pose(depth_timestamp, gt_poses)
    matched_poses.append(closest_pose)

# save to file but name it matched_poses.txt
output_file = gt_traj_path.replace("groundtruth.txt", "groundtruth_matched.txt")
print(f"Saving matched poses to {output_file}")

with open(output_file, 'w') as f:
    # f.write("# timestamp tx ty tz qx qy qz qw\n")
    for match in matched_poses:
        f.write(f"{match[0]} {match[1]} {match[2]} {match[3]} {match[4]} {match[5]} {match[6]} {match[7]}\n")