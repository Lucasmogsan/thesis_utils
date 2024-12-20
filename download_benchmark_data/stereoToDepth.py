import cv2
import numpy as np
from stereo_utils import load_images, compute_sgbm_disparity, calc_depth_map, get_calib_matrices, clip_depth_map_to_uint16, initialize_sgmb_params
import os
from tqdm import tqdm

dataset = 'KITTI'
scene = '2011_09_26_drive_0093'
max_depth = 30  # Maximum depth in meters

visualize_single_image = False

CAM_LEFT = 2
CAM_RIGHT = 3
CAM_RGB_LEFT = 2
CAM_RGB_RIGHT = 3

# path to data:
path_left = f"dataset/{dataset}/{scene}/2011_09_26_drive_0093_sync/image_0{CAM_LEFT}/data"
path_right = f"dataset/{dataset}/{scene}/2011_09_26_drive_0093_sync/image_0{CAM_RIGHT}/data"
path_rgb = f"dataset/{dataset}/{scene}/2011_09_26_drive_0093_sync/image_0{CAM_RGB_LEFT}/data"
# Load calibration file
path_calib = f"dataset/{dataset}/{scene}/2011_09_26_calib/calib_cam_to_cam.txt"


sgbm_obj = initialize_sgmb_params()

# Get calibration matrices
K_left, P_left, T_left, T_right = get_calib_matrices(CAM_LEFT, CAM_RIGHT, path_calib)

# If using the P-matrix to get the intrinsic matrix (if images are rectified):
if P_left is not None:
    K_left = P_left[:, :3]

# Load images
img_index = 0

img_indexes = len(os.listdir(path_left))

if visualize_single_image:
    img_indexes = 5

for img_index in tqdm(range(img_indexes)):
    left_image, right_image = load_images(path_left, path_right, CAM_LEFT, CAM_RIGHT, img_idx=img_index)

    # Compute disparity map
    disparity = compute_sgbm_disparity(sgbm_obj, left_image, right_image, CAM_LEFT=CAM_LEFT, CAM_RIGHT=CAM_RIGHT)

    # Compute depth map
    depth_map = calc_depth_map(disparity, K_left, T_left, T_right)

    # Clip and convert depth map to 16-bit range for visualization
    depth_map = clip_depth_map_to_uint16(depth_map, max_depth)

    # Filter depth map
    filtered_depth_map = cv2.medianBlur(depth_map, 5)

    # Save depth map to file
    img_number = str(img_index).zfill(10)
    path_new_depth_map = f"dataset/{dataset}/{scene}/depth_stereo_d{max_depth}"
    if not os.path.exists(path_new_depth_map):
        os.makedirs(path_new_depth_map)
        
    cv2.imwrite(f"{path_new_depth_map}/{img_number}.png", filtered_depth_map)

# IF DISPLAY:
if visualize_single_image:
    # plot disparity map with cv2
    cv2.imshow("left", left_image)
    cv2.imshow("right", right_image)
    cv2.imshow("disparity", disparity)
    cv2.imshow("depth", filtered_depth_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Disparity map:")
    print(disparity.shape)
    print(disparity.dtype)
    print("Disparity map range:", disparity.min(), disparity.max())

    print("Depth map:")
    print(depth_map.shape)
    print(depth_map.dtype)
    print("Depth map range:", depth_map[depth_map > 0].min(), depth_map[depth_map > 0].max())
    print("Average depth:", np.mean(depth_map[depth_map > 0]))



