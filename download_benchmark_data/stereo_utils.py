import numpy as np
import cv2

def load_images(path_left, path_right, CAM_LEFT, CAM_RIGHT, img_idx=0):

    img_number = str(img_idx).zfill(10)

    if CAM_LEFT == 0:
        left_image = cv2.imread(f"{path_left}/{img_number}.png", cv2.IMREAD_GRAYSCALE)
    else:
        left_image = cv2.cvtColor(cv2.imread(f"{path_left}/{img_number}.png"), cv2.COLOR_BGR2RGB)

    if CAM_RIGHT == 1:
        right_image = cv2.imread(f"{path_right}/{img_number}.png", cv2.IMREAD_GRAYSCALE)
    else:
        right_image = cv2.cvtColor(cv2.imread(f"{path_right}/{img_number}.png"), cv2.COLOR_BGR2RGB)

    # Check if images are loaded
    if left_image is None or right_image is None:
        print("Error: Could not load one or both images. Check the file paths.")
        exit()

    return left_image, right_image

def initialize_sgmb_params(num_disparities=10*16, block_size=7, window_size=5):
        # P1 and P2 control disparity smoothness (recommended values below)
    P1 = 2 * 3 * window_size**2
    P2 = 64 * 3 * window_size**2

    # Create stereo block matching object
    sgbm_obj = cv2.StereoSGBM_create(
        minDisparity=5,     # Minimum possible disparity value
        numDisparities=num_disparities,  # Wider disparity range for outdoor scenes
        blockSize=block_size,         # Moderate block size for balanced details and noise
        P1=P1,     # Adjusted for grayscale images
        P2=P2,
        disp12MaxDiff=5,     # Enforce left-right consistency - if disparity difference is > XX, reject the pixel
        uniquenessRatio=5,  # Avoid ambiguous matches
        speckleWindowSize=5, # Speckle window size is set to XX pixels (mea)
        speckleRange=5,       # Speckle range is set to X pixels (meaning points with disparity of X pixels or less are considered speckles and are removed)
        mode=cv2.STEREO_SGBM_MODE_HH  # Better results for calibrated cameras
    )

    return sgbm_obj

def compute_sgbm_disparity(sgbm_obj, left_image, right_image, CAM_LEFT=0, CAM_RIGHT=1):
    ''' Computes Disparity map from stereo images using the StereoBM algorithm.'''

    if CAM_LEFT == 2:
        # Convert RGB images to grayscale
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    if CAM_RIGHT == 3:
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    # Compute disparity map
    disparity = sgbm_obj.compute(left_image, right_image).astype(np.float32) / 16.0     # divide by 16 to get the real disparity values (as OpenCV uses fixed-point arithmetic, scaling by 16)

    return disparity


def calc_depth_map(disp_left, K_left, T_left, T_right):
    ''' Computes Depth map from Intrinsic Camera Matrix and Translations vectors.
        For KITTI, the depth is in meters.
        '''
    # Get the focal length from the K matrix
    f = K_left[0, 0]
    
    # Get the distance between the cameras from the t matrices (baseline)
    b = np.abs(T_left[0] - T_right[0])[0]

    #print(f"f: {f}, b: {b}")
    
    # Replace all instances of 0 and -1 disparity with a small minimum value (to avoid div by 0 or negatives)
    disp_left[disp_left <= 0] = 1e-5
    
    # Calculate the depths 
    depth_map = f * b / disp_left

    return depth_map.astype(np.float32)


def clip_depth_map_to_uint16(depth_map, max_depth):
    ''' Clips and converts the depth map to a 16-bit range for visualization.'''
    # Clip depth map to max_depth
    depth_map = np.clip(depth_map, 0, max_depth)
    # Remove all instances of max_depth
    depth_map[depth_map == max_depth] = 0
    # Normalize depth map to 16-bit range for visualization
    depth_map = cv2.normalize(depth_map, None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
    # Convert to uint16 for display or saving
    depth_map = depth_map.astype(np.uint16)

    return depth_map


def get_calib_matrices(CAM_LEFT, CAM_RIGHT, path_calib_cam_to_cam):
    # Read calibration file
    with open(path_calib_cam_to_cam, "r") as file:
        lines = file.readlines()
    K_left = None
    P_left = None
    T_left = None
    T_right = None
    for line in lines:
        if line.startswith(f"K_0{CAM_LEFT}"):
            K_left = np.array([float(x) for x in line.split(":")[1].strip().split()])
        if line.startswith(f"P_rect_0{CAM_LEFT}"):
            P_left = np.array([float(x) for x in line.split(":")[1].strip().split()])
        if line.startswith(f"T_0{CAM_LEFT}"):
            T_left = np.array([float(x) for x in line.split(":")[1].strip().split()])
        if line.startswith(f"T_0{CAM_RIGHT}"):
            T_right = np.array([float(x) for x in line.split(":")[1].strip().split()])
    # Reshape into matrices
    K_left = K_left.reshape(3, 3)
    P_left = P_left.reshape(3, 4)
    T_left = T_left.reshape(3, 1)
    T_right = T_right.reshape(3, 1)

    return K_left, P_left, T_left, T_right