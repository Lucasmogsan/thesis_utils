import sys
import os
import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms
from PIL import Image  # Use PIL for loading images
import cv2
import open3d as o3d
from tqdm import tqdm
from matplotlib import pyplot as plt

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from loss_utils import mse2psnr, ssim


def natural_sort(file_name):
    # Extract the numeric part from the filename using a simple approach
    number = ''.join(filter(str.isdigit, file_name))  # This extracts the numeric part of the filename
    return int(number)  # Convert the numeric part to an integer

dataset = "TUM"     # "TUM" or "Replica"
scene = "freiburg2_xyz"       # "freiburg3_long_office_household"
prediction_method = "Photo-SLAM"
num_runs = 3

if dataset == "TUM":
    rgb_folder = "rgb"
    depth_folder = "depth"
    dataset_scene = f"rgbd_dataset_{scene}"
elif dataset == "Replica":
    rgb_folder = "images"
    depth_folder = "depth_images"
    dataset_scene = scene

for i in range(num_runs):
    gt_rgb_path = f"../../download_benchmark_data/dataset/{dataset}/{dataset_scene}/{rgb_folder}"
    gt_depth_path = f"../../download_benchmark_data/dataset/{dataset}/{dataset_scene}/{depth_folder}"
    pred_rgb_path = f"../../output_analysis/comparison/{dataset}/{scene}/{prediction_method}/pred_rgb_{i}"

    # Load the data
    pred_rgb_files = sorted(os.listdir(pred_rgb_path), key=natural_sort)
    
    if dataset == "TUM":
        matched_gt_files = f"../../output_analysis/comparison/{dataset}/{scene}/{prediction_method}/gt_matched_rgb_d_{i}.txt"
        gt_rgb_files = []
        gt_depth_files = []
        with open(matched_gt_files, 'r') as f:
            for line in f:
                if line.startswith("#"):
                    continue
                values = line.strip().split()
                # Extract rgb and depth filenames
                rgb_filename = values[9].split("/")[-1]
                depth_filename = values[11].split("/")[-1]
                gt_rgb_files.append(rgb_filename)
                gt_depth_files.append(depth_filename)
    elif dataset == "Replica":
        gt_rgb_files = sorted(os.listdir(gt_rgb_path))
        gt_depth_files = sorted(os.listdir(gt_depth_path))

    print(f"Fifth gt_rgb_file: {gt_rgb_files[4]}")
    print(f"Fifth gt_depth_file: {gt_depth_files[4]}")
    print(f"Fifth pred_rgb_file: {pred_rgb_files[4]}") 

    psnrs = []
    ssims = []
    lpips = []

    cal_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to("cuda")

    for i in tqdm(range(len(pred_rgb_files))):
        gt_rgb_file = gt_rgb_files[i]
        gt_depth_file = gt_depth_files[i]
        pred_rgb_file = pred_rgb_files[i]
        
        # print(f"gt_rgb_file: {gt_rgb_file}", f"gt_depth_file: {gt_depth_file}", f"pred_rgb_file: {pred_rgb_file}")

        gt_rgb_image = cv2.imread(os.path.join(gt_rgb_path, gt_rgb_file))
        gt_depth_image = np.array(o3d.io.read_image(os.path.join(gt_depth_path, gt_depth_file))).astype(np.float32)
        pred_rgb_image = cv2.imread(os.path.join(pred_rgb_path, pred_rgb_file))

        # Convert to torch tensors
        gt_rgb = torch.from_numpy(gt_rgb_image).permute(2, 0, 1).float().to("cuda")
        gt_depth = torch.from_numpy(gt_depth_image).to("cuda")
        pred_rgb = torch.from_numpy(pred_rgb_image).permute(2, 0, 1).float().to("cuda")

        # Normalize the torch tensors
        pred_rgb = torch.clamp(pred_rgb / 255., 0., 1.).cuda()
        gt_rgb = torch.clamp(gt_rgb / 255., 0., 1.).cuda()
        
        # DEBUGGING:
        ## print min and max and avg values of the tensors
        # print(f"gt_rgb min: {torch.min(gt_rgb)} max: {torch.max(gt_rgb)} avg: {torch.mean(gt_rgb)}")
        # print(f"pred_rgb min: {torch.min(pred_rgb)} max: {torch.max(pred_rgb)} avg: {torch.mean(pred_rgb)}")
        
        ## save the images for debugging
        # gt_rgb_image = transforms.ToPILImage()(gt_rgb)
        # cv2.imwrite(f"gt_rgb_{i}.png", np.array(gt_rgb_image))
        # pred_rgb_image = transforms.ToPILImage()(pred_rgb)
        # cv2.imwrite(f"pred_rgb_{i}.png", np.array(pred_rgb_image))
        

        

        # Mask out invalid depth values
        valid_depth_mask_ = (gt_depth > 0).int()
        
        gt_rgb = gt_rgb * valid_depth_mask_
        pred_rgb = pred_rgb * valid_depth_mask_

        # Calculate PSNR
        mse_error = torch.nn.functional.mse_loss(pred_rgb, gt_rgb)
        #square_error = torch.square(gt_rgb - pred_rgb)
        #mse_error = torch.mean(square_error)
        psnr = mse2psnr(mse_error)
        if psnr == float("inf"):    # For debugging purposes
            print(f"PSNR is inf for iteration {i}")

        psnrs += [psnr.detach().cpu()]

        # Calculate SSIM
        _, ssim_error = ssim(gt_rgb, pred_rgb)      # could be using pytorch_msssim if desired
        ssims += [ssim_error.detach().cpu()]

        # Calculate LPIPS
        lpips_value = cal_lpips(gt_rgb.unsqueeze(0).float(), pred_rgb.unsqueeze(0).float())
        lpips += [lpips_value.detach().cpu()]
        
        
    print(f"PSNR: {torch.mean(torch.stack(psnrs))}")
    print(f"SSIM: {torch.mean(torch.stack(ssims))}")
    print(f"LPIPS: {torch.mean(torch.stack(lpips))}")
