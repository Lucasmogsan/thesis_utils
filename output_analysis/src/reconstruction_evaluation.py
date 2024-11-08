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

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from loss_utils import mse2psnr, ssim


dataset = "Replica"     # "TUM" or "Replica"
scene = "office0"       # "rgbd_freiburg3_long_office_household"
prediction_method = "GS_ICP_SLAM"
num_runs = 1

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
    pred_rgb_files = os.listdir(pred_rgb_path)
    
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
                rgb_filename = values.split(values[9], "/")[-1]
                depth_filename = values.split(values[11], "/")[-1]
                gt_rgb_files.append(rgb_filename)
                gt_depth_files.append(depth_filename)
    elif dataset == "Replica":
        gt_rgb_files = os.listdir(gt_rgb_path)
        gt_depth_files = os.listdir(gt_depth_path)

    print(f"Example gt_rgb_file: {gt_rgb_files[0]}")
    print(f"Example gt_depth_file: {gt_depth_files[0]}")  

    psnrs = []
    ssims = []
    lpips = []

    cal_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to("cuda")

    for i in tqdm(range(len(pred_rgb_files))):
        gt_rgb_file = gt_rgb_files[i]
        gt_depth_file = gt_depth_files[i]
        pred_rgb_file = pred_rgb_files[i]

        gt_rgb_image = cv2.imread(os.path.join(gt_rgb_path, gt_rgb_file))
        gt_depth_image = np.array(o3d.io.read_image(os.path.join(gt_depth_path, gt_depth_file))).astype(np.float32)
        pred_rgb_image = cv2.imread(os.path.join(pred_rgb_path, pred_rgb_file))

        # Convert to torch tensors
        gt_rgb = torch.from_numpy(gt_rgb_image).permute(2, 0, 1).float().to("cuda")
        gt_depth = torch.from_numpy(gt_depth_image).to("cuda")
        pred_rgb = torch.from_numpy(pred_rgb_image).permute(2, 0, 1).float().to("cuda")

        # Normalize the torch tensors
        pred_rgb = torch.clamp(pred_rgb, 0., 1.).cuda()
        gt_rgb = torch.clamp(gt_rgb, 0., 1.).cuda()

        # Mask out invalid depth values
        valid_depth_mask_ = (gt_depth>0)
                        
        gt_rgb = gt_rgb * valid_depth_mask_
        pred_rgb = pred_rgb * valid_depth_mask_

        # Calculate PSNR
        square_error = (gt_rgb-pred_rgb)**2
        mse_error = torch.mean(torch.mean(square_error, axis=2))
        psnr = mse2psnr(mse_error)
        psnrs += [psnr.detach().cpu()]

        # Calculate SSIM
        _, ssim_error = ssim(pred_rgb, gt_rgb)
        ssims += [ssim_error.detach().cpu()]

        # Calculate LPIPS
        lpips_value = cal_lpips(gt_rgb.unsqueeze(0), pred_rgb.unsqueeze(0))
        lpips += [lpips_value.detach().cpu()]


    print(f"PSNR: {torch.mean(torch.stack(psnrs))}")
    print(f"SSIM: {torch.mean(torch.stack(ssims))}")
    print(f"LPIPS: {torch.mean(torch.stack(lpips))}")
