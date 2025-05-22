""" This file contains some utils functions for visualization.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.
"""

import torch
import torchvision.transforms.functional as F
from einops import rearrange
import cv2
import numpy as np

def make_viz_from_samples(
    original_masks,
    masks,
    extra_results_dict,
    images=None
):
    """Generates visualization images from original images and reconstructed images.

    Args:
        original_masks: A torch.Tensor of shape (B,C,H,W), original masks.
        masks: A torch.Tensor of shape (B,C,H,W), reconstructed masks.
        masks_token_only: A torch.Tensor of shape (B,C,H,W), masks from token only.
        images: Optional torch.Tensor, background images to overlay masks on.

    Returns:
        A tuple containing two lists - images_for_saving and images_for_logging.
    """
    mean_iou, mean_biou, mean_mse = compute_metrics_per_image(original_masks, masks)
    masks = torch.clamp(masks, 0.0, 1.0)* 255.0
    masks_token_only = extra_results_dict["masks_token_only"]
    masks_token_only = torch.clamp(masks_token_only, 0.0, 1.0)* 255.0
    masks = masks.cpu().float()
    masks_token_only = masks_token_only.cpu().float()
    original_masks = torch.clamp(original_masks, 0.0, 1.0)* 255.0
    original_masks = original_masks.cpu().float()
    images = images.cpu().float()*255.0
    random_morphology_augment = extra_results_dict["random_morphology_augment"]
    random_morphology_augment = random_morphology_augment.cpu().float()*255.0

    if images is not None:
        # Resize masks to match image dimensions
        original_masks = torch.nn.functional.interpolate(original_masks, size=images.shape[-2:], mode='bilinear')
        masks = torch.nn.functional.interpolate(masks, size=images.shape[-2:], mode='bilinear')
        masks_token_only = torch.nn.functional.interpolate(masks_token_only, size=images.shape[-2:], mode='bilinear')
        random_morphology_augment = torch.nn.functional.interpolate(random_morphology_augment, size=images.shape[-2:], mode='bilinear')
        
        # Create overlays
        images = images.cpu()
        overlay_original = images.clone()
        overlay_recon = images.clone()
        overlay_token = images.clone()
        overlay_random_morphology_augment = images.clone()
        
        # Increase alpha for more visibility
        alpha = 0.7
        
        # Original masks overlay
        overlay_original[:, 0] = torch.clamp(overlay_original[:, 0] + alpha * original_masks[:, 0], 0, 255)
        overlay_original[:, 1] = torch.clamp(overlay_original[:, 1] * (1 - 0.3 * original_masks[:, 0] / 255.0), 0, 255)
        overlay_original[:, 2] = torch.clamp(overlay_original[:, 2] * (1 - 0.3 * original_masks[:, 0] / 255.0), 0, 255)
        
        # Reconstructed masks overlay
        overlay_recon[:, 0] = torch.clamp(overlay_recon[:, 0] + alpha * masks[:, 0], 0, 255)
        overlay_recon[:, 1] = torch.clamp(overlay_recon[:, 1] * (1 - 0.3 * masks[:, 0] / 255.0), 0, 255)
        overlay_recon[:, 2] = torch.clamp(overlay_recon[:, 2] * (1 - 0.3 * masks[:, 0] / 255.0), 0, 255)
        
        # Token-only masks overlay
        overlay_token[:, 0] = torch.clamp(overlay_token[:, 0] + alpha * masks_token_only[:, 0], 0, 255)
        overlay_token[:, 1] = torch.clamp(overlay_token[:, 1] * (1 - 0.3 * masks_token_only[:, 0] / 255.0), 0, 255)
        overlay_token[:, 2] = torch.clamp(overlay_token[:, 2] * (1 - 0.3 * masks_token_only[:, 0] / 255.0), 0, 255)
        
        # Random morphology augment overlay
        overlay_random_morphology_augment[:, 0] = torch.clamp(overlay_random_morphology_augment[:, 0] + alpha * random_morphology_augment[:, 0], 0, 255)
        overlay_random_morphology_augment[:, 1] = torch.clamp(overlay_random_morphology_augment[:, 1] * (1 - 0.3 * random_morphology_augment[:, 0] / 255.0), 0, 255)
        overlay_random_morphology_augment[:, 2] = torch.clamp(overlay_random_morphology_augment[:, 2] * (1 - 0.3 * random_morphology_augment[:, 0] / 255.0), 0, 255)
        
        diff_img = torch.abs(original_masks - masks)
        to_stack = [overlay_original, overlay_random_morphology_augment, overlay_recon, diff_img]
    else:
        diff_img = torch.abs(original_masks - masks)
        to_stack = [original_masks, masks, masks_token_only, diff_img]

    images_for_logging = rearrange(
            torch.stack(to_stack),
            "(l1 l2) b c h w -> b c h (l1 l2 w)",
            l1=2).byte()
    images_for_saving = [F.to_pil_image(image) for image in images_for_logging]

    return images_for_saving, images_for_logging, mean_iou, mean_biou, mean_mse




def make_viz_from_samples_generation(
    generated_images,
):
    generated = torch.clamp(generated_images, 0.0, 1.0) * 255.0
    images_for_logging = rearrange(
        generated, 
        "(l1 l2) c h w -> c (l1 h) (l2 w)",
        l1=2)

    images_for_logging = images_for_logging.cpu().byte()
    images_for_saving = F.to_pil_image(images_for_logging)

    return images_for_saving, images_for_logging


def compute_metrics_biou(gt_mask, pred_mask, dilation_ratio=0.02):
    """
    计算单张图片的IoU和Boundary IoU
    
    Args:
        pred_mask: 预测的mask，二值图像numpy数组 (H, W)
        gt_mask: 真实的mask，二值图像numpy数组 (H, W)
        dilation_ratio: boundary的宽度比例，默认0.02
        
    Returns:
        iou: 普通IoU值
        boundary_iou: Boundary IoU值
    """
    # 确保mask是二值的
    pred_mask = pred_mask > 0.5
    gt_mask = gt_mask > 0.5
    # 计算Boundary IoU
    def mask_to_boundary(mask, dilation_ratio):
        # 1. 计算边界宽度
        h, w = mask.shape
        img_diag = np.sqrt(h ** 2 + w ** 2)  # 计算图像对角线长度
        dilation = int(round(dilation_ratio * img_diag))  # 边界宽度 = 比例 * 对角线长度
        dilation = max(1, dilation)
        # 2. 对mask进行处理
        mask = mask.astype(np.uint8)
        # 四周填充1个像素，防止边缘处理时的边界效应
        new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        # 3. 腐蚀操作
        kernel = np.ones((3, 3), dtype=np.uint8)
        new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
        # 4. 裁剪回原始大小
        mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
        # 5. 原始mask减去腐蚀后的mask，得到边界
        return mask - mask_erode
    
    # 获取两个mask的边界
    gt_boundary = mask_to_boundary(gt_mask.astype(np.uint8), dilation_ratio)
    pred_boundary = mask_to_boundary(pred_mask.astype(np.uint8), dilation_ratio)
    # 计算boundary的IoU
    boundary_intersection = ((gt_boundary * pred_boundary) > 0).sum()
    boundary_union = ((gt_boundary + pred_boundary) > 0).sum()
    boundary_iou = boundary_intersection / (boundary_union + 1e-10)
    
    return boundary_intersection, boundary_union, boundary_iou

def compute_metrics_iou(img, rec):
    img_binary = (img > 0.5).float().cpu()
    rec_binary = (rec > 0.5).float().cpu()
    intersection = torch.logical_and(img_binary, rec_binary).sum(dim=[1,2,3])
    union = torch.logical_or(img_binary, rec_binary).sum(dim=[1,2,3])
    iou = (intersection + 1e-6) / (union + 1e-6)
    mean_iou = iou.mean()
    return mean_iou
def compute_metrics_mse(img, rec):
    mse = torch.mean((img.cpu() - rec.cpu()) ** 2, dim=[1,2,3])
    mean_mse = mse.mean()
    return mean_mse

def compute_metrics(img, rec, batch_i):
    # Calculate IOU
    mean_iou = compute_metrics_iou(img, rec)
    # Calculate MSE
    mean_mse = compute_metrics_mse(img, rec)
    # Calculate Boundary IoU
    batch_size = img.size(0)
    total_boundary_iou = 0
    img_binary = (img > 0.5).float().cpu()
    rec_binary = (rec > 0.5).float().cpu()
    for i in range(batch_size):
        img_np = img_binary[i, 0].numpy()  # Assuming single channel mask
        rec_np = rec_binary[i, 0].numpy()
        _, _, boundary_iou = compute_metrics_biou(img_np, rec_np)
        total_boundary_iou += boundary_iou
    
    batch_biou = total_boundary_iou / batch_size
    print(f"Batch {batch_i} - Mean IOU: {mean_iou:.4f}, Mean BIoU: {batch_biou:.4f}, Mean MSE: {mean_mse:.4f}")
    return mean_iou, batch_biou, mean_mse

def compute_metrics_per_image(img, rec):
    batch_size = img.size(0)
    ious = []
    mses = []
    bious = []
    
    for i in range(batch_size):
        # 计算单张图片的 IoU
        img_i = img[i:i+1]
        rec_i = rec[i:i+1]
        iou = compute_metrics_iou(img_i, rec_i)
        mse = compute_metrics_mse(img_i, rec_i)
        
        # 计算单张图片的 Boundary IoU
        img_np = (img_i > 0.5).float().cpu().numpy()[0, 0]
        rec_np = (rec_i > 0.5).float().cpu().numpy()[0, 0]
        _, _, boundary_iou = compute_metrics_biou(img_np, rec_np)
        
        ious.append(iou.item())
        mses.append(mse.item())
        bious.append(boundary_iou)
    
    # 计算批次平均值
    mean_iou = sum(ious) / batch_size
    mean_mse = sum(mses) / batch_size
    mean_biou = sum(bious) / batch_size
    
    # print(f"- Mean IOU: {mean_iou:.4f}, Mean BIoU: {mean_biou:.4f}, Mean MSE: {mean_mse:.4f}")
    return mean_iou, mean_biou, mean_mse