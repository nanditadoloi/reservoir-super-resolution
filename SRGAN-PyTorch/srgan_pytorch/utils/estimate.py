# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from .image_quality_assessment import GMSD
from .image_quality_assessment import LPIPS
from .image_quality_assessment import SSIM

__all__ = [
    "iqa", "test"
]


def iqa(source: torch.Tensor, target: torch.Tensor, gpu: int) -> [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Image quality evaluation function.

    Args:
        source (torch.Tensor): Original tensor picture.
        target (torch.Tensor): Target tensor picture.
        gpu (int): Graphics card index.

    Returns:
        MSE, RMSE, PSNR, SSIM, LPIPS, GMSD.
    """
    mse_loss = nn.MSELoss().cuda(gpu).eval()
    # Reference sources from https://github.com/richzhang/PerceptualSimilarity
    lpips_loss = LPIPS(gpu).cuda(gpu).eval()
    # Reference sources from https://github.com/dingkeyan93/IQA-optimization/blob/master/IQA_pytorch/SSIM.py
    ssim_loss = SSIM().cuda(gpu).eval()
    # Reference sources from http://www4.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.htm
    gmsd_loss = GMSD().cuda(gpu).eval()

    # Complete estimate.
    with torch.no_grad():
        mse_value = mse_loss(source, target)
        rmse_value = torch.sqrt(mse_value)
        psnr_value = 10 * torch.log10(1. / mse_value)
        ssim_value = ssim_loss(source, target)
        lpips_value = lpips_loss(source, target)
        gmsd_value = gmsd_loss(source, target)

    return mse_value, rmse_value, psnr_value, ssim_value, lpips_value, gmsd_value


def test(dataloader: torch.utils.data.DataLoader, model: nn.Module, gpu: int) -> [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mse_loss = nn.MSELoss().cuda(gpu).eval()
    # Reference sources from https://hub.fastgit.org/dingkeyan93/IQA-optimization/blob/master/IQA_pytorch/SSIM.py
    ssim_loss = SSIM().cuda(gpu).eval()
    # Reference sources from https://github.com/richzhang/PerceptualSimilarity
    lpips_loss = LPIPS(gpu).cuda(gpu).eval()
    # Reference sources from http://www4.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.htm
    gmsd_loss = GMSD().cuda(gpu).eval()

    # switch eval mode.
    model.eval()
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    total_psnr_value = 0.
    total_ssim_value = 0.
    total_lpips_value = 0.
    total_gmsd_value = 0.
    total = len(dataloader)
    total_hr_mass = 0.
    total_lr_mass = 0.
    total_gt_mass = 0.
    total_hs_mass = 0.

    with torch.no_grad():
        for i, (lr, _, hr, lr_C, hr_C) in progress_bar:
            # Move data to special device.
            if gpu is not None:
                lr = lr.cuda(gpu, non_blocking=True)
            if torch.cuda.is_available():
                hr = hr.cuda(gpu, non_blocking=True)
                lr_C = lr_C.cuda(gpu, non_blocking=True)
                hr_C = hr_C.cuda(gpu, non_blocking=True)

            sr = model(lr)
            np_sr = sr.cpu().numpy().astype(np.single)
            np_hr = hr.cpu().numpy().astype(np.single)
            np_lr = lr.cpu().numpy().astype(np.single)
            np_lr_C = lr_C.cpu().numpy().astype(np.single)
            np_hr_C = hr_C.cpu().numpy().astype(np.single)
            #print(np_sr.shape, np_hr.shape, np_lr_C.shape, "=============== Shape")

            ## Test Mass Balance
            sr_mass = np.sum(np.multiply(np_sr,np_hr_C))/(3.0*np_lr.shape[0]) # n * 3 * x * y
            lr_mass = np.sum(np.multiply(np_lr,np_lr_C))/(3.0*np_lr.shape[0])
            hr_mass = np.sum(np.multiply(np_hr,np_hr_C))/(3.0*np_lr.shape[0])
            total_hr_mass += (sr_mass-hr_mass)*100.0/hr_mass
            total_lr_mass += (sr_mass-lr_mass)*100.0/lr_mass
            total_gt_mass += (hr_mass-lr_mass)*100.0/lr_mass
            total_hs_mass = (total_lr_mass-total_gt_mass)*100.0/total_gt_mass

            # The MSE Loss of the generated fake high-resolution image and real high-resolution image is calculated.
            total_psnr_value += 10 * torch.log10(1. / mse_loss(sr, hr))
            # The SSIM of the generated fake high-resolution image and real high-resolution image is calculated.
            total_ssim_value += ssim_loss(sr, hr)
            # The LPIPS of the generated fake high-resolution image and real high-resolution image is calculated.
            total_lpips_value += lpips_loss(sr, hr)
            # The GMSD of the generated fake high-resolution image and real high-resolution image is calculated.
            total_gmsd_value += gmsd_loss(sr, hr)

            progress_bar.set_description(f"PSNR: {total_psnr_value / (i + 1):.2f} "
                                         f"SSIM: {total_ssim_value / (i + 1):.4f} "
                                         f"LPIPS: {total_lpips_value / (i + 1):.4f} "
                                         f"GMSD: {total_gmsd_value / (i + 1):.4f} "
                                         f"LR: {total_lr_mass / (i + 1):.4f} "
                                         f"HR: {total_hr_mass / (i + 1):.4f} "
                                         f"GT: {total_gt_mass / (i + 1):.4f} "
                                         f"HS: {total_hs_mass:.4f}")

    out = total_psnr_value / total, total_ssim_value / total, total_lpips_value / total, total_gmsd_value / total, total_lr_mass/total, total_hr_mass/total, total_gt_mass/total, total_hs_mass

    return out
