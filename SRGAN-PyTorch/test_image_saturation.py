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
#
# python test_image_saturation.py --lr same_geo_data/test/input/im6.npy --hr same_geo_data/test/target/im6.npy --lr_C same_geo_data/test/input_BPR/im6.npy --hr_C same_geo_data/test/target_BPR/im6.npy -a srgan_2x2 --upscale-factor 2 --model-path weights/PSNR.pth --pretrained --gpu 0
# python test_image_saturation.py --lr same_geo_data/test/input/im18.npy --hr same_geo_data/test/target/im18.npy --lr_C same_geo_data/test/input_BPR/im18.npy --hr_C same_geo_data/test/target_BPR/im18.npy -a srgan_2x2 --upscale-factor 2 --model-path weights_backup/SRGAN_2x2_DIV2K-9ec9dd11.pth --pretrained --gpu 0 --resume_psnr weights_backup/PSNR_samegeo_mtc_epoch5115.pth
#
# ==============================================================================
import argparse
import logging
import os
import random
import warnings
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from torchvision.transforms import InterpolationMode

import srgan_pytorch.models as models
from srgan_pytorch.utils.common import configure
from srgan_pytorch.utils.common import create_folder
from srgan_pytorch.utils.estimate import iqa
from srgan_pytorch.utils.transform import process_image

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)

parser = argparse.ArgumentParser("Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network")
parser.add_argument("--lr", type=str, required=True,
                    help="Test low resolution image name.")
parser.add_argument("--hr", type=str,
                    help="Raw high resolution image name.")
parser.add_argument("--lr_C", type=str,
                    help="Test low resolution image name.")
parser.add_argument("--hr_C", type=str,
                    help="Raw high resolution image name.")
parser.add_argument("-a", "--arch", metavar="ARCH", default="srgan",
                    choices=model_names,
                    help="Model architecture: " +
                         " | ".join(model_names) +
                         ". (Default: srgan)")
parser.add_argument("--upscale-factor", type=int, default=4, choices=[2, 4, 8],
                    help="Low to high resolution scaling factor. Optional: [2, 4, 8]. (Default: 4)")
parser.add_argument("--model-path", default="./weights/GAN.pth", type=str, metavar="PATH",
                    help="Path to latest checkpoint for model. (Default: `./weights/GAN.pth`)")
parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                    help="Use pre-trained model.")
parser.add_argument("--seed", default=666, type=int,
                    help="Seed for initializing training. (Default: 666)")
parser.add_argument("--gpu", default=None, type=int,
                    help="GPU id to use.")
parser.add_argument("--resume_psnr", default="", type=str, metavar="PATH",
                    help="Path to latest psnr-oral checkpoint.")


def main():
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True

    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu

    if args.gpu is not None:
        logger.info(f"Use GPU: {args.gpu} for testing.")

    model = configure(args)

    if args.resume_psnr:
        if os.path.isfile(args.resume_psnr):
            logger.info(f"Loading checkpoint '{args.resume_psnr}'.")
            if args.gpu is None:
                checkpoint = torch.load(args.resume_psnr)
            else:
                # Map model to be loaded to specified single gpu.
                checkpoint = torch.load(args.resume_psnr, map_location=f"cuda:{args.gpu}")
            args.start_psnr_epoch = checkpoint["epoch"]
            best_psnr = checkpoint["best_psnr"]
            if args.gpu is not None:
                # best_psnr may be from a checkpoint from a different GPU
                best_psnr = best_psnr.to(args.gpu)
            model.load_state_dict(checkpoint["state_dict"])

    if not torch.cuda.is_available():
        logger.warning("Using CPU, this will be slow.")
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # Set eval mode.
    model.eval()

    cudnn.benchmark = True

    # Get image filename.
    filename = os.path.basename(args.lr)

    # Read all pictures.

    lr_1ch = (np.load(args.lr)).astype(np.single)
    lr = np.expand_dims(lr_1ch[0,:,:],axis=2).repeat(3,axis=2)
    

    #bicubic = transforms.Resize((lr.size[1] * args.upscale_factor, lr.size[0] * args.upscale_factor), InterpolationMode.BICUBIC)(lr)
    lr = process_image(lr, args.gpu)
    #bicubic = process_image(bicubic, args.gpu)

    with torch.no_grad():
        sr = model(lr)

    if args.hr:
        lr_s = np.load(args.lr)[0,:,:]
        lr_C = np.load(args.lr_C)[0,:,:]
        hr_s = np.load(args.hr)[0,:,:]
        hr_C = np.load(args.hr_C)[0,:,:]
        sr_s = (sr.cpu().numpy().astype(float))[0,0,:,:]
        mass_lr = np.sum(lr_s*lr_C)
        mass_hr = np.sum(hr_s*hr_C)
        mass_sr = np.sum(sr_s*hr_C)
        mass_error_lr = (mass_sr-mass_lr)*100.0/mass_lr
        mass_error_hr = (mass_sr-mass_hr)*100.0/mass_hr
        mass_error_gt = (mass_hr-mass_lr)*100.0/mass_lr

        hr_1ch = (np.load(args.hr)).astype(np.single)
        hr_3ch = np.expand_dims(hr_1ch[0,:,:],axis=2).repeat(3,axis=2)
        hr = process_image(hr_3ch, args.gpu)
        vutils.save_image(hr, os.path.join("tests", f"hr_{filename}.png"), value_range=(0,1), normalize=True)
        #images = torch.cat([bicubic, sr, hr], dim=-1)
        images = torch.cat([ sr, hr], dim=-1)

        value = iqa(sr, hr, args.gpu)
        print(f"Performance avg results:\n")
        print(f"indicator Score\n")
        print(f"--------- -----\n")
        print(f"MSE       {value[0]:6.4f}\n"
              f"RMSE      {value[1]:6.4f}\n"
              f"PSNR      {value[2]:6.2f}\n"
              f"SSIM      {value[3]:6.4f}\n"
              f"LPIPS     {value[4]:6.4f}\n"
              f"GMSD      {value[5]:6.4f}\n"
              f"Mass LR   {mass_error_lr:6.4f}\n"
              f"Mass HR   {mass_error_hr:6.4f}\n"
              f"Mass GT   {mass_error_gt:6.4f}\n")
    else:
        # images = torch.cat([bicubic, sr], dim=-1)
        images = torch.cat([sr], dim=-1)

    vutils.save_image(lr, os.path.join("tests", f"lr_{filename}.png"), value_range=(0,1), normalize=True)
    # vutils.save_image(bicubic, os.path.join("tests", f"bicubic_{filename}"))
    vutils.save_image(sr, os.path.join("tests", f"sr_{filename}.png"), value_range=(0,1), normalize=True)
    vutils.save_image(images, os.path.join("tests", f"compare_{filename}.png"), padding=10, value_range=(0,1), normalize=True)


if __name__ == "__main__":
    print("##################################################\n")
    print("Run Testing Engine.\n")

    create_folder("tests")

    logger.info("TestingEngine:")
    print("\tAPI version .......... 0.2.1")
    print("\tBuild ................ 2021.04.09")
    print("##################################################\n")
    main()

    logger.info("Test single image performance evaluation completed successfully.\n")
