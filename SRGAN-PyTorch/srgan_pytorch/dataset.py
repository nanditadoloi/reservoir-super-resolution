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
import os
import random
import numpy as np
import torch.utils.data.dataset
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import InterpolationMode

__all__ = [
    "check_image_file",
    "BaseTrainDataset", "BaseTestDataset",
    "CustomTrainDataset", "CustomTestDataset"
]


def check_image_file(filename: str):
    r"""Filter non image files in directory.

    Args:
        filename (str): File name under path.

    Returns:
        Return True if bool(x) is True for any x in the iterable.
    """
    return any(filename.endswith(extension) for extension in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".BMP", ".npy"])


class BaseTrainDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, root: str, image_size: int = 96, upscale_factor: int = 4):
        r"""
        Args:
            root (str): The directory address where the data image is stored.
            image_size (optional, int): The size of image block is randomly cut out from the original image. (Default: 96)
            upscale_factor (optional, int): Image magnification. (Default: 4)
        """
        super(BaseTrainDataset, self).__init__()
        self.filenames = [os.path.join(root, x) for x in os.listdir(root) if check_image_file(x)]

        self.lr_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size // upscale_factor, image_size // upscale_factor), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
        self.hr_transforms = transforms.Compose([
            transforms.RandomCrop((image_size, image_size)),
            transforms.AutoAugment(),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        r""" Get image source file.

        Args:
            index (int): Index position in image list.

        Returns:
            Low resolution image, high resolution image.
        """
        hr = self.hr_transforms(Image.open(self.filenames[index]).convert("RGB"))
        lr = self.lr_transforms(hr)

        return lr, hr

    def __len__(self):
        return len(self.filenames)


class BaseTestDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, root: str, image_size: int = 96, upscale_factor: int = 4):
        r"""
        Args:
            root (str): The directory address where the data image is stored.
            image_size (optional, int): The size of image block is randomly cut out from the original image. (Default: 96)
            upscale_factor (optional, int): Image magnification. (Default: 4)
        """
        super(BaseTestDataset, self).__init__()
        self.filenames = [os.path.join(root, x) for x in os.listdir(root) if check_image_file(x)]

        self.lr_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size // upscale_factor, image_size // upscale_factor), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
        self.bicubic_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
        self.hr_transforms = transforms.Compose([
            transforms.RandomCrop((image_size, image_size)),
            transforms.AutoAugment(),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        r""" Get image source file.

        Args:
            index (int): Index position in image list.

        Returns:
            Low resolution image, high resolution image.
        """
        hr = self.hr_transforms(Image.open(self.filenames[index]).convert("RGB"))
        lr = self.lr_transforms(hr)
        bicubic = self.bicubic_transforms(lr)

        return lr, bicubic, hr

    def __len__(self):
        return len(self.filenames)


class CustomTrainDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, root: str, sampler_frequency: int = 1):
        r"""
        Args:
            root (str): The directory address where the data image is stored.
            sampler_frequency (int): If there are many datasets, this method can be used to increase the number of epochs. (Default: 1)
        """
        super(CustomTrainDataset, self).__init__()
        lr_dir = os.path.join(root, "input")
        hr_dir = os.path.join(root, "target")
        self.filenames = os.listdir(lr_dir)
        self.sampler_filenames = random.sample(self.filenames, len(self.filenames) // sampler_frequency)
        self.lr_filenames = [os.path.join(lr_dir, x) for x in self.sampler_filenames if check_image_file(x)]
        self.hr_filenames = [os.path.join(hr_dir, x) for x in self.sampler_filenames if check_image_file(x)]

        self.transforms = transforms.ToTensor()

    def __getitem__(self, index):
        r""" Get image source file.

        Args:
            index (int): Index position in image list.

        Returns:
            Low resolution image, high resolution image.
        """
        lr = self.transforms(Image.open(self.lr_filenames[index]))
        hr = self.transforms(Image.open(self.hr_filenames[index]))
        return lr, hr

    def __len__(self):
        return len(self.sampler_filenames)


class CustomTestDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, root: str, image_size: int = 256, sampler_frequency: int = 1):
        r"""
        Args:
            root (str): The directory address where the data image is stored.
            image_size (optional, int): The size of image block is randomly cut out from the original image. (Default: 256)
            sampler_frequency (list): If there are many datasets, this method can be used to increase the number of epochs. (Default: 1)
        """
        super(CustomTestDataset, self).__init__()
        lr_dir = os.path.join(root, "input")
        hr_dir = os.path.join(root, "target")
        self.filenames = os.listdir(lr_dir)
        self.sampler_filenames = random.sample(self.filenames, len(self.filenames) // sampler_frequency)
        self.lr_filenames = [os.path.join(lr_dir, x) for x in self.sampler_filenames if check_image_file(x)]
        self.hr_filenames = [os.path.join(hr_dir, x) for x in self.sampler_filenames if check_image_file(x)]

        self.transforms = transforms.ToTensor()
        self.bicubic_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        r""" Get image source file.

        Args:
            index (int): Index position in image list.

        Returns:
            Low resolution image, high resolution image.
        """
        lr = self.transforms(Image.open(self.lr_filenames[index]))
        bicubic = self.bicubic_transforms(lr)
        hr = self.transforms(Image.open(self.hr_filenames[index]))

        return lr, bicubic, hr

class CustomTrainDataset_np(torch.utils.data.dataset.Dataset):
    def __init__(self, root: str, sampler_frequency: int = 1):
        r"""
        Args:
            root (str): The directory address where the data image is stored.
            sampler_frequency (int): If there are many datasets, this method can be used to increase the number of epochs. (Default: 1)
        """
        super(CustomTrainDataset_np, self).__init__()
        lr_dir = os.path.join(root, "input")
        hr_dir = os.path.join(root, "target")
        self.filenames = os.listdir(lr_dir)
        self.sampler_filenames = random.sample(self.filenames, len(self.filenames) // sampler_frequency)
        self.lr_filenames = [os.path.join(lr_dir, x) for x in self.sampler_filenames if check_image_file(x)]
        self.hr_filenames = [os.path.join(hr_dir, x) for x in self.sampler_filenames if check_image_file(x)]

        self.transforms = transforms.ToTensor()

    def __getitem__(self, index):
        r""" Get image source file.

        Args:
            index (int): Index position in image list.

        Returns:
            Low resolution image, high resolution image.
        """
        lr_1ch = (np.load(self.lr_filenames[index])*255).astype(np.uint8)
        lr_3ch = np.expand_dims(lr_1ch[0,:,:],axis=2).repeat(3,axis=2)
        hr_1ch = (np.load(self.hr_filenames[index])*255).astype(np.uint8)
        hr_3ch = np.expand_dims(hr_1ch[0,:,:],axis=2).repeat(3,axis=2)
        lr = self.transforms(lr_3ch)
        hr = self.transforms(hr_3ch)
        return lr, hr

    def __len__(self):
        return len(self.sampler_filenames)


class CustomTestDataset_np(torch.utils.data.dataset.Dataset):
    def __init__(self, root: str, image_size: int = 256, sampler_frequency: int = 1):
        r"""
        Args:
            root (str): The directory address where the data image is stored.
            image_size (optional, int): The size of image block is randomly cut out from the original image. (Default: 256)
            sampler_frequency (list): If there are many datasets, this method can be used to increase the number of epochs. (Default: 1)
        """
        super(CustomTestDataset_np, self).__init__()
        lr_dir = os.path.join(root, "input")
        hr_dir = os.path.join(root, "target")
        self.filenames = os.listdir(lr_dir)
        self.sampler_filenames = random.sample(self.filenames, len(self.filenames) // sampler_frequency)
        self.lr_filenames = [os.path.join(lr_dir, x) for x in self.sampler_filenames if check_image_file(x)]
        self.hr_filenames = [os.path.join(hr_dir, x) for x in self.sampler_filenames if check_image_file(x)]

        self.transforms = transforms.ToTensor()
        self.bicubic_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        r""" Get image source file.

        Args:
            index (int): Index position in image list.

        Returns:
            Low resolution image, high resolution image.
        """
        lr_1ch = (np.load(self.lr_filenames[index])*255).astype(np.uint8)
        lr_3ch = np.expand_dims(lr_1ch[0,:,:],axis=2).repeat(3,axis=2)
        hr_1ch = (np.load(self.hr_filenames[index])*255).astype(np.uint8)
        hr_3ch = np.expand_dims(hr_1ch[0,:,:],axis=2).repeat(3,axis=2)
        lr = self.transforms(lr_3ch)
        hr = self.transforms(hr_3ch)
        bicubic = self.bicubic_transforms(lr)

        return lr, bicubic, hr

    def __len__(self):
        return len(self.sampler_filenames)

class CustomTrainDataset_np_crop(torch.utils.data.dataset.Dataset):
    def __init__(self, root: str, sampler_frequency: int = 1):
        r"""
        Args:
            root (str): The directory address where the data image is stored.
            sampler_frequency (int): If there are many datasets, this method can be used to increase the number of epochs. (Default: 1)
        """
        super(CustomTrainDataset_np_crop, self).__init__()
        lr_dir = os.path.join(root, "input")
        hr_dir = os.path.join(root, "target")
        lr_BPR_dir = os.path.join(root, "input_BPR")
        hr_BPR_dir = os.path.join(root, "target_BPR")
        self.filenames = os.listdir(lr_dir)
        self.sampler_filenames = random.sample(self.filenames, len(self.filenames) // sampler_frequency)
        self.lr_filenames = [os.path.join(lr_dir, x) for x in self.sampler_filenames if check_image_file(x)]
        self.hr_filenames = [os.path.join(hr_dir, x) for x in self.sampler_filenames if check_image_file(x)]
        self.lr_BPR_filenames = [os.path.join(lr_BPR_dir, x) for x in self.sampler_filenames if check_image_file(x)]
        self.hr_BPR_filenames = [os.path.join(hr_BPR_dir, x) for x in self.sampler_filenames if check_image_file(x)]

        self.transforms = transforms.ToTensor()

    def __getitem__(self, index):
        r""" Get image source file.

        Args:
            index (int): Index position in image list.

        Returns:
            Low resolution image, high resolution image.
        """
        lr_1ch = (np.load(self.lr_filenames[index])).astype(np.single)
        lr_3ch = np.expand_dims(lr_1ch[0,:,:],axis=2).repeat(3,axis=2)
        lr_BPR_1ch = np.load(self.lr_BPR_filenames[index]).astype(np.single)
        lr_BPR_3ch = np.expand_dims(lr_BPR_1ch[0,:,:],axis=2).repeat(3,axis=2)
        hr_1ch = (np.load(self.hr_filenames[index])).astype(np.single)
        hr_3ch = np.expand_dims(hr_1ch[0,:,:],axis=2).repeat(3,axis=2)
        hr_BPR_1ch = (np.load(self.hr_BPR_filenames[index])).astype(np.single)
        hr_BPR_3ch = np.expand_dims(hr_BPR_1ch[0,:,:],axis=2).repeat(3,axis=2)

        idx = torch.randint(0,110-30,[1]).numpy()[0]
        sidx = int(idx/2)
        # lr = self.transforms(lr_3ch[:,sidx:sidx+15,:])
        # hr = self.transforms(hr_3ch[:,idx:idx+30,:])
        # lr_BPR = self.transforms(lr_BPR_3ch[:,sidx:sidx+15,:])
        # hr_BPR = self.transforms(hr_BPR_3ch[:,idx:idx+30,:])

        lr = self.transforms(lr_3ch) # Makes channels go to axis 0
        hr = self.transforms(hr_3ch)
        lr_BPR = self.transforms(lr_BPR_3ch)
        hr_BPR = self.transforms(hr_BPR_3ch)

        return lr, hr, lr_BPR, hr_BPR

    def __len__(self):
        return len(self.sampler_filenames)


class CustomTestDataset_np_crop(torch.utils.data.dataset.Dataset):
    def __init__(self, root: str, sampler_frequency: int = 1):
        r"""
        Args:
            root (str): The directory address where the data image is stored.
            image_size (optional, int): The size of image block is randomly cut out from the original image. (Default: 256)
            sampler_frequency (list): If there are many datasets, this method can be used to increase the number of epochs. (Default: 1)
        """
        super(CustomTestDataset_np_crop, self).__init__()
        lr_dir = os.path.join(root, "input")
        hr_dir = os.path.join(root, "target")
        lr_BPR_dir = os.path.join(root, "input_BPR")
        hr_BPR_dir = os.path.join(root, "target_BPR")
        self.filenames = os.listdir(lr_dir)
        self.sampler_filenames = random.sample(self.filenames, len(self.filenames) // sampler_frequency)
        self.lr_filenames = [os.path.join(lr_dir, x) for x in self.sampler_filenames if check_image_file(x)]
        self.hr_filenames = [os.path.join(hr_dir, x) for x in self.sampler_filenames if check_image_file(x)]
        self.lr_BPR_filenames = [os.path.join(lr_BPR_dir, x) for x in self.sampler_filenames if check_image_file(x)]
        self.hr_BPR_filenames = [os.path.join(hr_BPR_dir, x) for x in self.sampler_filenames if check_image_file(x)]

        self.transforms = transforms.ToTensor()
        self.bicubic_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((30, 30), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        r""" Get image source file.

        Args:
            index (int): Index position in image list.

        Returns:
            Low resolution image, high resolution image.
        """
        lr_1ch = (np.load(self.lr_filenames[index])).astype(np.single)
        lr_3ch = np.expand_dims(lr_1ch[0,:,:],axis=2).repeat(3,axis=2)
        lr_BPR_1ch = np.load(self.lr_BPR_filenames[index]).astype(np.single)
        lr_BPR_3ch = np.expand_dims(lr_BPR_1ch[0,:,:],axis=2).repeat(3,axis=2)
        hr_1ch = (np.load(self.hr_filenames[index])).astype(np.single)
        hr_3ch = np.expand_dims(hr_1ch[0,:,:],axis=2).repeat(3,axis=2)
        hr_BPR_1ch = (np.load(self.hr_BPR_filenames[index])).astype(np.single)
        hr_BPR_3ch = np.expand_dims(hr_BPR_1ch[0,:,:],axis=2).repeat(3,axis=2)

        idx = torch.randint(0,110-30,[1]).numpy()[0]
        sidx = int(idx/2)
        lr = self.transforms(lr_3ch[:,:,:])
        hr = self.transforms(hr_3ch[:,:,:])
        lr_BPR = self.transforms(lr_BPR_3ch[:,:,:])
        hr_BPR = self.transforms(hr_BPR_3ch[:,:,:])

        bicubic = self.bicubic_transforms(lr)

        return lr, bicubic, hr, lr_BPR, hr_BPR

    def __len__(self):
        return len(self.sampler_filenames)
    

class CustomTrainDataset_np_crop_tempo(torch.utils.data.dataset.Dataset):
    def __init__(self, root: str, sampler_frequency: int = 1):
        r"""
        Args:
            root (str): The directory address where the data image is stored.
            sampler_frequency (int): If there are many datasets, this method can be used to increase the number of epochs. (Default: 1)
        """
        super(CustomTrainDataset_np_crop_tempo, self).__init__()
        lr_dir_0 = os.path.join(root, "input_0")
        hr_dir_0 = os.path.join(root, "target_0")
        lr_BPR_dir_0 = os.path.join(root, "input_BPR_0")
        hr_BPR_dir_0 = os.path.join(root, "target_BPR_0")
        self.filenames_0 = os.listdir(lr_dir_0)
        self.sampler_filenames_0 = random.sample(self.filenames_0, len(self.filenames_0) // sampler_frequency)
        self.lr_filenames_0 = [os.path.join(lr_dir_0, x) for x in self.sampler_filenames_0 if check_image_file(x)]
        self.hr_filenames_0 = [os.path.join(hr_dir_0, x) for x in self.sampler_filenames_0 if check_image_file(x)]
        self.lr_BPR_filenames_0 = [os.path.join(lr_BPR_dir_0, x) for x in self.sampler_filenames_0 if check_image_file(x)]
        self.hr_BPR_filenames_0 = [os.path.join(hr_BPR_dir_0, x) for x in self.sampler_filenames_0 if check_image_file(x)]

        lr_dir_1 = os.path.join(root, "input_1")
        hr_dir_1 = os.path.join(root, "target_1")
        lr_BPR_dir_1 = os.path.join(root, "input_BPR_1")
        hr_BPR_dir_1 = os.path.join(root, "target_BPR_1")
        self.filenames_1 = os.listdir(lr_dir_1)
        self.sampler_filenames_1 = random.sample(self.filenames_1, len(self.filenames_1) // sampler_frequency)
        self.lr_filenames_1 = [os.path.join(lr_dir_1, x) for x in self.sampler_filenames_1 if check_image_file(x)]
        self.hr_filenames_1 = [os.path.join(hr_dir_1, x) for x in self.sampler_filenames_1 if check_image_file(x)]
        self.lr_BPR_filenames_1 = [os.path.join(lr_BPR_dir_1, x) for x in self.sampler_filenames_1 if check_image_file(x)]
        self.hr_BPR_filenames_1 = [os.path.join(hr_BPR_dir_1, x) for x in self.sampler_filenames_1 if check_image_file(x)]

        self.transforms = transforms.ToTensor()

    def __getitem__(self, index):
        r""" Get image source file.

        Args:
            index (int): Index position in image list.

        Returns:
            Low resolution image, high resolution image.
        """
        lr_1ch_0 = (np.load(self.lr_filenames_0[index])).astype(np.single)
        lr_BPR_1ch_0 = np.load(self.lr_BPR_filenames_0[index]).astype(np.single)
        lr_1ch_1 = (np.load(self.lr_filenames_1[index])).astype(np.single)
        lr_BPR_1ch_1 = np.load(self.lr_BPR_filenames_1[index]).astype(np.single)
        m0 = lr_1ch_0*lr_BPR_1ch_0
        m1 = lr_1ch_1*lr_BPR_1ch_1
        delm = 10**(-6)*(m1-m0)
        lr = np.array([lr_1ch_0[0,:,:], delm[0,:,:], lr_1ch_1[0,:,:]])
        lr = np.moveaxis(lr, 0, -1)
        lr_BPR = np.moveaxis(lr_1ch_1, 0, -1)

        hr_1ch = (np.load(self.hr_filenames_1[index])).astype(np.single)
        hr = np.moveaxis(hr_1ch, 0, -1)
        hr_1ch_0 = (np.load(self.hr_filenames_0[index])).astype(np.single)
        hr_0 = np.moveaxis(hr_1ch_0, 0, -1)
        hr_BPR_1ch = (np.load(self.hr_BPR_filenames_1[index])).astype(np.single)
        hr_BPR = np.moveaxis(hr_BPR_1ch, 0, -1)

        lr = self.transforms(lr)
        hr = self.transforms(hr)
        hr_0 = self.transforms(hr_0)
        lr_BPR = self.transforms(lr_BPR)
        hr_BPR = self.transforms(hr_BPR)

        return lr, hr, lr_BPR, hr_BPR, hr_0

    def __len__(self):
        return len(self.sampler_filenames_0)

class CustomTestDataset_np_crop_tempo(torch.utils.data.dataset.Dataset):
    def __init__(self, root: str, sampler_frequency: int = 1):
        r"""
        Args:
            root (str): The directory address where the data image is stored.
            image_size (optional, int): The size of image block is randomly cut out from the original image. (Default: 256)
            sampler_frequency (list): If there are many datasets, this method can be used to increase the number of epochs. (Default: 1)
        """
        super(CustomTestDataset_np_crop_tempo, self).__init__()
        lr_dir_0 = os.path.join(root, "input_0")
        hr_dir_0 = os.path.join(root, "target_0")
        lr_BPR_dir_0 = os.path.join(root, "input_BPR_0")
        hr_BPR_dir_0 = os.path.join(root, "target_BPR_0")
        self.filenames_0 = os.listdir(lr_dir_0)
        self.sampler_filenames_0 = random.sample(self.filenames_0, len(self.filenames_0) // sampler_frequency)
        self.lr_filenames_0 = [os.path.join(lr_dir_0, x) for x in self.sampler_filenames_0 if check_image_file(x)]
        self.hr_filenames_0 = [os.path.join(hr_dir_0, x) for x in self.sampler_filenames_0 if check_image_file(x)]
        self.lr_BPR_filenames_0 = [os.path.join(lr_BPR_dir_0, x) for x in self.sampler_filenames_0 if check_image_file(x)]
        self.hr_BPR_filenames_0 = [os.path.join(hr_BPR_dir_0, x) for x in self.sampler_filenames_0 if check_image_file(x)]

        lr_dir_1 = os.path.join(root, "input_1")
        hr_dir_1 = os.path.join(root, "target_1")
        lr_BPR_dir_1 = os.path.join(root, "input_BPR_1")
        hr_BPR_dir_1 = os.path.join(root, "target_BPR_1")
        self.filenames_1 = os.listdir(lr_dir_1)
        self.sampler_filenames_1 = random.sample(self.filenames_1, len(self.filenames_1) // sampler_frequency)
        self.lr_filenames_1 = [os.path.join(lr_dir_1, x) for x in self.sampler_filenames_1 if check_image_file(x)]
        self.hr_filenames_1 = [os.path.join(hr_dir_1, x) for x in self.sampler_filenames_1 if check_image_file(x)]
        self.lr_BPR_filenames_1 = [os.path.join(lr_BPR_dir_1, x) for x in self.sampler_filenames_1 if check_image_file(x)]
        self.hr_BPR_filenames_1 = [os.path.join(hr_BPR_dir_1, x) for x in self.sampler_filenames_1 if check_image_file(x)]

        self.transforms = transforms.ToTensor()
        self.bicubic_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((30, 30), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        r""" Get image source file.

        Args:
            index (int): Index position in image list.

        Returns:
            Low resolution image, high resolution image.
        """
        lr_1ch_0 = (np.load(self.lr_filenames_0[index])).astype(np.single)
        lr_BPR_1ch_0 = np.load(self.lr_BPR_filenames_0[index]).astype(np.single)
        lr_1ch_1 = (np.load(self.lr_filenames_1[index])).astype(np.single)
        lr_BPR_1ch_1 = np.load(self.lr_BPR_filenames_1[index]).astype(np.single)
        m0 = lr_1ch_0*lr_BPR_1ch_0
        m1 = lr_1ch_1*lr_BPR_1ch_1
        delm = 10**(-6)*(m1-m0)
        lr = np.array([lr_1ch_0[0,:,:], delm[0,:,:], lr_1ch_1[0,:,:]])
        lr = np.moveaxis(lr, 0, -1)
        lr_BPR = np.moveaxis(lr_1ch_1, 0, -1)

        hr_1ch = (np.load(self.hr_filenames_1[index])).astype(np.single)
        hr = np.moveaxis(hr_1ch, 0, -1)
        hr_1ch_0 = (np.load(self.hr_filenames_0[index])).astype(np.single)
        hr_0 = np.moveaxis(hr_1ch_0, 0, -1)
        hr_BPR_1ch = (np.load(self.hr_BPR_filenames_1[index])).astype(np.single)
        hr_BPR = np.moveaxis(hr_BPR_1ch, 0, -1)

        lr = self.transforms(lr)
        hr = self.transforms(hr)
        hr_0 = self.transforms(hr_0)
        lr_BPR = self.transforms(lr_BPR)
        hr_BPR = self.transforms(hr_BPR)

        bicubic = self.bicubic_transforms(lr)

        return lr, bicubic, hr, lr_BPR, hr_BPR, hr_0

    def __len__(self):
        return len(self.sampler_filenames_0)
