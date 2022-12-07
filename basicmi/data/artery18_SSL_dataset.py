import os
import glob
import math

import torch
from monai import data, transforms

from basicmi.utils.registry import DATASET_REGISTRY
from basicmi.utils import get_root_logger


@DATASET_REGISTRY.register()
class Artery18SSLTrainDataset(torch.utils.data.Dataset):

    def __init__(self, opt):
        super(Artery18SSLTrainDataset, self).__init__()
        self.opt = opt
        data_dir = self.opt["dataroot"]
        u_data_dir = self.opt["udataroot"]

        logger = get_root_logger()

        self.images = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))
        labels = sorted(glob.glob(os.path.join(data_dir, "labels", "*.nii.gz")))
        l_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(self.images, labels)]
        logger.info(f'labeled data: {len(l_files)}')
        
        u_images = sorted(glob.glob(os.path.join(u_data_dir, "images", "*.nii.gz")))
        u_labels = sorted(glob.glob(os.path.join(u_data_dir, "labels", "*.nii.gz")))
        u_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(u_images, u_labels)]
        logger.info(f'unlabeled data: {len(u_files)}')

        self.files = l_files * math.ceil(len(u_files) / len(l_files)) + u_files

        transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(
                    keys=["image", "label"], pixdim=opt["space"], mode=("bilinear", "nearest")
                ),
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=opt["a_min"], a_max=opt["a_max"], b_min=opt["b_min"], b_max=opt["b_max"], clip=True
                ),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.EnsureTyped(keys=["image", "label"], device='cuda', track_meta= False),
                transforms.RandRotated(
                    keys=["image", "label"],
                    range_x=opt['rotate_range'],
                    range_y=opt['rotate_range'],
                    range_z=opt['rotate_range'],
                    prob=opt["rotate_prob"],
                    mode=["bilinear", "nearest"], 
                    padding_mode="zeros",
                ),
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=opt["spatial_size"],
                    pos=1,
                    neg=1,
                    num_samples=2,
                    image_key="image",
                    image_threshold=0,
                    allow_smaller=True,
                ),
                transforms.ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=opt["spatial_size"]),
                transforms.RandFlipd(keys=["image", "label"], prob=opt["RandFlipd_prob"], spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"], prob=opt["RandFlipd_prob"], spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"], prob=opt["RandFlipd_prob"], spatial_axis=2),
                transforms.RandRotate90d(keys=["image", "label"], prob=opt["RandRotate90d_prob"], max_k=3),

            ]
        )

        self.u_transform = transforms.Compose(
            [
                transforms.RandScaleIntensityd(keys=["image"], factors=0.2, prob=opt["RandScaleIntensityd_prob"]),
                transforms.RandShiftIntensityd(keys=["image"], offsets=0.2, prob=opt["RandShiftIntensityd_prob"]),
                transforms.Rand3DElasticd(
                    keys=["image", "label"],
                    sigma_range=(0.,0.),
                    magnitude_range=(0.,0.),
                    prob=opt["elastic_prob"],
                    shear_range=opt["shear_range"],
                    scale_range=0.1,
                    mode=["bilinear", "nearest"],
                    padding_mode="zeros",
                )
            ]
        )

        if opt["dataset_type"] == 'normal':
            self.dataset = data.Dataset(
                data=self.files, transform=transform
            )
        elif opt["dataset_type"] == 'cache':
            self.dataset = data.CacheDataset(
                data=self.files, transform=transform, cache_num=max(16, len(self.files)), cache_rate=1.0, num_workers=opt["workers"]
            )
        elif opt["dataset_type"] == 'smart':
            self.dataset = data.SmartCacheDataset(
                data=self.files, transform=transform, replace_rate=1.0, cache_num=32
            )
        elif opt["dataset_type"] == 'persist':
            self.dataset = data.PersistentDataset(
                data=self.files, transform=transform, cache_dir='experiments/dataset/.train_cache'
            )

    def __getitem__(self, index):
        if self.files[index]["image"] in self.images:
            return self.dataset[index]
        else:
            return self.u_transform(self.dataset[index])

    def __len__(self):
        # return len(self.files)
        return 32
