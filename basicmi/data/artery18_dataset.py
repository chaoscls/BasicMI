import os
import glob

import torch
import numpy as np
from monai import data, transforms

from basicmi.utils.registry import DATASET_REGISTRY

class RandCropWithCenterCrop(transforms.Randomizable, transforms.MapTransform):
    def __init__(
        self,
        keys,
        label_key,
        spatial_size,
        pos,
        neg,
        num_samples,
        image_key,
        image_threshold,
        allow_smaller,
        ) -> None:
        super(transforms.Randomizable, self).__init__(keys)
        self.randcrop = transforms.RandCropByPosNegLabeld(keys=keys, label_key=label_key, spatial_size=spatial_size, pos=pos, neg=neg, num_samples=num_samples, image_key=image_key, image_threshold=image_threshold, allow_smaller=allow_smaller)
        self.centercrop = transforms.CenterSpatialCropd(keys=keys, roi_size=spatial_size)
        
    def __call__(self, data):
        randcrop = self.randcrop(data)
        center = self.centercrop(data)
        for i in range(len(randcrop)):
            randcrop[i]['center_image'] = center['image']
            # randcrop[i]['center_label'] = center['label']
        return randcrop

class AddCenterCrop(transforms.MapTransform):
    def __init__(
        self,
        keys,
        spatial_size,
        ) -> None:
        super(transforms.Randomizable, self).__init__(keys)
        self.centercrop = transforms.CenterSpatialCropd(keys=keys, roi_size=spatial_size)
        
    def __call__(self, data):
        center = self.centercrop(data)
        data["center_image"] = center
        return data
        
        
@DATASET_REGISTRY.register()
class Artery18TrainDataset(torch.utils.data.Dataset):

    def __init__(self, opt):
        super(Artery18TrainDataset, self).__init__()
        self.opt = opt
        data_dir = self.opt["dataroot"]
        images = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))
        labels = sorted(glob.glob(os.path.join(data_dir, "labels", "*.nii.gz")))
        files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(images, labels)]

        keys_tmp = ["image", "label", "center_image"] if opt["center_crop"] else ["image", "label"]

        transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(
                    keys=["image", "label"], pixdim=(opt["space_x"], opt["space_y"], opt["space_z"]), mode=("bilinear", "nearest")
                ),
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=opt["a_min"], a_max=opt["a_max"], b_min=opt["b_min"], b_max=opt["b_max"], clip=True
                ),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                # transforms.ToDeviced(keys=["image", "label"], device='cuda'),
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
                # transforms.Rand3DElasticd(
                #     keys=["image", "label"],
                #     sigma_range=(0.,0.),
                #     magnitude_range=(0.,0.),
                #     prob=opt["elastic_prob"],
                #     rotate_range=opt["rotate_range"],
                #     shear_range=opt["shear_range"],
                #     scale_range=0.1,
                #     mode=["bilinear", "nearest"],
                #     padding_mode="zeros",
                #     # device="cuda",
                # ),

                RandCropWithCenterCrop(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(opt["roi_x"], opt["roi_y"], opt["roi_z"]),
                    pos=1,
                    neg=1,
                    num_samples=opt['num_samples'],
                    image_key="image",
                    image_threshold=0,
                    allow_smaller=True, 
                ) 
                if opt["center_crop"] else
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(opt["roi_x"], opt["roi_y"], opt["roi_z"]),
                    pos=1,
                    neg=1,
                    num_samples=opt['num_samples'],
                    image_key="image",
                    image_threshold=0,
                    allow_smaller=True,
                ),

                transforms.ResizeWithPadOrCropd(keys=keys_tmp, spatial_size=(opt["roi_x"], opt["roi_y"], opt["roi_z"])),
                transforms.RandFlipd(keys=keys_tmp, prob=opt["RandFlipd_prob"], spatial_axis=0),
                transforms.RandFlipd(keys=keys_tmp, prob=opt["RandFlipd_prob"], spatial_axis=1),
                transforms.RandFlipd(keys=keys_tmp, prob=opt["RandFlipd_prob"], spatial_axis=2),
                # transforms.RandRotate90d(keys=keys_tmp, prob=opt["RandRotate90d_prob"], max_k=3),
                transforms.ToTensord(keys=keys_tmp),
            ]
        )

        if opt["dataset_type"] == 'normal':
            self.dataset = data.Dataset(data=files, transform=transform)
        elif opt["dataset_type"] == 'cache':
            self.dataset = data.CacheDataset(
                data=files, transform=transform, cache_num=max(16, len(files)), cache_rate=1.0, num_workers=opt["workers"]
            )
        elif opt["dataset_type"] == 'persist':
            self.dataset = data.PersistentDataset(data=files, transform=transform, cache_dir='experiments/dataset/.train_cache')

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


@DATASET_REGISTRY.register()
class Artery18ValidationDataset(torch.utils.data.Dataset):

    def __init__(self, opt):
        super(Artery18ValidationDataset, self).__init__()
        self.opt = opt
        data_dir = self.opt["dataroot"]
        images = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))
        labels = sorted(glob.glob(os.path.join(data_dir, "labels", "*.nii.gz")))
        valid_num = opt.get('valid_num', len(images))
        images, labels = images[:valid_num], labels[:valid_num]
        files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(images, labels)]

        self.transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.EnsureChannelFirstd(keys=["image", "label"]),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(
                    keys=["image", "label"], pixdim=(opt["space_x"], opt["space_y"], opt["space_z"]), mode=("bilinear", "nearest")
                ),
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=opt["a_min"], a_max=opt["a_max"], b_min=opt["b_min"], b_max=opt["b_max"], clip=True
                ),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                # transforms.ToTensord(keys=["image", "label"]),
                # transforms.EnsureTyped(keys=["image", "label"], device='cuda', track_meta= False),
            ]
        )

        if opt["dataset_type"] == 'normal':
            self.dataset = data.Dataset(data=files, transform=self.transform)
        elif opt["dataset_type"] == 'cache':
            self.dataset = data.CacheDataset(
                data=files, transform=self.transform, cache_num=max(16, len(files)), cache_rate=opt["cache_rate"], num_workers=opt["workers"]
            )
        elif opt["dataset_type"] == 'persist':
            self.dataset = data.PersistentDataset(data=files, transform=self.transform, cache_dir='experiments/dataset/.val_cache')

    def get_post_transform(self):
        post_transform = transforms.Compose([
                transforms.Invertd(
                    keys="pred",
                    transform= self.transform,
                    orig_keys='image',
                    orig_meta_keys="image_meta_dict",
                    meta_key_postfix="meta_dict",
                    nearest_interp=False,
                    to_tensor=True,
                ),
                transforms.AsDiscreted(keys="pred", argmax=True, to_onehot=19),
                # transforms.AsDiscreted(keys="label", to_onehot=19),
                transforms.SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir="./out", output_postfix="seg", resample=False),
            ])
        return post_transform

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
