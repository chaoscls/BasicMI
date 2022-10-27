import os
import glob

import torch
from monai import data, transforms

from basicmi.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class Artery18TrainDataset(torch.utils.data.Dataset):

    def __init__(self, opt):
        super(Artery18TrainDataset, self).__init__()
        self.opt = opt
        data_dir = self.opt["dataroot"]
        images = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))
        labels = sorted(glob.glob(os.path.join(data_dir, "labels", "*.nii.gz")))
        files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(images, labels)]

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
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(opt["roi_x"], opt["roi_y"], opt["roi_z"]),
                    pos=1,
                    neg=1,
                    num_samples=2,
                    image_key="image",
                    image_threshold=0,
                ),
                transforms.RandFlipd(keys=["image", "label"], prob=opt["RandFlipd_prob"], spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"], prob=opt["RandFlipd_prob"], spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"], prob=opt["RandFlipd_prob"], spatial_axis=2),
                transforms.RandRotate90d(keys=["image", "label"], prob=opt["RandRotate90d_prob"], max_k=3),
                transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=opt["RandScaleIntensityd_prob"]),
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=opt["RandShiftIntensityd_prob"]),
                transforms.ToTensord(keys=["image", "label"]),
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
