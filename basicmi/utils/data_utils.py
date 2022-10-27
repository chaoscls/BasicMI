# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os

import numpy as np
import torch
import glob

from monai import data, transforms
from basicmi.utils import get_root_logger

class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_loader(opt):
    logger = get_root_logger()
    data_dir = opt["data_dir"]
    train_images = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "labels", "*.nii.gz")))
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
    train_files, val_files = data_dicts[:-opt["val_num"]], data_dicts[-opt["val_num"]:]
    logger.info(f"training data number: {len(train_files)}")
    logger.info(f"validation data number: {len(val_files)}")

    train_opt = opt["train"]
    val_opt = opt["val"]
    train_transform = transforms.Compose(
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
                num_samples=4,
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
    val_transform = transforms.Compose(
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
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if not opt["is_train"]:
        pass
        # test_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        # test_ds = data.Dataset(data=test_files, transform=test_transform)
        # test_sampler = Sampler(test_ds, shuffle=False) if opt["distributed"] else None
        # test_loader = data.DataLoader(
        #     test_ds,
        #     batch_size=1,
        #     shuffle=False,
        #     num_workers=opt["workers"],
        #     sampler=test_sampler,
        #     pin_memory=True,
        #     persistent_workers=True,
        # )
        # loader = test_loader
    else:
        if opt["use_normal_dataset"]:
            train_ds = data.Dataset(data=train_files*opt["repeat_num"], transform=train_transform)
        else:
            train_ds = data.CacheDataset(
                data=train_files*opt["repeat_num"], transform=train_transform, cache_num=max(16, len(train_files)*opt["repeat_num"]), cache_rate=1.0, num_workers=opt["workers"]
            )
        train_sampler = Sampler(train_ds) if opt["distributed"] else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=opt["batch_size"],
            shuffle=(train_sampler is None),
            num_workers=opt["workers"],
            sampler=train_sampler,
            pin_memory=opt["pin_memory"],
        )

        if opt["use_normal_dataset"]:
            train_ds2 = data.Dataset(data=train_files, transform=val_transform)
        else:
            train_ds2 = data.CacheDataset(data=train_files, transform=val_transform, cache_num=max(16, len(train_files)), cache_rate=1.0, num_workers=opt["workers"])
        train_sampler2 = Sampler(train_ds2, shuffle=False) if opt["distributed"] else None
        train_loader2 = data.DataLoader(
            train_ds2, batch_size=1, shuffle=False, num_workers=opt["workers"], sampler=train_sampler2, pin_memory=opt["pin_memory"]
        )

        val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if opt["distributed"] else None
        val_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=opt["workers"], sampler=val_sampler, pin_memory=opt["pin_memory"]
        )
        loader = [train_loader, train_loader2, val_loader]

    return loader
