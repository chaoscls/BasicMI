import torch
import os.path as osp
import os
import glob
import argparse

from monai.transforms import (
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
)
from monai.networks.nets import UNet
from monai.data import DataLoader, Dataset, decollate_batch
from monai.metrics import DiceMetric
from monai.handlers.utils import from_engine

from basicmi.inferers.utils import sliding_window_inference

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='experiments/dataset/18artery/train', type=str)
parser.add_argument('--load_path', default='experiments/unet_baseline_artery_192/models/net_latest.pth', type=str)
args = parser.parse_args()

data_dir = args.data_dir
load_path = args.load_path
images = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))
labels = sorted(glob.glob(os.path.join(data_dir, "labels", "*.nii.gz")))
files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(images, labels)]
# print(files)

transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear")
        ),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1200.0, a_max=400, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        # transforms.ToTensord(keys=["image"]),
    ]
)

dataset = Dataset(data=files, transform=transforms)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

post_transforms = Compose([
    Invertd(
        keys="pred",
        transform=transforms,
        orig_keys="image",
        meta_keys="pred_meta_dict",
        orig_meta_keys="image_meta_dict",
        meta_key_postfix="meta_dict",
        nearest_interp=False,
        to_tensor=True,
    ),
    AsDiscreted(keys="pred", argmax=True),
    SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir="./out/train", output_postfix="seg", resample=False),
    # AsDiscreted(keys="pred", to_onehot=19),
    # AsDiscreted(keys="label", to_onehot=19),
])

acc_fun = DiceMetric(
    include_background=True,
    reduction='mean',
    get_not_nans=False
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=[32, 64, 128, 256, 512],
    strides=[2, 2, 2, 2],
    num_res_units=2,
    act="LEAKYRELU",
    norm="INSTANCE",
).to(device)

model.load_state_dict(torch.load(load_path)['params'])
model.eval()

with torch.no_grad():
    for val_data in loader:
        test_inputs = val_data["image"].to(device)
        roi_size = (192, 192, 96)
        sw_batch_size = 4
        val_data["pred"] = sliding_window_inference(
            test_inputs, roi_size, sw_batch_size, model, overlap=0.5, center_crop=False)
        val_data = [post_transforms(i) for i in decollate_batch(val_data)]
        # val_outputs, val_labels = from_engine(["pred", "label"])(val_data)
        # acc_fun.reset()
        # acc_fun(val_outputs, val_labels)
        # acc = acc_fun.aggregate().item()
        # val_data = val_data[0]
        # print(f"{osp.basename(val_data['image'].meta['filename_or_obj']).split('.')[0]}, origin shape: {val_data['label'].shape[1:]}, transform shape: {val_data['image'].shape[1:]}, acc: {acc}")

        del val_data
        torch.cuda.empty_cache()

