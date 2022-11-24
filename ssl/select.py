import os.path as osp
import os
import glob
import argparse
import shutil


import torch
from tqdm import tqdm
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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='experiments/dataset/18artery/train', type=str)
parser.add_argument('--load_root', type=str)
parser.add_argument('--reliable_path', type=str)
parser.add_argument('--unreliable_path', type=str)
parser.add_argument('--models', type=str, help='split by comma')
args = parser.parse_args()

data_dir = args.data_dir
load_path = args.load_path
images = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))
files = [{"image": image_name, "label": None} for image_name in images]
# print(files)

transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear")
        ),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1200.0, a_max=400, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image"], source_key="image"),
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
    AsDiscreted(keys="pred", argmax=True, to_onehot=19),
])

acc_fun = DiceMetric(
    include_background=True,
    reduction='mean',
    get_not_nans=False
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
models = []
for path in args.models.split(","):
    model = UNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=19,
        channels=[32, 64, 128, 256, 512],
        strides=[2, 2, 2, 2],
        num_res_units=2,
        act="LEAKYRELU",
        norm="INSTANCE",
    ).to(device)

    model.load_state_dict(torch.load(osp.join(load_path, path))['params'])
    model.eval()
    models.append(model)

reliability_list = []

with torch.no_grad():
    for val_data in tqdm(loader):
        test_inputs = val_data["image"].to(device)
        roi_size = (160, 160, 96)
        sw_batch_size = 4
        preds = []
        for model in models:
            val_data["pred"] = sliding_window_inference(
                test_inputs, roi_size, sw_batch_size, model, overlap=0.5, center_crop=True)
            val_data = [post_transforms(i) for i in decollate_batch(val_data)]
            val_outputs = from_engine(["pred"])(val_data)

        acc_list = []
        for i in range(len(preds) - 1):
            acc_fun.reset()
            acc_fun(preds[0], preds[-1])
            acc_list.append(acc_fun.aggregate().item())
        val_data = val_data[0]
        acc = sum(acc_list) / len(acc_list)

        reliability_list.append((val_data['image'].meta['filename_or_obj'], acc))
        
        del val_data
        torch.cuda.empty_cache()

reliability_list.sort(key=lambda elem: elem[1], reverse=True)

for i, elem in enumerate(reliability_list):
    if i < len(reliability_list) // 2:
        shutil.copyfile(elem[0], osp.join(args.reliable_path, osp.basename(elem[0])))
    else:
        shutil.copyfile(elem[0], osp.join(args.unreliable_path, osp.basename(elem[0])))
