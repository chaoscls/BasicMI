import os
import glob
import argparse
# from multiprocessing import Pool

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

from basicmi.inferers.utils import sliding_window_inference

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='experiments/dataset/LUNA16/st/images', type=str)
parser.add_argument('--load_path', default='experiments/unet_baseline_center/models/net_36500.pth', type=str)
parser.add_argument('--out_dir', default='experiments/dataset/LUNA16/st/labels', type=str)
args = parser.parse_args()

data_dir = args.data_dir
load_path = args.load_path
images = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))
exist_name = os.listdir(args.out_dir)
files = [{"image": image_name} for image_name in images if os.path.basename(image_name).split('.')[0] not in exist_name]
print(len(files))
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
    AsDiscreted(keys="pred", argmax=True),
    SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=args.out_dir, output_postfix="seg", resample=False),
])


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=19,
    channels=[32, 64, 128, 256, 512],
    strides=[2, 2, 2, 2],
    num_res_units=2,
    act="LEAKYRELU",
    norm="INSTANCE",
).to(device)

model.load_state_dict(torch.load(load_path)['params'])
model.eval()


# print(exist_name)

# pool = Pool(processes=8)

# def fun(val_data):
#     [post_transforms(i) for i in decollate_batch(val_data)]


with torch.no_grad():
    for val_data in tqdm(loader):
        # print("=>", val_data['image'].meta['filename_or_obj'])
        # if os.path.basename(val_data['image'].meta['filename_or_obj'][0]).split('.')[0] in exist_name:
        #     continue
        test_inputs = val_data["image"].to(device)
        roi_size = (192, 192, 96)
        sw_batch_size = 8
        val_data["pred"] = sliding_window_inference(
            test_inputs, roi_size, sw_batch_size, model, overlap=0.5).to('cpu')
        
        del val_data["image"]
        torch.cuda.empty_cache()
        
        # pool.apply_async(fun, (val_data,))

        val_data = [post_transforms(i) for i in decollate_batch(val_data)]


