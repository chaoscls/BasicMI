import os.path as osp
import os
import glob
import argparse
import shutil


import torch
import pandas as pd
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

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='experiments/dataset/18artery/train', type=str)
parser.add_argument('--load_root', type=str)
parser.add_argument('--reliable_path', type=str)
parser.add_argument('--unreliable_path', type=str)
parser.add_argument('--models', type=str, help='split by comma')
args = parser.parse_args()

data_dir = args.data_dir
load_root = args.load_root
images = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))
files = [{"image": image_name} for image_name in images]
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
    # Invertd(
    #     keys="pred",
    #     transform=transforms,
    #     orig_keys="image",
    #     meta_keys="pred_meta_dict",
    #     orig_meta_keys="image_meta_dict",
    #     meta_key_postfix="meta_dict",
    #     nearest_interp=False,
    #     to_tensor=True,
    # ),
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
    model_path = osp.join(load_root, path+'.pth')
    print(model_path)
    model.load_state_dict(torch.load(model_path)['params'])
    model.eval()
    models.append(model)

reliability_list = []

# import time
# import datetime
# idx = 0
with torch.no_grad():
    for val_data in tqdm(loader):
        # idx += 1
        # if idx > 10:
        #     break
        name = val_data['image'].meta['filename_or_obj'][0]
        test_inputs = val_data["image"].to(device)
        roi_size = (160, 160, 96)
        sw_batch_size = 16
        preds = []
        # start_time = time.time()
        for model in models:
            val_data['pred'] = sliding_window_inference(
                test_inputs, roi_size, sw_batch_size, model, overlap=0.5, center_crop=True).to('cpu')
            preds.append(from_engine(["pred"])([post_transforms(i) for i in decollate_batch(val_data)]))
        # consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        # print(consumed_time)

        del val_data["image"]
        torch.cuda.empty_cache()

        acc_list = []
        for i in range(len(preds) - 1):
            acc_fun.reset()
            acc_fun(preds[i], preds[-1])
            acc_list.append(acc_fun.aggregate().item())

        acc = sum(acc_list) / len(acc_list)

        reliability_list.append((name, acc))

# df = pd.DataFrame(reliability_list)
# writer = pd.ExcelWriter('1.xlsx')
# df.to_excel(writer)
# writer.save()


reliability_list.sort(key=lambda elem: elem[1], reverse=True)

for i, elem in enumerate(reliability_list):
    print(elem[0], osp.join(args.reliable_path, osp.basename(elem[0])))
    if i < len(reliability_list) // 2:
        shutil.copyfile(elem[0], osp.join(args.reliable_path, 'images', osp.basename(elem[0])))
    else:
        shutil.copyfile(elem[0], osp.join(args.unreliable_path, 'images', osp.basename(elem[0])))
