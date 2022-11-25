python ssl/select_reliable.py --data_dir experiments/dataset/LUNA16/st/images --load_root experiments/unet_baseline_center/models --models net_15000,net_30000,net_36500 --reliable_path experiments/dataset/LUNA16/st_pp/reliable --unreliable_path experiments/dataset/LUNA16/st_pp/unreliable

python ssl/label_pseudo.py --data_dir experiments/dataset/LUNA16/st_pp/reliable/images --load_path experiments/unet_baseline_center/models/net_36500.pth --out_dir experiments/dataset/LUNA16/st_pp/reliable/labels
