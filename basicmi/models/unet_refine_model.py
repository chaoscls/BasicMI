import os
import os.path as osp
from collections import OrderedDict

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from basicmi.archs import build_network
from basicmi.metrics import build_metric
from basicmi.models.base_model import BaseModel
from basicmi.utils import get_root_logger
from basicmi.utils.registry import MODEL_REGISTRY
from basicmi.inferers.utils import sliding_window_inference


@MODEL_REGISTRY.register()
class UNetRefineModel(BaseModel):
    """The GFPGAN model for Towards real-world blind face restoratin with generative facial prior"""

    def __init__(self, opt):
        super(UNetRefineModel, self).__init__(opt)
        self.idx = 0  # it is used for saving data for check

        # define network
        self.net = build_network(opt['network'])
        self.net = self.model_to_device(self.net)
        self.print_network(self.net)

        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key', 'params')
            self.load_network(self.net, load_path, self.opt['path'].get('strict_load', True), param_key)

        if self.is_train:
            self.init_training_settings()
        
        self.init_val_settings()
    
    def init_val_settings(self):
        dice_opt = self.opt['val']['metrics']['dice']
        self.dice_metric = build_metric(dice_opt)
        self.best_acc = -1

    def setup_optimizers(self):
        train_opt = self.opt['train']

        # ----------- optimizer g ----------- #
        normal_params = []
        for _, param in self.net.named_parameters():
            normal_params.append(param)
        optim_params = [{  # add normal params first
            'params': normal_params,
            'lr': train_opt['optim']['lr']
        }]
        optim_type = train_opt['optim'].pop('type')
        lr = train_opt['optim']['lr']
        weight_decay = train_opt['optim']['weight_decay']
        self.optimizer = self.get_optimizer(optim_type, optim_params, lr, weight_decay=weight_decay)
        self.scaler = None
        if self.opt['amp']:
            self.scaler = GradScaler()
        self.optimizers.append(self.optimizer)

    def feed_data(self, batch_data):
        if isinstance(batch_data, list):
            self.data, self.target = batch_data
        else:
            self.data, self.target= batch_data["image"], batch_data["label"]

        self.data = self.data.to(self.device)
        self.target = self.target.to(self.device)

        # uncomment to check data
        # import torchvision
        # if self.opt['rank'] == 0:
        #     import os
        #     os.makedirs('tmp/gt', exist_ok=True)
        #     os.makedirs('tmp/lq', exist_ok=True)
        #     print(self.idx)
        #     torchvision.utils.save_image(
        #         self.target, f'tmp/gt/gt_{self.idx}.png', nrow=4, padding=2, normalize=True, range=(-1, 1))
        #     torchvision.utils.save_image(
        #         self.data, f'tmp/lq/lq{self.idx}.png', nrow=4, padding=2, normalize=True, range=(-1, 1))
        #     self.idx = self.idx + 1
    
    def forward(self):
        loss_total = 0
        loss_dict = OrderedDict()
        with autocast(enabled=self.opt['amp']):
            losses = {}
            outputs = self.net(self.data, return_mid=True)
            for i, output in enumerate(outputs[:-1]):
                for criterion in self.criterions:
                    losses.update(criterion(output, self.target, return_dict=True, suffix=str(i+1)))
            for criterion in self.criterions:
                losses.update(criterion(outputs[-1], self.target, return_dict=True))

        for key, val in losses.items():
            loss_total += val
            loss_dict['l_'+key] = val

        if self.opt['amp']:
            self.scaler.scale(loss_total).backward()
        else:
            loss_total.backward()

        log_dict = self.reduce_loss_dict(loss_dict)
        self.update_log(log_dict)

    def test(self):
        self.net.eval()
        model_inferer_opt = self.opt['val']['model_inferer']
        with torch.no_grad():
            with autocast(self.opt['amp']):
                self.output = sliding_window_inference(
                    inputs=self.data,
                    roi_size=model_inferer_opt['roi_size'],
                    sw_batch_size=model_inferer_opt['sw_batch_size'],
                    predictor=self.net,
                    overlap=model_inferer_opt['infer_overlap'],
                )
        self.net.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
            # zero self.metric_results
            self.metric_results = {metric: 0 for metric in self.metric_results}

        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        if save_img:
            prediction_folder = osp.join(self.opt['path']['visualization'], dataset_name, f'preds_{current_iter}')
            os.makedirs(prediction_folder, exist_ok=True)

        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)
            img_name = osp.basename(self.data[0].meta['filename_or_obj']).split('.')[0]
            self.test()

            # if save_img:
                # original_affine = val_data[0]["image_meta_dict"]["affine"].numpy()
                # preds = val_outputs[0].argmax(dim=0).cpu().numpy().astype(np.int8)
                # nib.save(
                #     nib.Nifti1Image(preds.astype(np.uint8), original_affine), osp.join(prediction_folder, img_name+'_seg.nii.gz')
                # )

            if with_metrics:
                # calculate metrics
                # dice metric
                self.metric_results['dice'] += self.dice_metric(self.output, self.target).item()

            # tentative for out of GPU memory
            del self.data
            del self.output
            del val_data
            torch.cuda.empty_cache()

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)
            
            if current_iter > 5000 and dataset_name == "validation" and self.best_acc < self.metric_results['dice']:
                logger = get_root_logger()
                logger.info('Save the best model.')
                self.save_network(self.net, 'net', -2) # save best
                self.best_acc = self.metric_results['dice'] 

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)
