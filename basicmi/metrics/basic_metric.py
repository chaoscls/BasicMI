import torch.nn as nn
from monai import metrics
from monai.utils.enums import MetricReduction
from monai.transforms import AsDiscrete

from basicmi.utils.registry import METRIC_REGISTRY
from monai.data import decollate_batch

@METRIC_REGISTRY.register()
class DiceMetric(nn.Module):
    
    def __init__(self, to_onehot, include_background=False, reduction='mean', get_not_nans=False):
        super(DiceMetric, self).__init__()
        self.post_label = AsDiscrete(to_onehot=to_onehot)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=to_onehot)
        self.acc_func = metrics.DiceMetric(include_background=include_background, reduction=reduction, get_not_nans=get_not_nans)
    
    def forward(self, logits, target):
        val_labels_list = decollate_batch(target)
        val_labels_convert = [self.post_label(val_label_tensor) for val_label_tensor in val_labels_list]
        val_outputs_list = decollate_batch(logits)
        val_output_convert = [self.post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
        self.acc_func.reset()
        self.acc_func(y_pred=val_output_convert, y=val_labels_convert)
        acc = self.acc_func.aggregate()
        return acc
