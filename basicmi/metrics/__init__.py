import importlib
from copy import deepcopy
from os import path as osp

from basicmi.utils import get_root_logger, scandir
from basicmi.utils.registry import METRIC_REGISTRY

__all__ = ['build_metric']

# automatically scan and import metric modules for registry
# scan all the files under the 'metrics' folder and collect files ending with '_metric.py'
metric_folder = osp.dirname(osp.abspath(__file__))
metric_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(metric_folder) if v.endswith('_metric.py')]
# import all the metric modules
_model_modules = [importlib.import_module(f'basicmi.metrics.{file_name}') for file_name in metric_filenames]


def build_metric(opt):
    """Build metric from options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**opt)
    logger = get_root_logger()
    logger.info(f'metric [{metric.__class__.__name__}] is created.')
    return metric

