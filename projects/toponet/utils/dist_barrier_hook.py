import torch.distributed as dist
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class DistBarrierHook(Hook):
    def after_train_epoch(self, runner):
        if not dist.is_available() or not dist.is_initialized():
            return
        checkpoint_cfg = runner.checkpoint_config
        if checkpoint_cfg is None:
            return
        interval = checkpoint_cfg.get('interval', 1)
        if (runner.epoch + 1) % interval != 0:
            return
        dist.barrier()
