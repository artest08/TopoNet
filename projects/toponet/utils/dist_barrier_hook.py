import torch.distributed as dist
from mmcv.runner import HOOKS, Hook

try:
    from mmcv.runner.hooks import CheckpointHook
except Exception:
    CheckpointHook = None


def _find_checkpoint_hook(runner):
    hooks = getattr(runner, 'hooks', None)
    if hooks is None:
        hooks = getattr(runner, '_hooks', None)
    if not hooks or CheckpointHook is None:
        return None
    for hook in hooks:
        if isinstance(hook, CheckpointHook):
            return hook
    return None


@HOOKS.register_module()
class DistBarrierHook(Hook):
    def after_train_epoch(self, runner):
        if not dist.is_available() or not dist.is_initialized():
            return
        checkpoint_hook = _find_checkpoint_hook(runner)
        if checkpoint_hook is not None:
            interval = getattr(checkpoint_hook, 'interval', 1)
            by_epoch = getattr(checkpoint_hook, 'by_epoch', True)
            if by_epoch:
                if interval <= 0 or (runner.epoch + 1) % interval != 0:
                    return
        dist.barrier()
