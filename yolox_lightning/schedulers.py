"""Per-step adapter around ``yolox.utils.LRScheduler``.

Lightning wants a ``torch.optim.lr_scheduler._LRScheduler`` instance and will
call its ``.step()`` every training step when configured with
``interval="step", frequency=1``. YOLOX's ``LRScheduler`` is function-like: it
takes a global iteration count and returns a learning rate. This wrapper
bridges the two without touching the optimizer directly on each forward pass.
"""

from torch.optim.lr_scheduler import _LRScheduler

from yolox.utils import LRScheduler as YoloXLRScheduler

__all__ = ["PerStepLRScheduler", "build_per_step_scheduler"]


class PerStepLRScheduler(_LRScheduler):
    """Wrap a ``yolox.utils.LRScheduler`` so Lightning can step it per batch.

    The wrapper treats ``self.last_epoch`` (which PyTorch increments on every
    ``.step()`` call) as the global training iteration. All parameter groups
    receive the same learning rate — matching the original YOLOX trainer which
    writes one computed LR into every ``param_group["lr"]``.
    """

    def __init__(self, optimizer, yolox_scheduler: YoloXLRScheduler, last_epoch: int = -1):
        self._yolox_scheduler = yolox_scheduler
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        lr = self._yolox_scheduler.update_lr(max(self.last_epoch, 0))
        return [lr for _ in self.optimizer.param_groups]


def build_per_step_scheduler(optimizer, exp, iters_per_epoch: int, batch_size: int):
    """Construct the YOLOX schedule for one optimizer and wrap it for Lightning.

    ``exp.basic_lr_per_img * batch_size`` is the peak LR, matching the
    ``LRScheduler(... lr=exp.basic_lr_per_img * batch_size, ...)`` call in
    ``yolox/core/trainer.py:162``.
    """
    yolox_scheduler = YoloXLRScheduler(
        exp.scheduler,
        exp.basic_lr_per_img * batch_size,
        iters_per_epoch,
        exp.max_epoch,
        warmup_epochs=exp.warmup_epochs,
        warmup_lr_start=exp.warmup_lr,
        no_aug_epochs=exp.no_aug_epochs,
        min_lr_ratio=exp.min_lr_ratio,
    )
    return PerStepLRScheduler(optimizer, yolox_scheduler)
