"""Disable mosaic/mixup and enable L1 loss for the final ``no_aug_epochs``.

Mirrors the transition that ``yolox.core.Trainer.before_epoch`` performs at
``epoch + 1 == max_epoch - no_aug_epochs``:

* Flip ``train_dataset.enable_mosaic = False`` so ``BoundedMosaicDetection``
  returns un-augmented samples.
* Flip ``model.head.use_l1 = True`` so the extra L1 loss term activates.
* Force per-epoch eval by setting ``exp.eval_interval = 1``.
* Call ``trainer.reset_train_dataloader(pl_module)`` so dataloader workers
  respawn and observe the new flag.

Idempotent: the transition runs at most once; once ``self._applied`` is set,
subsequent epochs skip the logic.
"""

from __future__ import annotations

import pytorch_lightning as pl

__all__ = ["NoAugToggleCallback"]


class NoAugToggleCallback(pl.Callback):
    def __init__(self):
        self._applied = False

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._applied:
            return

        exp = pl_module.exp
        current_epoch = int(trainer.current_epoch)
        threshold = exp.max_epoch - exp.no_aug_epochs
        if current_epoch < threshold:
            return

        datamodule = trainer.datamodule
        if datamodule is not None and getattr(datamodule, "train_dataset", None) is not None:
            datamodule.train_dataset.enable_mosaic = False

        head = getattr(pl_module.model, "head", None)
        if head is not None:
            head.use_l1 = True

        exp.eval_interval = 1
        trainer.reset_train_dataloader(pl_module)
        self._applied = True
