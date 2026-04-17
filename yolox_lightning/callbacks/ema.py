"""EMA callback — wraps ``yolox.utils.ModelEMA``.

Three jobs:

1. Build a ``ModelEMA`` mirror of the training model once the model is on its
   final device (``on_train_start`` — Lightning has already moved parameters).
2. Step the EMA every optimizer step so the shadow weights stay current.
3. Swap EMA weights in for validation (so ``LightningYOLOX.validation_step``
   evaluates the EMA model) and swap them back out afterwards.

Checkpoint integration: the shadow state is written under ``ema_state_dict``
inside the Lightning checkpoint, and — when ``legacy_ckpt_dir`` is configured
— also dumped alongside in the original ``yolox.utils.save_checkpoint``
``.pth`` format so ``-c ckpt.pth`` tooling in the legacy pipeline keeps
loading.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch

from yolox.utils import ModelEMA, save_checkpoint

__all__ = ["EMACallback"]


class EMACallback(pl.Callback):
    def __init__(
        self,
        decay: float = 0.9998,
        legacy_ckpt_dir: Optional[str] = None,
        legacy_ckpt_name: str = "ema_latest",
    ):
        self.decay = decay
        self.legacy_ckpt_dir = legacy_ckpt_dir
        self.legacy_ckpt_name = legacy_ckpt_name

        self.ema: Optional[ModelEMA] = None
        self._swapped_state: Optional[Dict[str, torch.Tensor]] = None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.ema is None:
            self.ema = ModelEMA(pl_module.model, self.decay)
            # Align the EMA step counter with global_step so a resumed run
            # continues the decay ramp from the right point.
            self.ema.updates = int(trainer.global_step)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.ema is not None:
            self.ema.update(pl_module.model)

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.ema is None:
            return
        # Swap live weights with EMA weights for the duration of validation.
        live = pl_module.model.state_dict()
        self._swapped_state = {k: v.detach().clone() for k, v in live.items()}
        ema_state = self.ema.ema.state_dict()
        pl_module.model.load_state_dict(ema_state, strict=False)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._swapped_state is None:
            return
        pl_module.model.load_state_dict(self._swapped_state, strict=False)
        self._swapped_state = None

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        if self.ema is None:
            return
        checkpoint["ema_state_dict"] = self.ema.ema.state_dict()
        checkpoint["ema_updates"] = int(self.ema.updates)

        if self.legacy_ckpt_dir and trainer.is_global_zero:
            os.makedirs(self.legacy_ckpt_dir, exist_ok=True)
            save_checkpoint(
                {
                    "start_epoch": int(trainer.current_epoch) + 1,
                    "model": self.ema.ema.state_dict(),
                },
                False,
                self.legacy_ckpt_dir,
                self.legacy_ckpt_name,
            )

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        ema_state = checkpoint.get("ema_state_dict")
        if ema_state is None:
            return
        if self.ema is None:
            self.ema = ModelEMA(pl_module.model, self.decay)
        self.ema.ema.load_state_dict(ema_state)
        self.ema.updates = int(checkpoint.get("ema_updates", 0))
