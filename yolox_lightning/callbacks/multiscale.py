"""Multiscale training callback.

Every ``resize_every`` batches, rank 0 samples a new ``(h, w)`` from the
exp-configured multiscale range, broadcasts it to the other ranks via
``dist.broadcast`` and writes it to ``pl_module.input_size``. The actual image
resize happens inside ``LightningYOLOX.training_step`` via
``exp.preprocess(inputs, targets, self.input_size)`` — this callback only
decides *which* size to use.

Correctness hinges on the callback firing on **every** rank on **every**
batch: if some rank skips the broadcast, NCCL will deadlock. The frequency
check runs identically on every rank before the broadcast.
"""

from __future__ import annotations

import random
from typing import Any

import pytorch_lightning as pl
import torch
import torch.distributed as dist

__all__ = ["MultiscaleCallback"]


class MultiscaleCallback(pl.Callback):
    def __init__(self, resize_every: int = 10):
        self.resize_every = resize_every

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        step = int(trainer.global_step)
        if step == 0 or step % self.resize_every != 0:
            return

        exp = pl_module.exp
        base_h, base_w = exp.input_size
        size_factor = base_w / base_h

        if trainer.is_global_zero:
            if not hasattr(exp, "random_size") or exp.random_size is None:
                min_size = base_h // 32 - exp.multiscale_range
                max_size = base_h // 32 + exp.multiscale_range
                random_size = (min_size, max_size)
            else:
                random_size = exp.random_size
            s = random.randint(*random_size)
            new_h = int(32 * s)
            new_w = int(32 * int(s * size_factor))
        else:
            new_h = 0
            new_w = 0

        if dist.is_initialized():
            buf = torch.tensor([new_h, new_w], device=pl_module.device, dtype=torch.long)
            dist.broadcast(buf, src=0)
            new_h, new_w = int(buf[0].item()), int(buf[1].item())

        pl_module.input_size = (new_h, new_w)
