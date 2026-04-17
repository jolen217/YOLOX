"""LightningModule wrapper around YOLOX.

Maps the hook-structured ``yolox.core.Trainer`` loop onto Lightning:

* ``training_step`` = one call of ``train_one_iter``: preprocess (for
  multiscale), forward with ``targets`` so the model returns its loss dict,
  return ``total_loss``. Backward/optimizer/scheduler stepping are handled by
  Lightning.
* ``configure_optimizers`` returns ``exp.get_optimizer(batch_size)`` and a
  :class:`~yolox_lightning.schedulers.PerStepLRScheduler` with
  ``interval="step"`` so the YOLOX schedule advances once per batch.
* Validation is intentionally minimal. The COCO evaluator drives its own loop
  over its own dataloader, so the val dataloader supplied to Lightning is a
  single-batch dummy that just ensures every rank enters the validation epoch
  (DDP needs all ranks to participate). The real mAP computation runs inside
  :meth:`validation_step` exactly once per eval cycle.

The ``input_size`` attribute is deliberately mutable — :class:`MultiscaleCallback`
rewrites it every N steps with a distributed broadcast.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch import nn

__all__ = ["LightningYOLOX"]


class LightningYOLOX(pl.LightningModule):
    def __init__(self, exp, batch_size: int):
        super().__init__()
        self.exp = exp
        self.batch_size = batch_size
        self.model: nn.Module = exp.get_model()
        self.input_size: Tuple[int, int] = tuple(exp.input_size)

        self._iters_per_epoch: Optional[int] = None
        self._evaluator = None
        self._best_ap: float = 0.0

    def configure_optimizers(self):
        from yolox_lightning.schedulers import build_per_step_scheduler

        optimizer = self.exp.get_optimizer(self.batch_size)
        iters_per_epoch = self._iters_per_epoch
        if iters_per_epoch is None:
            iters_per_epoch = len(self.trainer.datamodule.train_dataloader())
            self._iters_per_epoch = iters_per_epoch

        scheduler = build_per_step_scheduler(
            optimizer, self.exp, iters_per_epoch, self.batch_size
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def training_step(self, batch, batch_idx: int):
        inps, targets = batch[0], batch[1]
        targets.requires_grad_(False)
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        outputs: Dict[str, torch.Tensor] = self.model(inps, targets)

        log_payload = {f"train/{k}": v for k, v in outputs.items()}
        log_payload["train/lr"] = self.optimizers().param_groups[0]["lr"]
        log_payload["train/input_h"] = float(self.input_size[0])
        self.log_dict(
            log_payload,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            batch_size=inps.shape[0],
        )
        return outputs["total_loss"]

    def _ensure_evaluator(self):
        if self._evaluator is None:
            self._evaluator = self.exp.get_evaluator(
                batch_size=self.batch_size,
                is_distributed=(self.trainer.world_size > 1),
            )
        return self._evaluator

    def validation_step(self, batch, batch_idx: int):
        # One dummy batch per rank — skip repeated invocations (only possible
        # if the dummy dataloader length is >1 for any reason).
        if batch_idx != 0:
            return None

        evaluator = self._ensure_evaluator()
        is_distributed = self.trainer.world_size > 1

        (ap50_95, ap50, summary), _ = self.exp.eval(
            self.model, evaluator, is_distributed, return_outputs=True
        )

        self._best_ap = max(self._best_ap, float(ap50_95))
        self.log("val/COCOAP50_95", float(ap50_95), rank_zero_only=True, sync_dist=False)
        self.log("val/COCOAP50", float(ap50), rank_zero_only=True, sync_dist=False)
        self.log("val/best_ap", float(self._best_ap), rank_zero_only=True, sync_dist=False)
        if self.global_rank == 0 and summary:
            self.print(summary)
        return None

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["yolox_input_size"] = tuple(self.input_size)
        checkpoint["yolox_best_ap"] = self._best_ap

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        size = checkpoint.get("yolox_input_size")
        if size is not None:
            self.input_size = tuple(size)
        self._best_ap = float(checkpoint.get("yolox_best_ap", 0.0))
