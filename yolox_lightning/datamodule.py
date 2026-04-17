"""``YOLOXDataModule``: train/val dataloaders for the Lightning runtime.

The train loader differs from ``yolox.exp.Exp.get_data_loader`` in three
deliberate ways:

* The dataset is wrapped with :class:`BoundedMosaicDetection` instead of
  ``yolox.data.datasets.MosaicDetection`` — preallocated canvas, bounded MixUp
  retries, and an ``enable_mosaic`` attribute flip (toggled by
  :class:`NoAugToggleCallback`) in place of the ``InfiniteSampler`` +
  ``YoloBatchSampler`` tuple protocol.
* The sampler is left unset so Lightning's ``DDPStrategy`` installs a stock
  ``DistributedSampler`` with per-epoch ``set_epoch`` handled automatically.
  This replaces YOLOX's ``InfiniteSampler`` (which relied on
  ``YoloBatchSampler`` to advance indefinitely) — Lightning's ``max_epochs``
  already bounds training, so an infinite sampler is unnecessary.
* ``persistent_workers=True`` so worker processes survive across epochs and
  retain their RAM caches / mmap handles.

The val loader is a 1-sample-per-rank ``IterableDataset`` because the COCO
evaluator drives its own dataloader internally — Lightning only needs a
stub that triggers ``validation_step`` once per rank.
"""

from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, IterableDataset

from yolox.data import TrainTransform
from yolox.data.dataloading import worker_init_reset_seed

from .data.mosaic import BoundedMosaicDetection

__all__ = ["YOLOXDataModule"]


class _DummyValDataset(IterableDataset):
    """Yields a single zero-tensor batch — exists only to enter ``validation_step`` once per rank."""

    def __iter__(self):
        yield torch.zeros(1)


class YOLOXDataModule(pl.LightningDataModule):
    def __init__(self, exp, batch_size: int, cache: Optional[str] = None):
        super().__init__()
        self.exp = exp
        self.batch_size = batch_size
        self.cache = cache
        self.train_dataset: Optional[BoundedMosaicDetection] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit") and self.train_dataset is None:
            base = self.exp.get_dataset(
                cache=self.cache is not None,
                cache_type=self.cache or "ram",
            )
            self.train_dataset = BoundedMosaicDetection(
                dataset=base,
                img_size=self.exp.input_size,
                mosaic=True,
                preproc=TrainTransform(
                    max_labels=120,
                    flip_prob=self.exp.flip_prob,
                    hsv_prob=self.exp.hsv_prob,
                ),
                degrees=self.exp.degrees,
                translate=self.exp.translate,
                mosaic_scale=self.exp.mosaic_scale,
                mixup_scale=self.exp.mixup_scale,
                shear=self.exp.shear,
                enable_mixup=self.exp.enable_mixup,
                mosaic_prob=self.exp.mosaic_prob,
                mixup_prob=self.exp.mixup_prob,
            )

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None, "setup() was not called before train_dataloader()"
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        per_rank_batch = max(1, self.batch_size // world_size)

        return DataLoader(
            self.train_dataset,
            batch_size=per_rank_batch,
            num_workers=self.exp.data_num_workers,
            pin_memory=True,
            persistent_workers=self.exp.data_num_workers > 0,
            drop_last=False,
            worker_init_fn=worker_init_reset_seed,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(_DummyValDataset(), batch_size=1, num_workers=0)
