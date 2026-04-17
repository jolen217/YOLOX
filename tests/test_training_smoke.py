"""End-to-end CPU smoke test.

Not a correctness test — just validates that ``LightningYOLOX`` +
``configure_optimizers`` + ``EMACallback`` + ``MultiscaleCallback`` all plug
into ``pl.Trainer`` without shape errors, type errors, or hook-ordering bugs.

Uses a synthetic DataLoader so the test is hermetic (no COCO, no disk I/O).
Validation is skipped because the COCOEvaluator needs a real dataset;
validation-path coverage lives in ``test_ema_callback`` + future GPU tests.
"""

import unittest
from typing import Iterator, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, IterableDataset

from yolox.exp import Exp
from yolox_lightning.callbacks.ema import EMACallback
from yolox_lightning.callbacks.multiscale import MultiscaleCallback
from yolox_lightning.module import LightningYOLOX


class _TinyExp(Exp):
    def __init__(self):
        super().__init__()
        self.depth = 0.33
        self.width = 0.25
        self.num_classes = 3
        self.input_size = (64, 64)
        self.test_size = (64, 64)
        self.multiscale_range = 0
        self.max_epoch = 2
        self.warmup_epochs = 1
        self.no_aug_epochs = 0
        self.eval_interval = 1000
        self.print_interval = 1
        self.basic_lr_per_img = 0.01 / 64
        self.data_num_workers = 0


class _SyntheticBatches(IterableDataset):
    """Yields ``num_batches`` random (image, target) pairs."""

    def __init__(self, num_batches: int, batch_size: int, img_size: int, max_labels: int = 50):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.img_size = img_size
        self.max_labels = max_labels

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for _ in range(self.num_batches):
            img = torch.rand(self.batch_size, 3, self.img_size, self.img_size) * 255.0
            # target layout per YOLOXHead: [class, cx, cy, w, h]; only first 2 boxes populated.
            target = torch.zeros(self.batch_size, self.max_labels, 5)
            target[:, 0] = torch.tensor([0, 16.0, 16.0, 10.0, 10.0])
            target[:, 1] = torch.tensor([1, 40.0, 40.0, 8.0, 8.0])
            yield img, target


def _collate_noop(batch):
    # Dataset already yields batched tensors; DataLoader wraps each in a list of 1.
    return batch[0]


class TestTrainingSmoke(unittest.TestCase):
    def test_two_steps_cpu(self):
        exp = _TinyExp()
        module = LightningYOLOX(exp, batch_size=2)
        # Tell configure_optimizers how many iters/epoch to assume (sidesteps the
        # trainer.datamodule lookup, since we don't use a DataModule here).
        module._iters_per_epoch = 2

        loader = DataLoader(
            _SyntheticBatches(num_batches=2, batch_size=2, img_size=64),
            batch_size=1,
            collate_fn=_collate_noop,
            num_workers=0,
        )

        trainer = pl.Trainer(
            max_steps=2,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            callbacks=[EMACallback(decay=0.9), MultiscaleCallback(resize_every=1)],
        )
        trainer.fit(module, train_dataloaders=loader)

        for p in module.model.parameters():
            self.assertTrue(torch.isfinite(p).all(), "parameter became non-finite")
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all(), "gradient became non-finite")


if __name__ == "__main__":
    unittest.main()
