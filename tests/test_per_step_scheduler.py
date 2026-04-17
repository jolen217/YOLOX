"""``PerStepLRScheduler`` must produce the same LR sequence as ``yolox.utils.LRScheduler.update_lr``."""

import unittest

import torch

from yolox.utils import LRScheduler
from yolox_lightning.schedulers import PerStepLRScheduler


class TestPerStepLRScheduler(unittest.TestCase):
    def test_sequence_matches_update_lr(self):
        iters_per_epoch = 100
        max_epoch = 5
        total_iters = iters_per_epoch * max_epoch

        model = torch.nn.Linear(4, 4)
        opt = torch.optim.SGD(model.parameters(), lr=1.0)

        yolo_sched = LRScheduler(
            "yoloxwarmcos",
            lr=0.01,
            iters_per_epoch=iters_per_epoch,
            total_epochs=max_epoch,
            warmup_epochs=1,
            warmup_lr_start=0.0,
            no_aug_epochs=1,
            min_lr_ratio=0.05,
        )

        ref = [yolo_sched.update_lr(i) for i in range(total_iters)]

        model2 = torch.nn.Linear(4, 4)
        opt2 = torch.optim.SGD(model2.parameters(), lr=1.0)
        yolo_sched2 = LRScheduler(
            "yoloxwarmcos",
            lr=0.01,
            iters_per_epoch=iters_per_epoch,
            total_epochs=max_epoch,
            warmup_epochs=1,
            warmup_lr_start=0.0,
            no_aug_epochs=1,
            min_lr_ratio=0.05,
        )
        wrapped = PerStepLRScheduler(opt2, yolo_sched2)

        observed = []
        for _ in range(total_iters):
            observed.append(opt2.param_groups[0]["lr"])
            opt2.step()
            wrapped.step()

        # The wrapper writes LR at last_epoch=max(last_epoch,0); the first
        # observed value corresponds to iter 0, same as ref[0].
        for i, (a, b) in enumerate(zip(observed, ref)):
            self.assertAlmostEqual(a, b, places=7, msg=f"mismatch at iter {i}")


if __name__ == "__main__":
    unittest.main()
