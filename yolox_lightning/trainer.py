"""Drop-in replacement for ``yolox.core.Trainer``.

``LightningYOLOXTrainer(exp, args).train()`` constructs the Lightning module,
datamodule, callbacks, strategy and ``pl.Trainer`` in the shape the original
``yolox.core.Trainer`` would have used, and calls ``trainer.fit``.

The distributed context is already initialised by
:func:`yolox_lightning.core.launch.launch` — environment variables
(``LOCAL_RANK``/``RANK``/``WORLD_SIZE``/``MASTER_ADDR``/``MASTER_PORT``) are
populated and ``torch.distributed`` has called ``init_process_group``. Passing
``cluster_environment=LightningEnvironment()`` tells Lightning to adopt that
existing group rather than spawn a second layer of workers.
"""

from __future__ import annotations

import os
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning.strategies import DDPStrategy, SingleDeviceStrategy

from .callbacks import EMACallback, MultiscaleCallback, NoAugToggleCallback
from .datamodule import YOLOXDataModule
from .module import LightningYOLOX

__all__ = ["LightningYOLOXTrainer"]


class LightningYOLOXTrainer:
    def __init__(self, exp, args):
        self.exp = exp
        self.args = args
        self.output_dir = os.path.join(exp.output_dir, args.experiment_name or exp.exp_name)

    def _build_strategy(self, world_size: int):
        if world_size <= 1:
            return SingleDeviceStrategy(device=f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
        return DDPStrategy(
            cluster_environment=LightningEnvironment(),
            process_group_backend="nccl",
            find_unused_parameters=False,
        )

    def _build_callbacks(self) -> list:
        ckpt_dir = os.path.join(self.output_dir, "checkpoints")
        return [
            EMACallback(
                decay=0.9998,
                legacy_ckpt_dir=self.output_dir,
                legacy_ckpt_name="ema_latest",
            ),
            MultiscaleCallback(resize_every=10),
            NoAugToggleCallback(),
            ModelCheckpoint(
                dirpath=ckpt_dir,
                monitor="val/COCOAP50_95",
                mode="max",
                save_top_k=1,
                filename="best",
                save_last=True,
            ),
            LearningRateMonitor(logging_interval="step"),
        ]

    def train(self) -> None:
        args = self.args
        exp = self.exp

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        devices = max(1, world_size) if world_size > 1 else (args.devices or 1)

        os.makedirs(self.output_dir, exist_ok=True)

        module = LightningYOLOX(exp, batch_size=args.batch_size)
        datamodule = YOLOXDataModule(exp, batch_size=args.batch_size, cache=args.cache)

        logger: Optional[TensorBoardLogger] = None
        if getattr(args, "logger", "tensorboard") == "tensorboard":
            logger = TensorBoardLogger(save_dir=self.output_dir, name="tensorboard")

        strategy = self._build_strategy(world_size)

        trainer = pl.Trainer(
            max_epochs=exp.max_epoch,
            accelerator="gpu",
            devices=devices,
            strategy=strategy,
            precision="16-mixed" if args.fp16 else "32-true",
            callbacks=self._build_callbacks(),
            logger=logger,
            default_root_dir=self.output_dir,
            num_sanity_val_steps=0,
            check_val_every_n_epoch=exp.eval_interval,
            log_every_n_steps=exp.print_interval,
            enable_progress_bar=(int(os.environ.get("RANK", "0")) == 0),
            use_distributed_sampler=True,
        )

        ckpt_path = args.ckpt if getattr(args, "resume", False) else None
        trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)
