"""Monkey-patch ``yolox.exp.Exp.get_trainer`` to route through Lightning.

User-authored Exp files in the wild inherit from ``yolox.exp.Exp`` and call
``exp.get_trainer(args)`` in their CLI entry points. Rather than require those
files to switch their base class, we rebind the method at import time so
*any* subclass of ``yolox.exp.Exp`` builds a ``LightningYOLOXTrainer`` instead
of the original ``yolox.core.Trainer``.

The patch is idempotent: repeated calls (e.g. from a reload) leave the class
in the same final state.
"""

from __future__ import annotations

__all__ = ["patch_exp"]

_PATCH_MARKER = "_yolox_lightning_patched"


def patch_exp() -> None:
    from yolox.exp import Exp as YoloXExp

    if getattr(YoloXExp, _PATCH_MARKER, False):
        return

    def get_trainer(self, args):
        from yolox_lightning.trainer import LightningYOLOXTrainer

        trainer = LightningYOLOXTrainer(self, args)
        return trainer

    YoloXExp.get_trainer = get_trainer
    setattr(YoloXExp, _PATCH_MARKER, True)
