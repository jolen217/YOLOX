"""Re-exports for ``Exp`` and ``check_exp_value``.

``yolox_lightning.exp.Exp`` is intentionally *the same class* as
``yolox.exp.Exp`` — after ``yolox_lightning`` is imported, the class's
``get_trainer`` method has been rebound by :mod:`yolox_lightning.exp.patch` to
construct a ``LightningYOLOXTrainer``. Re-exporting the same class keeps
``isinstance(exp, yolox.exp.Exp) is isinstance(exp, yolox_lightning.exp.Exp)``
true, so user code that inherits from either base behaves identically.
"""

from yolox.exp.yolox_base import Exp, check_exp_value

__all__ = ["Exp", "check_exp_value"]
