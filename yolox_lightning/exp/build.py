"""``get_exp`` shim that returns patched Exp instances.

Delegates to ``yolox.exp.get_exp``. Because
:mod:`yolox_lightning.exp.patch` has already rebound
``yolox.exp.Exp.get_trainer`` at package import time, the instance returned
here routes through Lightning even though it is still nominally a
``yolox.exp.Exp`` subclass.
"""

from yolox.exp.build import get_exp as _get_exp

__all__ = ["get_exp"]


def get_exp(exp_file=None, exp_name=None):
    return _get_exp(exp_file=exp_file, exp_name=exp_name)
