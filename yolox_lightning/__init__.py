"""yolox_lightning — PyTorch Lightning replacement for the YOLOX training loop.

Public API mirrors the three submodules that downstream projects use:

* ``yolox_lightning.core.launch``
* ``yolox_lightning.exp.{Exp, check_exp_value, get_exp}``
* ``yolox_lightning.utils.{configure_module, configure_nccl, configure_omp, get_num_devices}``

Importing this package also patches ``yolox.exp.Exp.get_trainer`` so that
user-authored Exp files inheriting from the original ``yolox.exp.Exp`` run
through the Lightning loop without any source change.
"""

__version__ = "0.1.0"

from .exp.patch import patch_exp as _patch_exp

_patch_exp()
del _patch_exp
