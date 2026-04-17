"""Smoke-test that ``yolox_lightning`` exposes exactly the downstream API."""

import unittest


class TestPublicAPI(unittest.TestCase):
    def test_core_launch(self):
        from yolox_lightning.core import launch  # noqa: F401

    def test_exp_surface(self):
        from yolox_lightning.exp import Exp, check_exp_value, get_exp  # noqa: F401

    def test_utils_surface(self):
        from yolox_lightning.utils import (  # noqa: F401
            configure_module,
            configure_nccl,
            configure_omp,
            get_num_devices,
        )

    def test_get_trainer_is_patched(self):
        import yolox_lightning  # noqa: F401 — triggers the patch
        from yolox.exp import Exp as YoloXExp

        self.assertTrue(getattr(YoloXExp, "_yolox_lightning_patched", False))


if __name__ == "__main__":
    unittest.main()
