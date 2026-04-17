"""``EMACallback`` must round-trip its state through a Lightning checkpoint."""

import unittest

import torch
from torch import nn

from yolox_lightning.callbacks.ema import EMACallback


class _DummyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(4, 4)


class _FakeTrainer:
    def __init__(self, global_step: int = 10, current_epoch: int = 2):
        self.global_step = global_step
        self.current_epoch = current_epoch
        self.is_global_zero = True


class TestEMACallback(unittest.TestCase):
    def test_state_dict_roundtrip(self):
        src = _DummyModule()
        cb = EMACallback(decay=0.99)
        cb.on_train_start(_FakeTrainer(), src)

        # Simulate a few updates so the EMA weights drift from init.
        for _ in range(5):
            with torch.no_grad():
                for p in src.model.parameters():
                    p.add_(torch.randn_like(p) * 0.01)
            cb.on_train_batch_end(_FakeTrainer(), src, outputs=None, batch=None, batch_idx=0)

        ckpt = {}
        cb.on_save_checkpoint(_FakeTrainer(), src, ckpt)
        self.assertIn("ema_state_dict", ckpt)
        self.assertIn("ema_updates", ckpt)

        dst = _DummyModule()
        cb2 = EMACallback(decay=0.99)
        cb2.on_load_checkpoint(_FakeTrainer(), dst, ckpt)

        for k, v in cb.ema.ema.state_dict().items():
            self.assertTrue(torch.allclose(v, cb2.ema.ema.state_dict()[k]))
        self.assertEqual(cb.ema.updates, cb2.ema.updates)

    def test_validation_swap_restores_weights(self):
        src = _DummyModule()
        cb = EMACallback(decay=0.99)
        cb.on_train_start(_FakeTrainer(), src)

        # Drift EMA away from live weights.
        with torch.no_grad():
            for p in cb.ema.ema.parameters():
                p.add_(1.0)

        live_before = {k: v.detach().clone() for k, v in src.model.state_dict().items()}
        cb.on_validation_start(_FakeTrainer(), src)

        # During validation, live weights should equal EMA.
        for k, v in cb.ema.ema.state_dict().items():
            self.assertTrue(torch.allclose(v, src.model.state_dict()[k]))

        cb.on_validation_end(_FakeTrainer(), src)

        # After validation, live weights should be restored.
        for k, v in live_before.items():
            self.assertTrue(torch.allclose(v, src.model.state_dict()[k]))


if __name__ == "__main__":
    unittest.main()
