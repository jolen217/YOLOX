# yolox_lightning

PyTorch Lightning replacement for the YOLOX training loop. Lives beside the
stock `yolox/` tree and reuses its model, dataset, evaluator, and utility
code. Nothing under `yolox/` is source-edited; one method is rebound at
import time.

## Public API

Exactly seven symbols — the set a downstream project imports from
`yolox.{core,exp,utils}`. Swap `yolox` → `yolox_lightning` on those import
lines and nothing else needs to change.

| Path | Symbols |
|---|---|
| `yolox_lightning.core` | `launch` |
| `yolox_lightning.exp` | `Exp`, `check_exp_value`, `get_exp` |
| `yolox_lightning.utils` | `configure_module`, `configure_nccl`, `configure_omp`, `get_num_devices` |

`yolox_lightning.exp.Exp` is literally the same class as `yolox.exp.Exp` —
importing `yolox_lightning` rebinds `Exp.get_trainer` to return a
`LightningYOLOXTrainer` instead of the original `yolox.core.Trainer`. User
Exp files inheriting from either base are picked up unchanged.

Everything else in this package is internal.

## Usage

```python
# downstream/main.py
from yolox_lightning.core import launch
from yolox_lightning.exp import get_exp
from yolox_lightning.utils import configure_module, configure_nccl, configure_omp, get_num_devices

def main_func(exp, args):
    configure_nccl()
    configure_omp()
    trainer = exp.get_trainer(args)   # LightningYOLOXTrainer
    trainer.train()                   # pl.Trainer.fit(...)

if __name__ == "__main__":
    exp = get_exp(exp_name="yolox-s")
    args = parse_args()
    launch(main_func, num_gpus_per_machine=get_num_devices(), args=(exp, args))
```

`launch` spawns one process per GPU via `torch.multiprocessing`, initialises
NCCL, and populates the rank / world-size environment variables that
Lightning's `LightningEnvironment` reads. Lightning therefore adopts the
existing process group instead of spawning its own.

## Why this rewrite

The original training path leaks data-pipeline throughput in several places.
`yolox_lightning` fixes them while keeping numerics bit-for-bit compatible
where possible:

| Original | Replacement |
|---|---|
| `yolox/data/datasets/datasets_wrapper.py:289` — `copy.deepcopy(img)` on every RAM-cache read | `data/cache.py::MemmapCOCODataset` — returns views for RAM, `np.load(..., mmap_mode="r")` for disk |
| `yolox/data/datasets/mosaicdetection.py` — fresh `114`-filled canvas every sample, unbounded MixUp retry | `data/mosaic.py::BoundedMosaicDetection` — preallocated per-worker canvas, `max_mixup_tries=8` |
| `InfiniteSampler` + `YoloBatchSampler` + hand-rolled `DataPrefetcher` + per-iter `F.interpolate` from the outer loop | stock `DistributedSampler` (wrapped by `DDPStrategy`), `persistent_workers=True`, `pin_memory=True`; multiscale lives in `MultiscaleCallback` with a single `dist.broadcast` per resize step |

License audit: YOLOX itself is Apache 2.0. Every runtime dependency is OSI-
approved permissive (BSD / MIT / Apache / MPL-2.0). No clean-room rewrite was
required — `yolox_lightning` freely reuses `yolox/*` internals.

## Internal layout

```
yolox_lightning/
  __init__.py            # patches yolox.exp.Exp.get_trainer at import
  core/launch.py         # mp.spawn one worker per GPU; NCCL + env vars
  exp/
    patch.py             # the one monkey-patch
    yolox_base.py        # re-exports Exp, check_exp_value
    build.py             # re-exports get_exp
  utils/setup_env.py     # configure_{module,nccl,omp} + get_num_devices
  schedulers.py          # PerStepLRScheduler wraps yolox.utils.LRScheduler
  module.py              # LightningYOLOX — training_step / validation_step
  datamodule.py          # YOLOXDataModule — train loader + 1-batch dummy val
  trainer.py             # LightningYOLOXTrainer.train() → pl.Trainer.fit
  data/
    mosaic.py            # BoundedMosaicDetection
    cache.py             # MemmapCOCODataset
  callbacks/
    ema.py               # wraps yolox.utils.ModelEMA; EMA-swap on validation
    multiscale.py        # rank-0 sample + dist.broadcast every N steps
    no_aug.py            # close mosaic + enable head.use_l1 at the right epoch
```

Val path is deliberately unusual: the COCO evaluator drives its own
dataloader internally, so `val_dataloader` returns a 1-sample-per-rank
`IterableDataset` and `validation_step` runs `evaluator.evaluate()` exactly
once per eval cycle. `EMACallback` swaps EMA weights in/out around the
validation window.

## Testing

```
source .venv/bin/activate
PYTHONPATH=. python -m unittest discover -s tests
```

The suite under `tests/` adds four entries that cover this package:

- `test_public_api.py` — all seven symbols importable; patch marker set on
  `yolox.exp.Exp`.
- `test_per_step_scheduler.py` — `PerStepLRScheduler` matches
  `yolox.utils.LRScheduler.update_lr(iters)` over 500 iterations.
- `test_ema_callback.py` — `ema_state_dict` round-trips through a Lightning
  checkpoint; `on_validation_start/end` swap then restore live weights.
- `test_training_smoke.py` — CPU `pl.Trainer.fit` of a YOLOX-nano-scale
  model against synthetic batches (2 steps, `EMACallback` +
  `MultiscaleCallback` both active). No COCO, no GPU.

Full suite runs in ~0.3 s.

## Not yet verified

The following require hardware this machine doesn't have:

- **Overfit test** — 1 GPU, 8 images, 200 iters on `yolox-s`; loss ↓
  monotonically, mAP > 0.
- **Short-run parity** — 2 GPU, 30 epochs on COCO; val mAP within ±1 of a
  legacy run on the same commit.
- **Full parity** — 8 GPU, 300 epochs; val mAP within ±0.3 of 40.5 on
  val2017.
- **Downstream smoke** — the other project swaps `yolox` → `yolox_lightning`
  on import lines only, runs one epoch on 2 GPU, produces a `-c`-loadable
  checkpoint.

## Out of scope for this version

- VOC dataset / `voc_eval.py`
- ONNX / TensorRT / OpenVINO / ncnn / MegEngine export tools
- `tools/visualize_assign.py`
- Multi-node launch (single-node multi-GPU only)
- Replacing the `yolox/layers/cocoeval/*` C++ eval op
