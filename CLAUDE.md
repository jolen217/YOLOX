# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Install

```shell
pip3 install -v -e .
```

`setup.py` pre-compiles the `FastCOCOEvalOp` C++ extension on non-Windows platforms via `torch.utils.cpp_extension`, so a working CUDA/PyTorch toolchain must be available at install time.

## Common commands

All training/evaluation entrypoints accept either `-n <name>` (e.g. `yolox-s`, which resolves to `exps/default/yolox_s.py` via `get_exp_by_name`) or `-f <path>` to a custom Exp file. They're interchangeable.

```shell
# Train (uses yolox.core.launch for multi-GPU/multi-node spawning)
python -m yolox.tools.train -n yolox-s -d 8 -b 64 --fp16 -o [--cache ram|disk] [-c ckpt.pth] [--resume]

# Evaluate
python -m yolox.tools.eval -n yolox-s -c yolox_s.pth -b 64 -d 8 --conf 0.001 [--fp16] [--fuse]

# Demo inference (image/video/webcam)
python tools/demo.py image -n yolox-s -c yolox_s.pth --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result

# Export
python tools/export_onnx.py -n yolox-s -c yolox_s.pth --output-name yolox_s.onnx
python tools/export_torchscript.py -n yolox-s -c yolox_s.pth
python tools/trt.py -n yolox-s -c yolox_s.pth

# Visualize label assignment (SimOTA) on a trained model
python tools/visualize_assign.py -f exps/default/yolox_s.py -c yolox_s.pth -d 1 -b 8 --max-batch 1
```

The `opts` trailing argument on train/eval is parsed by `BaseExp.merge` in `yolox/exp/base_exp.py` — pairs of `key value` tokens override Exp attributes, with types coerced from the existing attribute. Example: `... max_epoch 50 basic_lr_per_img 0.00015625`.

Dataset layout: symlink datasets into `./datasets/` (e.g. `./datasets/COCO`, `./datasets/VOCdevkit`). `Exp.data_dir = None` resolves to this directory.

## Tests and lint

```shell
python -m unittest discover -s tests     # unit tests (tests/utils/…)
flake8 yolox tools exps                  # configured in setup.cfg, max-line-length=100
isort -rc yolox tools exps               # config in setup.cfg (custom section ordering)
```

CI (`.github/workflows/ci.yaml`) runs `./.github/workflows/format_check.sh` only, against Python 3.8/3.9/3.10. Pre-commit hooks live in `.pre-commit-config.yaml` (flake8, isort, autoflake, trailing whitespace, gitlint).

To run a single test:

```shell
python -m unittest tests.utils.test_model_utils.TestModelUtils.test_freeze_module
```

## Architecture

YOLOX's central design idea: **one `Exp` object owns the entire recipe** — model, dataset, dataloader, optimizer, LR scheduler, evaluator, and trainer factory. Everything composes through the Exp instance. Users customize by subclassing.

### Exp system (`yolox/exp/`)

- `base_exp.py::BaseExp` — abstract interface (`get_model`, `get_dataset`, `get_data_loader`, `get_optimizer`, `get_lr_scheduler`, `get_evaluator`, `eval`) plus `merge(cfg_list)` for CLI overrides.
- `yolox_base.py::Exp` — the concrete default. All hyperparameters (depth/width, aug probs, LR schedule, EMA, eval cadence, test size/conf/nms) are plain attributes set in `__init__`. Subclasses in `exps/default/` (yolox_s/m/l/x/nano/tiny/yolov3) typically only change `depth`, `width`, `input_size`, `random_size`, and aug strengths.
- `build.py::get_exp` — resolves either `-n yolox-s` (imports `yolox.exp.default.yolox_s`) or `-f path/to/exp.py` (imports the module directly, expects a class `Exp`).
- `exps/example/` — templates for custom datasets (`custom/`) and VOC (`yolox_voc/`); custom Exps override `get_data_loader`/`get_eval_loader`/`get_evaluator`.

To add a new config, create `exps/example/yourname.py` with `class Exp(MyExp)` inheriting from `yolox.exp.Exp` and overriding only what differs. Do **not** edit `yolox_base.py` for per-experiment tweaks.

### Training pipeline (`yolox/core/`)

- `launch.py::launch` — detectron2-style multi-process launcher that spawns `main_func` across GPUs/nodes using `torch.multiprocessing` + NCCL, wiring rank/world-size via `yolox.utils.dist`.
- `trainer.py::Trainer` — the hook-structured training loop (`before_train` / `train_in_epoch` → `before_epoch` / `train_in_iter` → `train_one_iter` / `after_*`). It pulls model/optimizer/loader/scheduler/evaluator from `self.exp`, wraps the model in DDP when distributed, maintains an EMA copy (`ModelEMA`), and uses `DataPrefetcher` for async CUDA copy. Mosaic/MixUp aug is disabled in the last `exp.no_aug_epochs` and the L1 loss is turned on via `YOLOXHead.use_l1` at the same boundary (see `before_epoch`). `Trainer` must not be stored on the Exp.

### Model (`yolox/models/`)

Anchor-free detector composed as `YOLOX(backbone, head)`:

- `darknet.py` — CSPDarknet / Darknet53 backbones.
- `yolo_pafpn.py::YOLOPAFPN` — default neck (CSPDarknet + PAN). `yolo_fpn.py::YOLOFPN` is the Darknet53 variant.
- `yolo_head.py::YOLOXHead` — decoupled cls/reg/obj heads, dynamic SimOTA assignment (`get_assignments`), returns `(loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg)` in training and decoded predictions in eval.
- `network_blocks.py` — shared building blocks (BaseConv, DWConv, Focus, SPPBottleneck, CSPLayer), activation switched by `exp.act`.
- `losses.py` — IoU loss variants.
- `build.py` — functions exposed through `hubconf.py` for `torch.hub.load("Megvii-BaseDetection/YOLOX", "yolox_s")`.

### Data (`yolox/data/`)

- `datasets/coco.py` / `datasets/voc.py` — base datasets; both implement `pull_item`, `load_anno`, and `__getitem__` decorated with `Dataset.resize_getitem` (from `datasets_wrapper.py`) so input-resolution can be mutated per-iter.
- `datasets/mosaicdetection.py::MosaicDetection` — wraps a base dataset to produce Mosaic+MixUp samples; disables itself when the batch sampler's `mosaic` flag is False.
- `data_augment.py` — `TrainTransform` / `ValTransform` (HSV, flip, letterbox, normalization).
- `samplers.py::InfiniteSampler` + `YoloBatchSampler` — sampler stack used in `Exp.get_data_loader`; `YoloBatchSampler.mosaic` is toggled by `Trainer` around `no_aug_epochs`.
- `data_prefetcher.py::DataPrefetcher` — CUDA stream-based prefetcher used only on the train loop.

### Evaluation (`yolox/evaluators/`)

- `coco_evaluator.py::COCOEvaluator` — default; uses `FastCOCOEvalOp` (C++) from `yolox/layers/` for faster pycocotools evaluation.
- `voc_evaluator.py` + `voc_eval.py` — PASCAL VOC mAP.

### Utilities worth knowing

- `yolox/utils/dist.py` — rank/world-size helpers (`get_rank`, `get_local_rank`, `synchronize`, `wait_for_the_master`).
- `yolox/utils/ema.py::ModelEMA` — EMA of model weights, updated each iter; the EMA model is what gets evaluated/saved as "best".
- `yolox/utils/lr_scheduler.py::LRScheduler` — schedules keyed by name (`"yoloxwarmcos"` default): warmup → cosine → flat during `no_aug_epochs`.
- `yolox/utils/model_utils.py` — `freeze_module(model, name)` freezes submodules by name prefix; used for transfer-learning (see `docs/freeze_module.md`).
- `yolox/utils/logger.py` + `mlflow_logger.py` — tensorboard (default), `wandb` and `mlflow` loggers selected via `--logger` CLI flag.
- `yolox/layers/jit_ops.py` + `fast_coco_eval_api.py` — runtime JIT compilation of C++ ops when the pre-compiled extension is unavailable.

## Deployment

Top-level `demo/` holds reference implementations for ONNXRuntime, TensorRT, OpenVINO, ncnn, MegEngine, and nebullvm (C++ and Python where applicable). Each has its own README; the Python exports above feed these pipelines.
