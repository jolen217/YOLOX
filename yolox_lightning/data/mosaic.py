"""Mosaic + MixUp wrapper with a preallocated compositing buffer and bounded retries.

Functionally equivalent to ``yolox.data.datasets.MosaicDetection`` but with two
practical fixes:

1. The ``114``-filled compositing canvas (``input_h * 2`` × ``input_w * 2``) is
   allocated once per worker and reused, not freshly allocated on every sample.
2. ``mixup``'s search for an annotated partner is capped at ``max_mixup_tries``
   attempts; if exhausted, MixUp is skipped for that sample instead of spinning
   forever on a degenerate dataset.

The ``enable_mosaic`` flag is a plain attribute — toggled directly by
``NoAugToggleCallback`` — rather than being smuggled through per-batch sampler
tuples. This lets us use a stock ``DistributedSampler`` instead of
``InfiniteSampler`` + ``YoloBatchSampler``.
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset as TorchDataset

from yolox.data.data_augment import random_affine
from yolox.data.datasets.mosaicdetection import get_mosaic_coordinate
from yolox.utils import adjust_box_anns

__all__ = ["BoundedMosaicDetection"]


class BoundedMosaicDetection(TorchDataset):
    def __init__(
        self,
        dataset,
        img_size: Tuple[int, int],
        mosaic: bool = True,
        preproc=None,
        degrees: float = 10.0,
        translate: float = 0.1,
        mosaic_scale: Tuple[float, float] = (0.5, 1.5),
        mixup_scale: Tuple[float, float] = (0.5, 1.5),
        shear: float = 2.0,
        enable_mixup: bool = True,
        mosaic_prob: float = 1.0,
        mixup_prob: float = 1.0,
        max_mixup_tries: int = 8,
    ):
        super().__init__()
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = mosaic_scale
        self.shear = shear
        self.mixup_scale = mixup_scale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.max_mixup_tries = max_mixup_tries

        self._input_dim = tuple(img_size[:2])
        self._mosaic_buf: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self._dataset)

    @property
    def input_dim(self) -> Tuple[int, int]:
        return self._input_dim

    @input_dim.setter
    def input_dim(self, value: Tuple[int, int]) -> None:
        self._input_dim = tuple(value[:2])

    def _get_mosaic_canvas(self, input_h: int, input_w: int, channels: int) -> np.ndarray:
        target_shape = (input_h * 2, input_w * 2, channels)
        if self._mosaic_buf is None or self._mosaic_buf.shape != target_shape:
            self._mosaic_buf = np.empty(target_shape, dtype=np.uint8)
        self._mosaic_buf.fill(114)
        return self._mosaic_buf

    def __getitem__(self, idx: int):
        if self.enable_mosaic and random.random() < self.mosaic_prob:
            return self._mosaic_item(idx)
        return self._plain_item(idx)

    def _plain_item(self, idx: int):
        self._dataset._input_dim = self._input_dim
        img, label, img_info, img_id = self._dataset.pull_item(idx)
        if self.preproc is not None:
            img, label = self.preproc(img, label, self._input_dim)
        return img, label, img_info, img_id

    def _mosaic_item(self, idx: int):
        input_h, input_w = self._input_dim
        yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
        xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

        indices = [idx] + [
            random.randint(0, len(self._dataset) - 1) for _ in range(3)
        ]

        mosaic_labels = []
        mosaic_img: Optional[np.ndarray] = None
        last_img_id = None

        for i_mosaic, index in enumerate(indices):
            img, _labels, _, img_id = self._dataset.pull_item(index)
            last_img_id = img_id
            h0, w0 = img.shape[:2]
            scale = min(input_h / h0, input_w / w0)
            img = cv2.resize(
                img,
                (int(w0 * scale), int(h0 * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
            h, w, c = img.shape
            if i_mosaic == 0:
                mosaic_img = self._get_mosaic_canvas(input_h, input_w, c)

            (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
            )
            mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
            padw, padh = l_x1 - s_x1, l_y1 - s_y1

            labels = _labels.copy()
            if _labels.size > 0:
                labels[:, 0] = scale * _labels[:, 0] + padw
                labels[:, 1] = scale * _labels[:, 1] + padh
                labels[:, 2] = scale * _labels[:, 2] + padw
                labels[:, 3] = scale * _labels[:, 3] + padh
            mosaic_labels.append(labels)

        if mosaic_labels:
            mosaic_labels = np.concatenate(mosaic_labels, 0)
            np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
            np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
            np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
            np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])
        else:
            mosaic_labels = np.zeros((0, 5), dtype=np.float32)

        # random_affine re-samples into a fresh (input_h, input_w) array, so the
        # preallocated canvas survives for the next call even if we hand this
        # ndarray to downstream code that mutates it.
        mosaic_img, mosaic_labels = random_affine(
            mosaic_img,
            mosaic_labels,
            target_size=(input_w, input_h),
            degrees=self.degrees,
            translate=self.translate,
            scales=self.scale,
            shear=self.shear,
        )

        if (
            self.enable_mixup
            and len(mosaic_labels) > 0
            and random.random() < self.mixup_prob
        ):
            mixed = self._mixup(mosaic_img, mosaic_labels, self._input_dim)
            if mixed is not None:
                mosaic_img, mosaic_labels = mixed

        mix_img, padded_labels = self.preproc(
            mosaic_img, mosaic_labels, self._input_dim
        )
        img_info = (mix_img.shape[1], mix_img.shape[0])
        return mix_img, padded_labels, img_info, last_img_id

    def _mixup(
        self, origin_img: np.ndarray, origin_labels: np.ndarray, input_dim: Tuple[int, int]
    ):
        cp_labels = np.zeros((0, 5), dtype=np.float32)
        cp_index = 0
        for _ in range(self.max_mixup_tries):
            cp_index = random.randint(0, len(self._dataset) - 1)
            cand = self._dataset.load_anno(cp_index)
            if cand is not None and len(cand) > 0:
                cp_labels = cand
                break
        else:
            return None  # every candidate was empty; skip MixUp for this sample

        jit_factor = random.uniform(*self.mixup_scale)
        flip = random.uniform(0, 1) > 0.5

        img, cp_labels, _, _ = self._dataset.pull_item(cp_index)

        if img.ndim == 3:
            cp_img = np.full((input_dim[0], input_dim[1], 3), 114, dtype=np.uint8)
        else:
            cp_img = np.full(input_dim, 114, dtype=np.uint8)

        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )
        cp_img[: resized_img.shape[0], : resized_img.shape[1]] = resized_img

        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor

        if flip:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
        )
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset = 0
        y_offset = 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
            y_offset : y_offset + target_h, x_offset : x_offset + target_w
        ]

        cp_bboxes = adjust_box_anns(
            cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if flip:
            cp_bboxes[:, 0::2] = origin_w - cp_bboxes[:, 0::2][:, ::-1]
        cp_bboxes[:, 0::2] = np.clip(cp_bboxes[:, 0::2] - x_offset, 0, target_w)
        cp_bboxes[:, 1::2] = np.clip(cp_bboxes[:, 1::2] - y_offset, 0, target_h)

        labels = np.hstack((cp_bboxes, cp_labels[:, 4:5].copy()))
        merged_labels = np.vstack((origin_labels, labels))
        mixed_img = (origin_img.astype(np.float32) * 0.5 + padded_cropped_img.astype(np.float32) * 0.5).astype(np.uint8)
        return mixed_img, merged_labels
