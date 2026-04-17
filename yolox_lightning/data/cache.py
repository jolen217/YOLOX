"""Zero-copy cache backend for ``COCODataset``.

The stock ``yolox.data.datasets.datasets_wrapper.cache_read_img`` decorator has
two pieces of overhead that dominate steady-state throughput once images are
cached:

* **RAM cache path**: every read ``copy.deepcopy(self.imgs[index])`` allocates
  a fresh ndarray. Our downstream mosaic pipeline immediately resizes into a
  fresh buffer via ``cv2.resize``, so the deepcopy is wasted work.
* **Disk cache path**: ``np.load(cache_file)`` fully materialises each ``.npy``
  into a private allocation. Switching to ``mmap_mode="r"`` lets the OS page
  cache back the reads, which means all dataloader workers on a rank share the
  same pages instead of each holding their own copy.

``MemmapCOCODataset`` is a drop-in subclass of ``COCODataset`` that replaces
``read_img`` with versions of both paths that avoid these costs. The resulting
arrays are consumed read-only by the rest of the pipeline; the mmap path
returns a writeable flag of ``False`` which surfaces any accidental in-place
mutation loudly instead of silently corrupting the cache.
"""

from __future__ import annotations

import os

import numpy as np

from yolox.data.datasets.coco import COCODataset

__all__ = ["MemmapCOCODataset"]


class MemmapCOCODataset(COCODataset):
    """``COCODataset`` with a mmap/no-deepcopy cache read path.

    Behaviour matches the parent class exactly when ``cache=False``. When
    ``cache_type="disk"`` the cache files are already produced by the parent
    class's ``cache_images`` routine; we only change how they are read back.
    When ``cache_type="ram"`` the parent populates ``self.imgs`` as before;
    we skip the ``deepcopy`` on the read path.
    """

    def read_img(self, index, use_cache: bool = True):
        cache = self.cache and use_cache
        if not cache:
            return self.load_resized_img(index)

        if self.cache_type == "ram":
            img = self.imgs[index]
            if img is None:
                img = self.load_resized_img(index)
            return img

        if self.cache_type == "disk":
            stem = self.path_filename[index].rsplit(".", 1)[0]
            cache_path = os.path.join(self.cache_dir, f"{stem}.npy")
            return np.load(cache_path, mmap_mode="r")

        raise ValueError(f"Unknown cache type: {self.cache_type}")
