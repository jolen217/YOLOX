import os
import subprocess

__all__ = ["configure_module", "configure_nccl", "configure_omp", "get_num_devices"]


def configure_nccl():
    """Set NCCL environment variables for multi-node training over InfiniBand."""
    os.environ.setdefault("NCCL_LAUNCH_MODE", "PARALLEL")
    try:
        ib_hca = subprocess.getoutput(
            "pushd /sys/class/infiniband/ > /dev/null; "
            "for i in mlx5_*; do "
            "cat $i/ports/1/gid_attrs/types/* 2>/dev/null | grep v >/dev/null && echo $i; "
            "done; "
            "popd > /dev/null"
        )
        if ib_hca:
            os.environ.setdefault("NCCL_IB_HCA", ib_hca)
    except Exception:
        pass
    os.environ.setdefault("NCCL_IB_GID_INDEX", "3")
    os.environ.setdefault("NCCL_IB_TC", "106")


def configure_omp(num_threads=1):
    """Set ``OMP_NUM_THREADS`` if it is not already set. Defaults to 1 for best dataloader throughput."""
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))


def configure_module(ulimit_value=8192):
    """Raise the open-file limit and disable OpenCV threading / OpenCL to avoid dataloader contention."""
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = max(soft, ulimit_value)
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except Exception:
        pass

    os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
    try:
        import cv2

        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass


def get_num_devices():
    """Return the number of visible CUDA devices.

    Respects ``CUDA_VISIBLE_DEVICES`` even before ``torch.cuda`` is initialised.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() != "":
        return len([d for d in cvd.split(",") if d.strip() != ""])
    import torch

    return torch.cuda.device_count()
