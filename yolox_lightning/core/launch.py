"""Distributed training launcher.

Spawns one worker per GPU, initialises ``torch.distributed`` (NCCL), populates the
rank / world-size environment variables Lightning's ``DDPStrategy`` expects, and
calls ``main_func(*args)`` in each worker. Because Lightning finds the process
group already initialised, it does not spawn a second layer of workers.
"""

import os
import socket
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

__all__ = ["launch"]


_DEFAULT_TIMEOUT = timedelta(minutes=30)


def _find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def launch(
    main_func,
    num_gpus_per_machine,
    num_machines=1,
    machine_rank=0,
    backend="nccl",
    dist_url=None,
    args=(),
    timeout=_DEFAULT_TIMEOUT,
):
    """Drop-in replacement for ``yolox.core.launch``.

    Args:
        main_func: callable invoked as ``main_func(*args)`` in each worker.
        num_gpus_per_machine: GPUs on this machine.
        num_machines: total machines participating.
        machine_rank: rank of this machine (0-based).
        backend: distributed backend, typically ``"nccl"``.
        dist_url: ``tcp://host:port`` rendezvous URL. ``"auto"`` or ``None`` picks a free
            local port (single-machine only).
        args: tuple of positional arguments forwarded to ``main_func``.
        timeout: process-group init timeout.
    """
    world_size = num_machines * num_gpus_per_machine

    if world_size <= 1:
        main_func(*args)
        return

    if dist_url is None or dist_url == "auto":
        assert num_machines == 1, "dist_url='auto' is only valid for single-machine training"
        port = _find_free_port()
        dist_url = f"tcp://127.0.0.1:{port}"

    mp.spawn(
        _worker,
        nprocs=num_gpus_per_machine,
        args=(
            main_func,
            world_size,
            num_gpus_per_machine,
            machine_rank,
            backend,
            dist_url,
            args,
            timeout,
        ),
        daemon=False,
    )


def _worker(
    local_rank,
    main_func,
    world_size,
    num_gpus_per_machine,
    machine_rank,
    backend,
    dist_url,
    main_args,
    timeout,
):
    global_rank = machine_rank * num_gpus_per_machine + local_rank

    # Populate env vars before init_process_group so Lightning's LightningEnvironment
    # picks up the same values and does not try to re-spawn.
    host, port = _parse_tcp_url(dist_url)
    os.environ["MASTER_ADDR"] = host
    os.environ["MASTER_PORT"] = port
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(global_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["NODE_RANK"] = str(machine_rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(num_gpus_per_machine)

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend=backend,
        init_method=dist_url,
        world_size=world_size,
        rank=global_rank,
        timeout=timeout,
    )
    dist.barrier(device_ids=[local_rank] if backend == "nccl" else None)

    try:
        main_func(*main_args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _parse_tcp_url(url):
    assert url.startswith("tcp://"), f"dist_url must start with tcp://, got {url!r}"
    host, _, port = url[len("tcp://"):].rpartition(":")
    return host, port
