"""Small torchrun/DDP compatibility layer."""

import os
from dataclasses import dataclass
from typing import Mapping, Optional

import torch
import torch.distributed as distributed
from torch import nn
from torch.nn.parallel import DistributedDataParallel


@dataclass
class DistributedContext:
    """Initialized process identity and device."""

    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    enabled: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class TrainingLaunch:
    """Resolved process-launch mode from the requested visible GPU count."""

    mode: str
    num_gpus: int
    device: str
    spawn: bool


def resolve_training_launch(
    requested_gpus: Optional[int],
    requested_device: str = "",
    environ: Optional[Mapping[str, str]] = None,
    cuda_available: Optional[bool] = None,
    cuda_device_count: Optional[int] = None,
) -> TrainingLaunch:
    """Choose CPU, one GPU, internal DDP, or an existing torchrun job.

    ``requested_gpus`` counts GPUs visible to this process, normally after
    applying ``CUDA_VISIBLE_DEVICES``. A value greater than one starts one DDP
    process per GPU when the script is not already running under torchrun.
    """
    environment = os.environ if environ is None else environ
    world_size = int(environment.get("WORLD_SIZE", "1"))
    local_world_size = int(
        environment.get("LOCAL_WORLD_SIZE", str(world_size))
    )
    available = (
        torch.cuda.is_available()
        if cuda_available is None
        else bool(cuda_available)
    )
    device_count = (
        torch.cuda.device_count()
        if cuda_device_count is None
        else int(cuda_device_count)
    )
    if requested_gpus is not None and int(requested_gpus) < 0:
        raise ValueError("--num-gpus must be zero or a positive integer")
    gpu_count = None if requested_gpus is None else int(requested_gpus)
    if requested_device == "cpu" and gpu_count not in (None, 0):
        raise ValueError("--device cpu cannot be combined with --num-gpus > 0")
    if requested_device == "cuda" and gpu_count == 0:
        raise ValueError("--device cuda cannot be combined with --num-gpus 0")

    if world_size > 1:
        if gpu_count is not None and gpu_count > 0 and gpu_count != local_world_size:
            raise ValueError(
                "--num-gpus={} does not match torchrun LOCAL_WORLD_SIZE={}".format(
                    gpu_count, local_world_size
                )
            )
        use_cpu = requested_device == "cpu" or gpu_count == 0
        if use_cpu:
            return TrainingLaunch("external_ddp", 0, "cpu", False)
        if not available:
            if requested_device != "cuda" and gpu_count is None:
                return TrainingLaunch("external_ddp", 0, "cpu", False)
            raise RuntimeError("CUDA DDP requested but CUDA is not available")
        selected = local_world_size if gpu_count is None else gpu_count
        if selected > device_count:
            raise ValueError(
                "Requested {} GPUs but only {} are visible".format(
                    selected, device_count
                )
            )
        return TrainingLaunch("external_ddp", selected, "cuda", False)

    if gpu_count is None:
        if requested_device == "cpu":
            gpu_count = 0
        elif available:
            gpu_count = 1
        elif requested_device == "cuda":
            raise RuntimeError("CUDA requested but CUDA is not available")
        else:
            gpu_count = 0
    if gpu_count == 0:
        return TrainingLaunch("cpu", 0, "cpu", False)
    if not available:
        raise RuntimeError(
            "{} GPU(s) requested but CUDA is not available".format(gpu_count)
        )
    if gpu_count > device_count:
        raise ValueError(
            "Requested {} GPUs but only {} are visible".format(
                gpu_count, device_count
            )
        )
    if gpu_count == 1:
        return TrainingLaunch("single_gpu", 1, "cuda", False)
    return TrainingLaunch("internal_ddp", gpu_count, "cuda", True)


def initialize_distributed(requested_device: str = "") -> DistributedContext:
    """Initialize from torchrun environment, using NCCL on CUDA and Gloo on CPU."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available() and requested_device != "cpu"
    if use_cuda:
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    enabled = world_size > 1
    if enabled and not distributed.is_initialized():
        distributed.init_process_group(backend="nccl" if use_cuda else "gloo")
    return DistributedContext(rank, world_size, local_rank, device, enabled)


def wrap_model(
    model: nn.Module,
    context: DistributedContext,
    data_parallel: bool = False,
) -> nn.Module:
    """Wrap model in DDP, or optional legacy DataParallel for one process."""
    model = model.to(context.device)
    if context.enabled:
        device_ids = [context.local_rank] if context.device.type == "cuda" else None
        return DistributedDataParallel(model, device_ids=device_ids)
    if data_parallel and context.device.type == "cuda" and torch.cuda.device_count() > 1:
        return nn.DataParallel(model)
    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return underlying module for DDP/DataParallel models."""
    return model.module if hasattr(model, "module") else model


def reduce_mean(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    """All-reduce a scalar mean."""
    if not context.enabled:
        return value
    result = value.detach().clone()
    distributed.all_reduce(result, op=distributed.ReduceOp.SUM)
    return result / context.world_size


def barrier(context: DistributedContext) -> None:
    if context.enabled:
        distributed.barrier()


def cleanup_distributed(context: DistributedContext) -> None:
    if context.enabled and distributed.is_initialized():
        distributed.destroy_process_group()
