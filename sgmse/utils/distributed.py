"""Small torchrun/DDP compatibility layer."""

import os
from dataclasses import dataclass

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

