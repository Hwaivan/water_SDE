"""Atomic SGMSE checkpoint with EMA and RNG state."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn

from .ema import ExponentialMovingAverage
from .seed import capture_rng_state, restore_rng_state


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    """Atomically persist checkpoint state."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(state, str(temporary))
    os.replace(str(temporary), str(target))


def training_state(
    model: nn.Module,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Any,
    epoch: int,
    global_step: int,
    best_metric: float,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Construct the complete required checkpoint payload."""
    return {
        "model": model.state_dict(),
        "ema_model": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": float(best_metric),
        "config": config,
        "rng_state": capture_rng_state(),
    }


def restore_training_state(
    checkpoint_path: str,
    model: nn.Module,
    ema: ExponentialMovingAverage,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: str = "cpu",
    restore_rng: bool = True,
) -> Dict[str, Any]:
    """Restore model, EMA, optimizer, scheduler, scaler, and optional RNG."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    ema.load_state_dict(checkpoint["ema_model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if restore_rng and checkpoint.get("rng_state") is not None:
        restore_rng_state(checkpoint["rng_state"])
    return checkpoint


def load_model_for_inference(
    checkpoint_path: str,
    model: nn.Module,
    device: str,
    use_ema: bool = True,
) -> Dict[str, Any]:
    """Load EMA shadow or raw parameters into an inference model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if use_ema:
        model.load_state_dict(checkpoint["ema_model"]["shadow"], strict=True)
    else:
        model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint

