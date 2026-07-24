"""Exponential moving average for stable diffusion evaluation."""

from contextlib import contextmanager
from typing import Dict, Iterator

import torch
from torch import nn


class ExponentialMovingAverage:
    """Maintain a floating-point shadow copy of model parameters and buffers."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0,1)")
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow values after an optimizer step."""
        for name, value in model.state_dict().items():
            shadow = self.shadow[name]
            if torch.is_floating_point(value):
                shadow.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                shadow.copy_(value)

    def state_dict(self) -> Dict[str, object]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.decay = float(state["decay"])
        self.shadow = {
            name: value.detach().clone()
            for name, value in state["shadow"].items()
        }

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        """Temporarily swap a model to EMA parameters."""
        backup = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        model.load_state_dict(self.shadow, strict=True)
        try:
            yield
        finally:
            model.load_state_dict(backup, strict=True)

