"""NCSN++ building blocks adapted for rectangular complex spectrograms.

The design follows the official SGMSE NCSN++ port and its Google Research
score-SDE ancestry. See ``THIRD_PARTY_NOTICES.md`` for exact source blobs.
"""

import math
from typing import Optional, Sequence

import torch
import torch.nn.functional as functional
from torch import nn


def group_count(channels: int, maximum: int = 32) -> int:
    """Choose the largest valid GroupNorm divisor up to ``maximum``."""
    for groups in reversed(range(1, min(maximum, channels) + 1)):
        if channels % groups == 0:
            return groups
    return 1


class GaussianFourierProjection(nn.Module):
    """Fixed Gaussian Fourier features for continuous diffusion time."""

    def __init__(self, embedding_size: int, scale: float = 16.0) -> None:
        super().__init__()
        self.register_buffer(
            "weight",
            torch.randn(embedding_size) * float(scale),
            persistent=True,
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        if time.ndim != 1:
            raise ValueError("time must have shape [B]")
        projected = time[:, None] * self.weight[None, :] * (2.0 * math.pi)
        return torch.cat((torch.sin(projected), torch.cos(projected)), dim=-1)


class FIRFilter2d(nn.Module):
    """Depthwise low-pass FIR filter using a separable configured kernel."""

    def __init__(self, kernel: Sequence[float] = (1, 3, 3, 1)) -> None:
        super().__init__()
        vector = torch.tensor(list(kernel), dtype=torch.float32)
        matrix = vector[:, None] * vector[None, :]
        matrix = matrix / matrix.sum()
        self.register_buffer("kernel", matrix[None, None], persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        channels = inputs.shape[1]
        kernel = self.kernel.to(device=inputs.device, dtype=inputs.dtype)
        kernel = kernel.expand(channels, 1, -1, -1)
        size = kernel.shape[-1]
        left = (size - 1) // 2
        right = size - 1 - left
        inputs = functional.pad(inputs, (left, right, left, right))
        return functional.conv2d(inputs, kernel, groups=channels)


class FIRDownsample2d(nn.Module):
    """FIR low-pass followed by factor-two average downsampling."""

    def __init__(self, kernel: Sequence[float] = (1, 3, 3, 1)) -> None:
        super().__init__()
        self.filter = FIRFilter2d(kernel)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        filtered = self.filter(inputs)
        return functional.avg_pool2d(filtered, kernel_size=2, stride=2)


class FIRUpsample2d(nn.Module):
    """Nearest factor-two upsampling followed by FIR low-pass filtering."""

    def __init__(self, kernel: Sequence[float] = (1, 3, 3, 1)) -> None:
        super().__init__()
        self.filter = FIRFilter2d(kernel)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        upsampled = functional.interpolate(inputs, scale_factor=2.0, mode="nearest")
        return self.filter(upsampled)


class BigGANResBlock(nn.Module):
    """BigGAN-style time-conditioned residual block used by NCSN++.

    Input/output are real feature maps ``[B,C,F,N]``. Optional ``up`` or
    ``down`` performs FIR resampling on both residual branches.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        dropout: float = 0.0,
        up: bool = False,
        down: bool = False,
        fir_kernel: Sequence[float] = (1, 3, 3, 1),
        skip_rescale: bool = True,
    ) -> None:
        super().__init__()
        if up and down:
            raise ValueError("Residual block cannot upsample and downsample")
        self.norm1 = nn.GroupNorm(group_count(in_channels), in_channels, eps=1.0e-6)
        self.norm2 = nn.GroupNorm(group_count(out_channels), out_channels, eps=1.0e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Linear(time_channels, out_channels)
        self.dropout = nn.Dropout(float(dropout))
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels or up or down
            else nn.Identity()
        )
        self.up = bool(up)
        self.down = bool(down)
        self.upsample = FIRUpsample2d(fir_kernel)
        self.downsample = FIRDownsample2d(fir_kernel)
        self.skip_rescale = bool(skip_rescale)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, inputs: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        residual = inputs
        hidden = functional.silu(self.norm1(inputs))
        if self.up:
            hidden = self.upsample(hidden)
            residual = self.upsample(residual)
        elif self.down:
            hidden = self.downsample(hidden)
            residual = self.downsample(residual)
        hidden = self.conv1(hidden)
        hidden = hidden + self.time_projection(functional.silu(time_embedding))[:, :, None, None]
        hidden = self.conv2(self.dropout(functional.silu(self.norm2(hidden))))
        residual = self.skip(residual)
        output = residual + hidden
        return output / math.sqrt(2.0) if self.skip_rescale else output


class AttentionBlock(nn.Module):
    """Low-resolution spatial self-attention over ``F x Frames`` positions."""

    def __init__(self, channels: int, skip_rescale: bool = True) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(group_count(channels), channels, eps=1.0e-6)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.projection = nn.Conv2d(channels, channels, 1)
        self.skip_rescale = bool(skip_rescale)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = inputs.shape
        q, k, v = self.qkv(self.norm(inputs)).chunk(3, dim=1)
        q = q.reshape(batch, channels, height * width).transpose(1, 2)
        k = k.reshape(batch, channels, height * width)
        weights = torch.bmm(q, k) * channels ** -0.5
        weights = torch.softmax(weights, dim=-1)
        v = v.reshape(batch, channels, height * width)
        hidden = torch.bmm(v, weights.transpose(1, 2)).reshape(
            batch, channels, height, width
        )
        output = inputs + self.projection(hidden)
        return output / math.sqrt(2.0) if self.skip_rescale else output


def align_spatial(inputs: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Crop/right-pad features to reference frequency/frame dimensions."""
    height, width = reference.shape[-2:]
    inputs = inputs[..., :height, :width]
    return functional.pad(
        inputs,
        (0, max(0, width - inputs.shape[-1]), 0, max(0, height - inputs.shape[-2])),
    )

