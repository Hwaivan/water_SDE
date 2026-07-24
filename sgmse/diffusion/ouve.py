"""Ornstein-Uhlenbeck variance-exploding conditional SDE."""

import math
from typing import Optional, Sequence, Tuple

import torch


def _batch_broadcast(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Broadcast ``[B]`` values over all non-batch dimensions of reference."""
    if value.ndim != 1 or value.shape[0] != reference.shape[0]:
        raise ValueError("Time/scalar tensor must have shape [B]")
    return value.reshape((value.shape[0],) + (1,) * (reference.ndim - 1))


def complex_gaussian(
    shape: Sequence[int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample circular complex Gaussian with ``E[|z|^2] = 1``."""
    real = torch.randn(tuple(shape), device=device, dtype=dtype, generator=generator)
    imag = torch.randn(tuple(shape), device=device, dtype=dtype, generator=generator)
    return torch.complex(real, imag) / math.sqrt(2.0)


class OUVESDE:
    """Conditional OUVE SDE from the SGMSE family.

    Forward process:
        ``dx = gamma * (y - x) dt + g(t) dw``.

    The reverse drift returned here is ``f - g^2 score``. Samplers integrate it
    on a decreasing time grid, so their Euler ``dt`` is negative.
    """

    def __init__(
        self,
        T: float = 1.0,
        t_eps: float = 0.03,
        gamma: float = 1.5,
        sigma_min: float = 0.05,
        sigma_max: float = 0.5,
        variance_floor: float = 1.0e-12,
    ) -> None:
        if not sigma_max > sigma_min > 0.0:
            raise ValueError("Require sigma_max > sigma_min > 0")
        if gamma <= 0.0 or not 0.0 <= t_eps < T:
            raise ValueError("Require gamma > 0 and 0 <= t_eps < T")
        self.T = float(T)
        self.t_eps = float(t_eps)
        self.gamma = float(gamma)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.log_ratio = math.log(self.sigma_max / self.sigma_min)
        self.variance_floor = float(variance_floor)

    def drift(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return forward drift ``gamma * (y - x)``; ``t`` validates batch shape."""
        _batch_broadcast(t, x)
        if x.shape != y.shape:
            raise ValueError("x and y must have equal shape")
        return self.gamma * (y - x)

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """Return scalar diffusion coefficient ``g(t)`` for each batch item."""
        if t.ndim != 1:
            raise ValueError("t must have shape [B]")
        ratio = self.sigma_max / self.sigma_min
        return self.sigma_min * ratio ** t * math.sqrt(2.0 * self.log_ratio)

    def mean(
        self, x0: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Closed marginal mean with batch-wise time broadcasting."""
        if x0.shape != y.shape:
            raise ValueError("x0 and y must have equal shape")
        interpolation = _batch_broadcast(torch.exp(-self.gamma * t), x0)
        return interpolation * x0 + (1.0 - interpolation) * y

    def variance(self, t: torch.Tensor) -> torch.Tensor:
        """Closed marginal variance requested by the integration specification.

        ``sigma_min^2 * (r^(2t)-exp(-2 gamma t))/(gamma+log(r))``.
        Negative round-off is clamped to ``variance_floor``.
        """
        if t.ndim != 1:
            raise ValueError("t must have shape [B]")
        ratio = self.sigma_max / self.sigma_min
        numerator = ratio ** (2.0 * t) - torch.exp(-2.0 * self.gamma * t)
        variance = self.sigma_min ** 2 * numerator / (self.gamma + self.log_ratio)
        return variance.clamp_min(self.variance_floor)

    def std(self, t: torch.Tensor) -> torch.Tensor:
        """Marginal standard deviation ``[B]``."""
        return torch.sqrt(self.variance(t))

    def marginal_prob(
        self, x0: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return marginal mean ``x0.shape`` and std ``[B]``."""
        return self.mean(x0, y, t), self.std(t)

    def prior_sampling(
        self,
        y: torch.Tensor,
        shape: Sequence[int],
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Sample ``X_T = y + std(T) * CN(0,I)``."""
        if tuple(shape) != tuple(y.shape):
            raise ValueError("Prior shape must match condition shape")
        time = torch.full((y.shape[0],), self.T, device=y.device, dtype=y.real.dtype)
        std = _batch_broadcast(self.std(time), y)
        noise = complex_gaussian(y.shape, y.device, y.real.dtype, generator)
        return y + std * noise

    def reverse_drift(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor,
    ) -> torch.Tensor:
        """Return reverse-SDE drift ``f - g(t)^2 score``."""
        if score.shape != x.shape:
            raise ValueError("score and state must have equal shape")
        diffusion = _batch_broadcast(self.diffusion(t), x)
        return self.drift(x, y, t) - diffusion.square() * score

    def probability_flow_drift(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor,
    ) -> torch.Tensor:
        """Return probability-flow ODE drift ``f - 0.5*g(t)^2 score``."""
        diffusion = _batch_broadcast(self.diffusion(t), x)
        return self.drift(x, y, t) - 0.5 * diffusion.square() * score

