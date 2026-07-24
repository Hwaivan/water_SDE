"""Predictor-corrector reverse diffusion and long-audio overlap-add."""

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as functional
from torch import nn

from sgmse.utils.stft import (
    ComplexSTFTTransform,
    channels_to_complex,
    complex_to_channels,
)
from .ouve import OUVESDE, complex_gaussian


@dataclass
class SamplingResult:
    """One sampling result with runtime diagnostics."""

    waveform: torch.Tensor
    spectrum: torch.Tensor
    nfe: int
    inference_time: float
    rtf: float


class PredictorCorrectorSampler:
    """Euler-Maruyama predictor with optional annealed Langevin corrector."""

    def __init__(
        self,
        sde: OUVESDE,
        transform: ComplexSTFTTransform,
        num_steps: int = 30,
        predictor: str = "euler_maruyama",
        corrector: str = "annealed_langevin",
        corrector_steps: int = 1,
        corrector_step_size: float = 0.5,
    ) -> None:
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        if predictor != "euler_maruyama":
            raise ValueError("Only euler_maruyama predictor is implemented")
        if corrector not in ("annealed_langevin", "none"):
            raise ValueError("corrector must be annealed_langevin or none")
        self.sde = sde
        self.transform = transform
        self.num_steps = int(num_steps)
        self.predictor = predictor
        self.corrector = corrector
        self.corrector_steps = int(corrector_steps)
        self.corrector_step_size = float(corrector_step_size)

    @staticmethod
    def _score(
        model: nn.Module,
        state: torch.Tensor,
        condition: torch.Tensor,
        time_value: torch.Tensor,
    ) -> torch.Tensor:
        network_input = torch.cat(
            (complex_to_channels(state), complex_to_channels(condition)), dim=1
        )
        channels = model(network_input, time_value)
        return channels_to_complex(channels.float())

    @torch.inference_mode()
    def sample_spectrum(
        self,
        model: nn.Module,
        condition: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Sample compressed clean spectrum conditioned on compressed noisy STFT."""
        if condition.ndim != 3 or not torch.is_complex(condition):
            raise ValueError("condition must be complex [B,F,N]")
        batch = condition.shape[0]
        state = self.sde.prior_sampling(condition, condition.shape, generator)
        grid = torch.linspace(
            self.sde.T,
            self.sde.t_eps,
            self.num_steps + 1,
            device=condition.device,
            dtype=condition.real.dtype,
        )
        nfe = 0
        for index in range(self.num_steps):
            current = grid[index]
            following = grid[index + 1]
            time_batch = torch.full(
                (batch,), current, device=condition.device, dtype=condition.real.dtype
            )
            if self.corrector == "annealed_langevin":
                for _ in range(self.corrector_steps):
                    score = self._score(model, state, condition, time_batch)
                    nfe += 1
                    std = self.sde.std(time_batch).reshape(batch, 1, 1)
                    step = self.corrector_step_size * std.square()
                    noise = complex_gaussian(
                        state.shape, state.device, state.real.dtype, generator
                    )
                    state = (
                        state
                        + step * score
                        + torch.sqrt(2.0 * step) * noise
                    )
            score = self._score(model, state, condition, time_batch)
            nfe += 1
            delta = following - current  # negative: reverse integration T -> t_eps
            drift = self.sde.reverse_drift(state, condition, time_batch, score)
            mean_state = state + drift * delta
            if index == self.num_steps - 1:
                state = mean_state
            else:
                diffusion = self.sde.diffusion(time_batch).reshape(batch, 1, 1)
                noise = complex_gaussian(
                    state.shape, state.device, state.real.dtype, generator
                )
                state = mean_state + diffusion * torch.sqrt(-delta) * noise
            if not torch.isfinite(state.real).all() or not torch.isfinite(state.imag).all():
                raise FloatingPointError(
                    "Non-finite reverse diffusion state at step {}".format(index)
                )
        return state, nfe

    @torch.inference_mode()
    def sample_waveform(
        self,
        model: nn.Module,
        noisy: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        sample_rate: int = 16000,
        generator: Optional[torch.Generator] = None,
    ) -> SamplingResult:
        """Enhance aligned noisy waveforms ``[B,T]`` and preserve padded length."""
        if noisy.ndim != 2:
            raise ValueError("noisy must have shape [B,T]")
        started = time.perf_counter()
        condition = self.transform.encode(noisy.float())
        estimate, nfe = self.sample_spectrum(model, condition, generator)
        waveform = self.transform.decode(estimate, noisy.shape[-1])
        if lengths is not None:
            valid = (
                torch.arange(noisy.shape[-1], device=noisy.device)[None, :]
                < lengths[:, None]
            )
            waveform = waveform * valid.to(waveform.dtype)
        if noisy.device.type == "cuda":
            torch.cuda.synchronize(noisy.device)
        elapsed = time.perf_counter() - started
        duration = (
            float(lengths.sum().item()) / (sample_rate * noisy.shape[0])
            if lengths is not None
            else noisy.shape[-1] / float(sample_rate)
        )
        return SamplingResult(
            waveform=waveform,
            spectrum=estimate,
            nfe=nfe,
            inference_time=elapsed,
            rtf=elapsed / max(duration, 1.0e-12),
        )

    def probability_flow_sample(self, *args: object, **kwargs: object) -> SamplingResult:
        """Reserved P1 probability-flow ODE sampler interface."""
        del args, kwargs
        raise NotImplementedError("Probability-flow ODE sampler is a P1 extension")


@torch.inference_mode()
def enhance_long_waveform(
    sampler: PredictorCorrectorSampler,
    model: nn.Module,
    waveform: torch.Tensor,
    sample_rate: int,
    generator: Optional[torch.Generator],
    chunk_samples: Optional[int] = None,
    overlap_samples: int = 0,
    fallback_on_oom: bool = True,
) -> SamplingResult:
    """Prefer full-spectrum inference, then Hann cross-fade chunks if requested/OOM."""
    if waveform.ndim != 1:
        raise ValueError("Long-audio input must be mono [T]")
    device = next(model.parameters()).device
    if chunk_samples is None or waveform.numel() <= chunk_samples:
        try:
            return sampler.sample_waveform(
                model,
                waveform[None].to(device),
                torch.tensor([waveform.numel()], device=device),
                sample_rate,
                generator,
            )
        except RuntimeError as error:
            if not fallback_on_oom or "out of memory" not in str(error).lower():
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if chunk_samples is None:
                chunk_samples = sample_rate * 4
                overlap_samples = sample_rate // 2
    if chunk_samples is None or overlap_samples < 0 or overlap_samples >= chunk_samples:
        raise ValueError("Invalid chunk/overlap configuration")

    step = chunk_samples - overlap_samples
    output = torch.zeros(waveform.numel(), device=device)
    weights = torch.zeros_like(output)
    total_nfe = 0
    total_time = 0.0
    last_spectrum = None
    starts = list(range(0, waveform.numel(), step))
    for chunk_index, start in enumerate(starts):
        valid = min(chunk_samples, waveform.numel() - start)
        chunk = waveform[start : start + valid]
        if valid < chunk_samples:
            chunk = functional.pad(chunk, (0, chunk_samples - valid))
        result = sampler.sample_waveform(
            model,
            chunk[None].to(device),
            torch.tensor([valid], device=device),
            sample_rate,
            generator,
        )
        enhanced = result.waveform[0, :valid]
        window = torch.ones(valid, device=device)
        fade = min(overlap_samples, valid)
        if fade:
            hann = torch.hann_window(2 * fade, periodic=False, device=device)
            if chunk_index > 0:
                window[:fade] = hann[:fade]
            if start + valid < waveform.numel():
                window[-fade:] = hann[fade:]
        output[start : start + valid] += enhanced * window
        weights[start : start + valid] += window
        total_nfe += result.nfe
        total_time += result.inference_time
        last_spectrum = result.spectrum
        if start + valid >= waveform.numel():
            break
    duration = waveform.numel() / float(sample_rate)
    return SamplingResult(
        waveform=(output / weights.clamp_min(1.0e-8))[None],
        spectrum=last_spectrum,
        nfe=total_nfe,
        inference_time=total_time,
        rtf=total_time / max(duration, 1.0e-12),
    )

