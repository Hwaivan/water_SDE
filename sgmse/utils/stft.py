"""Complex STFT and invertible magnitude compression."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as functional
from torch import nn


def complex_to_channels(value: torch.Tensor) -> torch.Tensor:
    """Convert complex ``[B,F,N]`` to real/imag channels ``[B,2,F,N]``."""
    if value.ndim != 3 or not torch.is_complex(value):
        raise ValueError("Expected complex tensor [B,F,N]")
    return torch.stack((value.real, value.imag), dim=1)


def channels_to_complex(value: torch.Tensor) -> torch.Tensor:
    """Convert real/imag channels ``[B,2,F,N]`` to complex ``[B,F,N]``."""
    if value.ndim != 4 or value.shape[1] != 2:
        raise ValueError("Expected channels [B,2,F,N]")
    return torch.complex(value[:, 0], value[:, 1])


def fit_spectrogram_frames(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    frames: Optional[int],
    start: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the same frame crop/right-zero-pad to aligned complex spectra."""
    if clean.shape != noisy.shape:
        raise ValueError("Clean/noisy spectra must be aligned")
    if frames is None:
        return clean, noisy
    available = clean.shape[-1]
    start = min(max(0, int(start)), max(0, available - frames))
    clean, noisy = clean[..., start : start + frames], noisy[..., start : start + frames]
    pad = max(0, frames - clean.shape[-1])
    if pad:
        clean = functional.pad(clean, (0, pad))
        noisy = functional.pad(noisy, (0, pad))
    return clean, noisy


class ComplexSTFTTransform(nn.Module):
    """Configured STFT with SGMSE magnitude compression.

    Compression is
    ``X_tilde = beta * |X|**alpha * exp(1j*angle(X))``.
    """

    def __init__(
        self,
        n_fft: int,
        win_length: int,
        hop_length: int,
        window: str = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        alpha: float = 0.5,
        beta: float = 0.15,
        eps: float = 1.0e-12,
    ) -> None:
        super().__init__()
        if window != "hann":
            raise ValueError("Only Hann window is currently supported")
        if not 0.0 < alpha <= 1.0 or beta <= 0.0:
            raise ValueError("Require 0 < alpha <= 1 and beta > 0")
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.center = bool(center)
        self.normalized = bool(normalized)
        self.onesided = bool(onesided)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)

    def stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """Transform float waveform ``[B,T]`` to complex ``[B,F,N]``."""
        if waveform.ndim != 2:
            raise ValueError("Waveform must have shape [B,T]")
        return torch.stft(
            waveform.float(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.float(),
            center=self.center,
            normalized=self.normalized,
            onesided=self.onesided,
            return_complex=True,
        )

    def istft(self, spectrum: torch.Tensor, length: Optional[int]) -> torch.Tensor:
        """Invert a complex spectrum and recover the requested waveform length."""
        if spectrum.ndim != 3 or not torch.is_complex(spectrum):
            raise ValueError("Spectrum must have shape [B,F,N] and complex dtype")
        return torch.istft(
            spectrum,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=spectrum.device, dtype=spectrum.real.dtype),
            center=self.center,
            normalized=self.normalized,
            onesided=self.onesided,
            length=length,
        )

    def compress(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Apply invertible power-law magnitude compression."""
        magnitude = spectrum.abs()
        phase = spectrum / magnitude.clamp_min(self.eps)
        compressed = self.beta * magnitude.pow(self.alpha) * phase
        return torch.where(magnitude > self.eps, compressed, torch.zeros_like(compressed))

    def decompress(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Invert :meth:`compress` without changing phase."""
        magnitude = spectrum.abs()
        phase = spectrum / magnitude.clamp_min(self.eps)
        restored_magnitude = (magnitude / self.beta).clamp_min(0.0).pow(1.0 / self.alpha)
        restored = restored_magnitude * phase
        return torch.where(magnitude > self.eps, restored, torch.zeros_like(restored))

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """STFT followed by magnitude compression."""
        return self.compress(self.stft(waveform))

    def decode(self, compressed: torch.Tensor, length: Optional[int]) -> torch.Tensor:
        """Inverse compression followed by exact-length iSTFT."""
        return self.istft(self.decompress(compressed), length)

