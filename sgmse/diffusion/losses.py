"""Complex denoising score-matching objective."""

from typing import Any, Dict, Optional

import torch
from torch import nn

from sgmse.utils.stft import (
    ComplexSTFTTransform,
    channels_to_complex,
    complex_to_channels,
    fit_spectrogram_frames,
)
from .ouve import OUVESDE, complex_gaussian


def complex_score_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Mean complex squared error ``E[(dr)^2 + (di)^2]``."""
    if prediction.shape != target.shape or not torch.is_complex(prediction):
        raise ValueError("prediction and target must be aligned complex tensors")
    error = prediction - target
    return (error.real.square() + error.imag.square()).mean()


class ScoreMatchingObjective(nn.Module):
    """Create OUVE training pairs and evaluate complex score loss."""

    def __init__(
        self,
        transform: ComplexSTFTTransform,
        sde: OUVESDE,
        crop_frames: Optional[int] = None,
        auxiliary_waveform_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.sde = sde
        self.crop_frames = int(crop_frames) if crop_frames is not None else None
        self.auxiliary_waveform_weight = float(auxiliary_waveform_weight)

    def forward(
        self,
        model: nn.Module,
        clean_waveform: torch.Tensor,
        noisy_waveform: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Return total score loss and scalar diagnostics.

        STFT, SDE and complex arithmetic are explicitly float32. Only the real
        NCSN++ convolution call is eligible for outer AMP autocast.
        """
        if clean_waveform.shape != noisy_waveform.shape:
            raise ValueError("Clean/noisy waveforms must be aligned")
        with torch.cuda.amp.autocast(enabled=False):
            clean = self.transform.encode(clean_waveform.float())
            noisy = self.transform.encode(noisy_waveform.float())
            if self.crop_frames is not None:
                maximum = max(0, clean.shape[-1] - self.crop_frames)
                if deterministic or maximum == 0:
                    start = 0
                else:
                    start = int(
                        torch.randint(
                            maximum + 1,
                            (1,),
                            device=clean.device,
                            generator=generator,
                        ).item()
                    )
                clean, noisy = fit_spectrogram_frames(
                    clean, noisy, self.crop_frames, start
                )
            batch = clean.shape[0]
            if deterministic:
                t = torch.full(
                    (batch,),
                    (self.sde.t_eps + self.sde.T) * 0.5,
                    device=clean.device,
                    dtype=clean.real.dtype,
                )
            else:
                t = torch.rand(
                    (batch,),
                    device=clean.device,
                    dtype=clean.real.dtype,
                    generator=generator,
                )
                t = self.sde.t_eps + (self.sde.T - self.sde.t_eps) * t
            noise = complex_gaussian(
                clean.shape, clean.device, clean.real.dtype, generator
            )
            mean, std = self.sde.marginal_prob(clean, noisy, t)
            std_broadcast = std.reshape(batch, 1, 1)
            perturbed = mean + std_broadcast * noise
            network_input = torch.cat(
                (complex_to_channels(perturbed), complex_to_channels(noisy)), dim=1
            )
            target_score = -noise / std_broadcast.clamp_min(1.0e-8)

        predicted_channels = model(network_input, t)
        predicted_score = channels_to_complex(predicted_channels.float())
        score_loss = complex_score_mse(predicted_score, target_score)
        total = score_loss
        # Reserved extension point. Waveform reconstruction is intentionally not
        # mixed into the default score objective.
        if self.auxiliary_waveform_weight != 0.0:
            raise NotImplementedError(
                "Auxiliary waveform loss is reserved and disabled by default"
            )
        return {
            "total": total,
            "score_loss": score_loss.detach(),
            "sigma_mean": std.mean().detach(),
            "t_mean": t.mean().detach(),
        }

