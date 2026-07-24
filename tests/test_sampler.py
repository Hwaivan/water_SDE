"""Predictor-corrector and single-file inference smoke tests."""

from pathlib import Path

import torch
from torch import nn

from sgmse.data.audio_io import load_audio, save_audio
from sgmse.diffusion.ouve import OUVESDE
from sgmse.diffusion.sampler import (
    PredictorCorrectorSampler,
    enhance_long_waveform,
)
from sgmse.utils.stft import ComplexSTFTTransform


class DummyScore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(self, inputs: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        del time
        return inputs[:, :2] * self.scale


def _sampler(corrector: str = "none") -> PredictorCorrectorSampler:
    transform = ComplexSTFTTransform(
        32, 32, 8, alpha=0.5, beta=0.15
    )
    return PredictorCorrectorSampler(
        OUVESDE(),
        transform,
        num_steps=2,
        corrector=corrector,
        corrector_steps=1,
        corrector_step_size=0.01,
    )


def test_sampler_dummy_model_completes() -> None:
    model = DummyScore().eval()
    waveform = torch.randn(2, 256) * 0.05
    generator = torch.Generator().manual_seed(7)
    result = _sampler().sample_waveform(
        model, waveform, torch.tensor([256, 200]), 16000, generator
    )
    assert result.waveform.shape == waveform.shape
    assert result.nfe == 2
    assert torch.isfinite(result.waveform).all()


def test_sampler_is_reproducible_with_fixed_generator() -> None:
    model = DummyScore().eval()
    waveform = torch.randn(1, 256) * 0.05
    first = _sampler().sample_waveform(
        model,
        waveform,
        torch.tensor([256]),
        16000,
        torch.Generator().manual_seed(77),
    )
    second = _sampler().sample_waveform(
        model,
        waveform,
        torch.tensor([256]),
        16000,
        torch.Generator().manual_seed(77),
    )
    assert torch.equal(first.waveform, second.waveform)


def test_single_file_cpu_inference_smoke(tmp_path: Path) -> None:
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "enhanced.wav"
    waveform = 0.05 * torch.sin(torch.linspace(0, 30, 400))
    save_audio(str(input_path), waveform, 16000)
    loaded = load_audio(str(input_path), 16000)
    model = DummyScore().eval()
    result = enhance_long_waveform(
        _sampler(),
        model,
        loaded,
        16000,
        torch.Generator().manual_seed(9),
    )
    save_audio(str(output_path), result.waveform[0], 16000)
    restored = load_audio(str(output_path), 16000)
    assert restored.numel() == waveform.numel()
    assert torch.isfinite(restored).all()
