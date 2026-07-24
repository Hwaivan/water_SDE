"""Complex STFT and compression numerical tests."""

import torch

from sgmse.utils.stft import ComplexSTFTTransform


def _transform() -> ComplexSTFTTransform:
    return ComplexSTFTTransform(
        n_fft=64,
        win_length=64,
        hop_length=16,
        center=True,
        normalized=False,
        onesided=True,
        alpha=0.5,
        beta=0.15,
    )


def test_stft_istft_round_trip() -> None:
    waveform = torch.randn(2, 1024)
    transform = _transform()
    restored = transform.istft(transform.stft(waveform), waveform.shape[-1])
    assert restored.shape == waveform.shape
    assert torch.allclose(restored, waveform, atol=1.0e-5, rtol=1.0e-4)


def test_compression_round_trip() -> None:
    spectrum = torch.complex(torch.randn(2, 33, 65), torch.randn(2, 33, 65))
    transform = _transform()
    restored = transform.decompress(transform.compress(spectrum))
    assert torch.allclose(restored, spectrum, atol=2.0e-5, rtol=1.0e-4)

