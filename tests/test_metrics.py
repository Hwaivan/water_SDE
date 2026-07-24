"""Required SDR/SDRi/SI-SNR/SI-SNRi behavior."""

import torch

from sgmse.metrics.audio_metrics import compute_audio_metrics


def test_unchanged_enhanced_has_zero_improvements() -> None:
    clean = torch.randn(1000)
    noisy = clean + 0.1 * torch.randn_like(clean)
    row = compute_audio_metrics(noisy, clean, noisy)[0]
    assert abs(row["sdri"]) < 1.0e-6
    assert abs(row["si_snri"]) < 1.0e-6


def test_clean_and_scaled_clean_distinguish_sdr_from_si_snr() -> None:
    clean = torch.randn(1000)
    noisy = clean + torch.randn_like(clean)
    perfect = compute_audio_metrics(clean, clean, noisy)[0]
    scaled = compute_audio_metrics(2.0 * clean, clean, noisy)[0]
    assert perfect["output_sdr"] > 80.0
    assert perfect["output_si_snr"] > 70.0
    assert scaled["output_si_snr"] > 70.0
    assert scaled["output_sdr"] < 1.0


def test_random_output_is_worse_and_silence_is_invalid() -> None:
    clean = torch.randn(1000)
    noisy = clean + 0.5 * torch.randn_like(clean)
    high_quality = clean + 0.01 * torch.randn_like(clean)
    random = torch.randn_like(clean)
    good = compute_audio_metrics(high_quality, clean, noisy)[0]
    bad = compute_audio_metrics(random, clean, noisy)[0]
    assert good["output_sdr"] > bad["output_sdr"]
    assert good["output_si_snr"] > bad["output_si_snr"]
    silent = compute_audio_metrics(
        torch.zeros(100), torch.zeros(100), torch.zeros(100)
    )[0]
    assert silent["valid"] is False
    assert silent["error"] == "silent_reference"


def test_length_policy_crop_or_error() -> None:
    row = compute_audio_metrics(
        torch.ones(90), torch.ones(100), torch.ones(110), alignment_policy="crop"
    )[0]
    assert row["valid"]
    try:
        compute_audio_metrics(
            torch.ones(90), torch.ones(100), torch.ones(110), alignment_policy="error"
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected explicit mismatch error")

