"""Strict paired alignment and on-the-fly SNR tests."""

import wave
from pathlib import Path

import numpy as np
import torch

from sgmse.data.dataset import SGMSEDataset


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 16000) -> None:
    pcm = np.round(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())


def _base_config(tmp_path: Path) -> dict:
    return {
        "data_mode": "paired",
        "sample_rate": 16000,
        "segment_seconds": 0.05,
        "train_manifest": str(tmp_path / "pairs.txt"),
        "valid_manifest": str(tmp_path / "pairs.txt"),
        "test_manifest": str(tmp_path / "pairs.txt"),
        "noise_manifest": None,
        "snr_min": 5.0,
        "snr_max": 5.0,
        "mono": True,
        "num_workers": 0,
        "segment_validation": False,
        "silence_threshold": 1.0e-8,
        "shared_peak_limit": None,
    }


def test_paired_manifest_preserves_alignment(tmp_path: Path) -> None:
    clean = 0.2 * np.sin(np.linspace(0, 20, 800, dtype=np.float32))
    noisy = clean + 0.02 * np.cos(np.linspace(0, 40, 800, dtype=np.float32))
    _write_wav(tmp_path / "clean.wav", clean)
    _write_wav(tmp_path / "noisy.wav", noisy)
    (tmp_path / "pairs.txt").write_text(
        "{}\t{}\n".format(tmp_path / "noisy.wav", tmp_path / "clean.wav"),
        encoding="utf-8",
    )
    dataset = SGMSEDataset(_base_config(tmp_path), "train", 3)
    item = dataset[0]
    assert item["clean"].shape == item["noisy"].shape == (800,)
    assert item["length"] == 800


def test_on_the_fly_mixture_matches_requested_snr(tmp_path: Path) -> None:
    clean = 0.2 * np.sin(np.linspace(0, 20, 800, dtype=np.float32))
    noise = 0.1 * np.cos(np.linspace(0, 73, 800, dtype=np.float32))
    _write_wav(tmp_path / "clean.wav", clean)
    _write_wav(tmp_path / "noise.wav", noise)
    (tmp_path / "clean.txt").write_text(
        str(tmp_path / "clean.wav") + "\n", encoding="utf-8"
    )
    (tmp_path / "noise.txt").write_text(
        str(tmp_path / "noise.wav") + "\n", encoding="utf-8"
    )
    config = _base_config(tmp_path)
    config.update(
        {
            "data_mode": "on_the_fly",
            "train_manifest": str(tmp_path / "clean.txt"),
            "noise_manifest": str(tmp_path / "noise.txt"),
        }
    )
    item = SGMSEDataset(config, "train", 5)[0]
    actual = 10.0 * torch.log10(
        item["clean"].square().mean() / item["noise"].square().mean()
    )
    assert abs(float(actual) - 5.0) < 0.05

