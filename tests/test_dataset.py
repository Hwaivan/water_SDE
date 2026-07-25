"""Strict paired alignment and waveform-domain environment-noise tests."""

import copy
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from sgmse.data.audio_io import load_audio
from sgmse.data.dataset import SGMSEDataset
from sgmse.utils.config import load_config, validate_config


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


def _online_config(
    tmp_path: Path,
    noise_mode: str,
    noise_manifest: str = None,
) -> dict:
    clean = 0.2 * np.sin(np.linspace(0, 20, 800, dtype=np.float32))
    _write_wav(tmp_path / "clean.wav", clean)
    (tmp_path / "clean.txt").write_text(
        str(tmp_path / "clean.wav") + "\n", encoding="utf-8"
    )
    config = _base_config(tmp_path)
    config.update(
        {
            "data_mode": "on_the_fly",
            "train_manifest": str(tmp_path / "clean.txt"),
            "noise_manifest": noise_manifest,
            "noise_mode": noise_mode,
            "white_noise_probability": 0.2,
            "white_noise_ratio": 0.3,
        }
    )
    return config


def _write_noise_manifest(
    tmp_path: Path,
    name: str = "noise",
    frequency: float = 73.0,
) -> str:
    noise = 0.1 * np.cos(
        np.linspace(0, frequency, 800, dtype=np.float32)
    )
    noise_path = tmp_path / "{}.wav".format(name)
    manifest_path = tmp_path / "{}.txt".format(name)
    _write_wav(noise_path, noise)
    manifest_path.write_text(str(noise_path) + "\n", encoding="utf-8")
    return str(manifest_path)


def _actual_snr(item: dict) -> float:
    valid = int(item["length"])
    clean_power = item["clean"][:valid].square().mean()
    noise_power = item["noise"][:valid].square().mean()
    return float(10.0 * torch.log10(clean_power / noise_power))


def _unit_norm(waveform: torch.Tensor) -> torch.Tensor:
    return waveform / waveform.square().sum().sqrt()


def test_paired_manifest_preserves_alignment(tmp_path: Path) -> None:
    clean = 0.2 * np.sin(np.linspace(0, 20, 800, dtype=np.float32))
    noisy = clean + 0.02 * np.cos(
        np.linspace(0, 40, 800, dtype=np.float32)
    )
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


def test_real_mode_matches_requested_total_snr(tmp_path: Path) -> None:
    manifest = _write_noise_manifest(tmp_path)
    config = _online_config(tmp_path, "real", manifest)
    item = SGMSEDataset(config, "train", 5)[0]
    assert item["noise_branch"] == "real"
    assert abs(_actual_snr(item) - item["input_snr"]) < 0.05


def test_white_mode_does_not_require_noise_manifest(tmp_path: Path) -> None:
    config = _online_config(tmp_path, "white", None)
    item = SGMSEDataset(config, "train", 5)[0]
    assert item["noise_branch"] == "pure_white"
    assert item["noisy_path"] == "generated:white"


def test_white_mode_matches_requested_total_snr(tmp_path: Path) -> None:
    config = _online_config(tmp_path, "white", None)
    config.update({"snr_min": -2.5, "snr_max": -2.5})
    item = SGMSEDataset(config, "train", 11)[0]
    assert abs(_actual_snr(item) + 2.5) < 0.05


def test_mixed_probability_one_is_always_pure_white(tmp_path: Path) -> None:
    manifest = _write_noise_manifest(tmp_path)
    config = _online_config(tmp_path, "mixed", manifest)
    config["white_noise_probability"] = 1.0
    dataset = SGMSEDataset(config, "train", 9)
    for epoch in range(8):
        dataset.set_epoch(epoch)
        assert dataset[0]["noise_branch"] == "pure_white"


def test_mixed_zero_probability_and_ratio_is_only_real(
    tmp_path: Path,
) -> None:
    manifest = _write_noise_manifest(tmp_path)
    config = _online_config(tmp_path, "mixed", manifest)
    config.update(
        {"white_noise_probability": 0.0, "white_noise_ratio": 0.0}
    )
    item = SGMSEDataset(config, "train", 13)[0]
    source = load_audio(
        str(tmp_path / "noise.wav"), config["sample_rate"], True
    )
    assert item["noise_branch"] == "real_white_mix"
    assert item["white_noise_ratio"] == 0.0
    assert torch.allclose(
        _unit_norm(item["noise"]), _unit_norm(source), atol=1.0e-6
    )


def test_mixed_zero_probability_and_ratio_one_is_only_white(
    tmp_path: Path,
) -> None:
    first_manifest = _write_noise_manifest(tmp_path, "noise_a", 37.0)
    second_manifest = _write_noise_manifest(tmp_path, "noise_b", 113.0)
    first = _online_config(tmp_path, "mixed", first_manifest)
    second = _online_config(tmp_path, "mixed", second_manifest)
    for config in (first, second):
        config.update(
            {"white_noise_probability": 0.0, "white_noise_ratio": 1.0}
        )
    first_item = SGMSEDataset(first, "train", 17)[0]
    second_item = SGMSEDataset(second, "train", 17)[0]
    assert first_item["noise_branch"] == "real_white_mix"
    assert first_item["white_noise_ratio"] == 1.0
    assert torch.equal(first_item["noise"], second_item["noise"])


def test_mixed_uses_constructed_total_power_for_target_snr(
    tmp_path: Path,
) -> None:
    manifest = _write_noise_manifest(tmp_path)
    config = _online_config(tmp_path, "mixed", manifest)
    config.update(
        {
            "white_noise_probability": 0.0,
            "white_noise_ratio": 0.3,
            "snr_min": 7.25,
            "snr_max": 7.25,
        }
    )
    item = SGMSEDataset(config, "train", 21)[0]
    assert item["noise_branch"] == "real_white_mix"
    assert abs(_actual_snr(item) - 7.25) < 0.05


def test_online_noise_is_exactly_reproducible_for_same_seed(
    tmp_path: Path,
) -> None:
    manifest = _write_noise_manifest(tmp_path)
    config = _online_config(tmp_path, "mixed", manifest)
    first = SGMSEDataset(config, "train", 29)[0]
    second = SGMSEDataset(config, "train", 29)[0]
    assert first["noise_branch"] == second["noise_branch"]
    assert first["input_snr"] == second["input_snr"]
    assert torch.equal(first["noise"], second["noise"])
    assert torch.equal(first["noisy"], second["noisy"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("white_noise_probability", -0.01),
        ("white_noise_probability", 1.01),
        ("white_noise_ratio", -0.01),
        ("white_noise_ratio", 1.01),
    ],
)
def test_invalid_probability_or_ratio_raises(
    tmp_path: Path, field: str, value: float
) -> None:
    config = _online_config(tmp_path, "white", None)
    config[field] = value
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        SGMSEDataset(config, "train", 1)


def test_config_validation_applies_backward_compatible_noise_defaults() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "sgmse_water_small.yaml"
    config = load_config(str(config_path))
    assert config["data"]["noise_mode"] == "real"
    assert config["data"]["white_noise_probability"] == 0.0
    assert config["data"]["white_noise_ratio"] == 0.0


@pytest.mark.parametrize(
    "field,value",
    [
        ("white_noise_probability", -1.0),
        ("white_noise_probability", 2.0),
        ("white_noise_ratio", -1.0),
        ("white_noise_ratio", 2.0),
    ],
)
def test_config_validation_rejects_invalid_noise_values(
    field: str, value: float
) -> None:
    config_path = Path(__file__).parents[1] / "configs" / "sgmse_water_full.yaml"
    config = copy.deepcopy(load_config(str(config_path)))
    config["data"][field] = value
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_config(config)
