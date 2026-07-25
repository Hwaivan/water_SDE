"""YAML loading and strict SGMSE configuration validation."""

from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def _require(mapping: Dict[str, Any], names: Iterable[str], section: str) -> None:
    missing = [name for name in names if name not in mapping]
    if missing:
        raise KeyError("{} is missing: {}".format(section, ", ".join(missing)))


def validate_config(config: Dict[str, Any]) -> None:
    """Validate mathematical and runtime constraints before constructing objects."""
    _require(
        config,
        (
            "experiment",
            "data",
            "stft",
            "compression",
            "sde",
            "model",
            "sampler",
            "training",
            "validation",
            "metrics",
            "logging",
            "distributed",
        ),
        "config",
    )
    data = config["data"]
    _require(
        data,
        (
            "sample_rate",
            "segment_seconds",
            "train_manifest",
            "valid_manifest",
            "test_manifest",
            "data_mode",
            "snr_min",
            "snr_max",
            "mono",
            "num_workers",
        ),
        "data",
    )
    if data["data_mode"] not in ("paired", "on_the_fly"):
        raise ValueError("data.data_mode must be paired or on_the_fly")
    # Defaults keep configurations created before waveform noise modes valid.
    data.setdefault("noise_mode", "real")
    data.setdefault("white_noise_probability", 0.0)
    data.setdefault("white_noise_ratio", 0.0)
    if data["noise_mode"] not in ("real", "white", "mixed"):
        raise ValueError("data.noise_mode must be real, white, or mixed")
    white_probability = float(data["white_noise_probability"])
    white_ratio = float(data["white_noise_ratio"])
    if not 0.0 <= white_probability <= 1.0:
        raise ValueError("data.white_noise_probability must be in [0, 1]")
    if not 0.0 <= white_ratio <= 1.0:
        raise ValueError("data.white_noise_ratio must be in [0, 1]")
    if (
        data["data_mode"] == "on_the_fly"
        and data["noise_mode"] in ("real", "mixed")
        and not data.get("noise_manifest")
    ):
        raise ValueError(
            "data.noise_manifest is required for real and mixed on-the-fly noise"
        )
    if float(data["snr_min"]) > float(data["snr_max"]):
        raise ValueError("data.snr_min must not exceed data.snr_max")
    if int(data["sample_rate"]) <= 0 or float(data["segment_seconds"]) <= 0:
        raise ValueError("sample_rate and segment_seconds must be positive")

    stft = config["stft"]
    _require(
        stft,
        ("n_fft", "win_length", "hop_length", "window", "center", "normalized", "onesided"),
        "stft",
    )
    if int(stft["hop_length"]) > int(stft["win_length"]):
        raise ValueError("stft.hop_length must not exceed stft.win_length")
    if int(stft["win_length"]) > int(stft["n_fft"]):
        raise ValueError("stft.win_length must not exceed stft.n_fft")

    compression = config["compression"]
    alpha, beta = float(compression["alpha"]), float(compression["beta"])
    if not 0.0 < alpha <= 1.0:
        raise ValueError("compression.alpha must be in (0, 1]")
    if beta <= 0.0:
        raise ValueError("compression.beta must be positive")

    sde = config["sde"]
    sigma_min, sigma_max = float(sde["sigma_min"]), float(sde["sigma_max"])
    if not sigma_max > sigma_min > 0.0:
        raise ValueError("sde must satisfy sigma_max > sigma_min > 0")
    if float(sde["gamma"]) <= 0.0:
        raise ValueError("sde.gamma must be positive")
    if not 0.0 <= float(sde["t_eps"]) < float(sde["T"]):
        raise ValueError("sde must satisfy 0 <= t_eps < T")
    if int(config["sampler"]["num_steps"]) < 1:
        raise ValueError("sampler.num_steps must be at least 1")
    if config["sampler"]["predictor"] != "euler_maruyama":
        raise ValueError("Only euler_maruyama predictor is currently supported")
    if config["sampler"]["corrector"] not in ("annealed_langevin", "none"):
        raise ValueError("sampler.corrector must be annealed_langevin or none")


def load_config(path: str) -> Dict[str, Any]:
    """Read and validate a YAML configuration."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError("Configuration not found: {}".format(config_path))
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    validate_config(config)
    config["_config_path"] = str(config_path)
    return config
