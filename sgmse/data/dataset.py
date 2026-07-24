"""Strict aligned-pair and on-the-fly waveform datasets."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as functional
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from sgmse.utils.seed import seed_worker
from .audio_io import load_audio, probe_audio


def _manifest_lines(path: str) -> List[str]:
    manifest = Path(path).expanduser()
    if not manifest.is_file():
        raise FileNotFoundError("Manifest not found: {}".format(manifest))
    with manifest.open("r", encoding="utf-8-sig") as stream:
        rows = [
            row.strip()
            for row in stream
            if row.strip() and not row.lstrip().startswith("#")
        ]
    if not rows:
        raise ValueError("Manifest contains no samples: {}".format(manifest))
    return rows


def _fit_waveform(
    waveform: torch.Tensor,
    target_length: Optional[int],
    start: int = 0,
) -> Tuple[torch.Tensor, int]:
    """Crop from ``start`` or right-zero-pad; never repeat audio."""
    original = int(waveform.shape[-1])
    if target_length is None:
        return waveform, original
    start = min(max(0, int(start)), max(0, original - target_length))
    fitted = waveform[start : start + target_length]
    valid = int(fitted.numel())
    if valid < target_length:
        fitted = functional.pad(fitted, (0, target_length - valid))
    return fitted, valid


def _shared_limit(clean: torch.Tensor, noisy: torch.Tensor, limit: Optional[float]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prevent overflow with one common gain; never normalize streams separately."""
    if limit is None:
        return clean, noisy
    peak = torch.maximum(clean.abs().max(), noisy.abs().max())
    if peak <= float(limit):
        return clean, noisy
    gain = float(limit) / peak.clamp_min(1.0e-12)
    return clean * gain, noisy * gain


class SGMSEDataset(Dataset):
    """Waveform dataset preserving legacy ``noisy<TAB>clean`` manifests.

    Returned clean/noisy tensors are strictly aligned ``[T]``. Legacy aliases
    ``target`` and ``mixture`` are included to ease integration.
    """

    def __init__(self, config: Dict[str, Any], split: str, seed: int) -> None:
        self.config = config
        self.split = split
        self.training = split == "train"
        self.seed = int(seed)
        self.epoch = 0
        self.sample_rate = int(config["sample_rate"])
        self.mono = bool(config.get("mono", True))
        if not self.mono:
            raise NotImplementedError("Current SGMSE workspace is single-channel")
        self.segment_length = (
            int(round(float(config["segment_seconds"]) * self.sample_rate))
            if self.training or config.get("segment_validation", False)
            else None
        )
        self.silence_threshold = float(config.get("silence_threshold", 1.0e-8))
        self.peak_limit = config.get("shared_peak_limit")
        self.mode = config["data_mode"] if self.training else "paired"
        manifest_key = "{}_manifest".format(split)
        if self.mode == "paired":
            self.pairs = self._parse_pairs(config[manifest_key])
            self.clean_paths: List[str] = []
            self.noise_paths: List[str] = []
        else:
            self.clean_paths = _manifest_lines(config["train_manifest"])
            noise_manifest = config.get("noise_manifest")
            if not noise_manifest:
                raise ValueError("on_the_fly mode requires data.noise_manifest")
            self.noise_paths = _manifest_lines(noise_manifest)
            self.pairs = []

    @staticmethod
    def _parse_pairs(path: str) -> List[Tuple[str, str]]:
        pairs = []
        for row in _manifest_lines(path):
            fields = row.split("\t")
            if len(fields) != 2:
                raise ValueError(
                    "Paired row must use legacy noisy<TAB>clean format: {}".format(row)
                )
            noisy_path, clean_path = fields
            noisy_rate, noisy_frames, _ = probe_audio(noisy_path)
            clean_rate, clean_frames, _ = probe_audio(clean_path)
            if noisy_rate != clean_rate:
                raise ValueError(
                    "Paired sample-rate mismatch: {} Hz vs {} Hz for {}".format(
                        noisy_rate, clean_rate, row
                    )
                )
            if noisy_frames != clean_frames:
                raise ValueError(
                    "Paired length mismatch: {} vs {} frames for {}".format(
                        noisy_frames, clean_frames, row
                    )
                )
            pairs.append((noisy_path, clean_path))
        return pairs

    def __len__(self) -> int:
        return len(self.pairs) if self.mode == "paired" else len(self.clean_paths)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _generator(self, index: int) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch * max(1, len(self)) + index)
        return generator

    def _paired(self, index: int, generator: torch.Generator) -> Dict[str, Any]:
        noisy_path, clean_path = self.pairs[index]
        noisy = load_audio(noisy_path, self.sample_rate, self.mono)
        clean = load_audio(clean_path, self.sample_rate, self.mono)
        if noisy.shape != clean.shape:
            raise ValueError("Pair lost alignment after loading: {}".format(noisy_path))
        target = self.segment_length
        max_start = max(0, clean.numel() - (target or clean.numel()))
        start = (
            int(torch.randint(max_start + 1, (1,), generator=generator).item())
            if self.training and max_start
            else 0
        )
        clean, valid = _fit_waveform(clean, target, start)
        noisy, noisy_valid = _fit_waveform(noisy, target, start)
        if valid != noisy_valid:
            raise RuntimeError("Internal paired crop alignment failure")
        clean, noisy = _shared_limit(clean, noisy, self.peak_limit)
        return {
            "clean": clean,
            "noisy": noisy,
            "noise": noisy - clean,
            "length": valid,
            "file_id": Path(noisy_path).stem,
            "noisy_path": noisy_path,
            "clean_path": clean_path,
        }

    def _online(self, index: int, generator: torch.Generator) -> Dict[str, Any]:
        clean_path = self.clean_paths[index]
        noise_index = int(torch.randint(len(self.noise_paths), (1,), generator=generator).item())
        noise_path = self.noise_paths[noise_index]
        clean = load_audio(clean_path, self.sample_rate, self.mono)
        noise = load_audio(noise_path, self.sample_rate, self.mono)
        target = self.segment_length
        clean_max = max(0, clean.numel() - (target or clean.numel()))
        noise_max = max(0, noise.numel() - (target or noise.numel()))
        clean_start = int(torch.randint(clean_max + 1, (1,), generator=generator).item()) if clean_max else 0
        noise_start = int(torch.randint(noise_max + 1, (1,), generator=generator).item()) if noise_max else 0
        clean, valid = _fit_waveform(clean, target, clean_start)
        noise, _ = _fit_waveform(noise, target, noise_start)
        clean_power = clean[:valid].square().mean() if valid else clean.new_zeros(())
        noise_power = noise.square().mean()
        if clean_power <= self.silence_threshold:
            raise ValueError("Silent clean sample: {}".format(clean_path))
        if noise_power <= self.silence_threshold:
            raise ValueError("Silent noise sample: {}".format(noise_path))
        snr = float(
            torch.empty(1).uniform_(
                float(self.config["snr_min"]),
                float(self.config["snr_max"]),
                generator=generator,
            ).item()
        )
        noise_scale = torch.sqrt(
            clean_power
            / (noise_power * clean.new_tensor(10.0).pow(snr / 10.0))
        )
        scaled_noise = noise * noise_scale
        noisy = clean + scaled_noise
        clean, noisy = _shared_limit(clean, noisy, self.peak_limit)
        scaled_noise = noisy - clean
        return {
            "clean": clean,
            "noisy": noisy,
            "noise": scaled_noise,
            "length": valid,
            "file_id": Path(clean_path).stem,
            "noisy_path": noise_path,
            "clean_path": clean_path,
            "input_snr": snr,
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        generator = self._generator(index)
        item = (
            self._paired(index, generator)
            if self.mode == "paired"
            else self._online(index, generator)
        )
        if item["clean"][: item["length"]].square().mean() <= self.silence_threshold:
            raise ValueError("Silent clean sample: {}".format(item["clean_path"]))
        item["target"] = item["clean"]
        item["mixture"] = item["noisy"]
        return item


def collate_batch(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad aligned waveforms and preserve legacy batch aliases."""
    if not items:
        raise ValueError("Cannot collate an empty batch")
    clean = pad_sequence([item["clean"] for item in items], batch_first=True)
    noisy = pad_sequence([item["noisy"] for item in items], batch_first=True)
    noise = pad_sequence([item["noise"] for item in items], batch_first=True)
    result = {
        "clean": clean,
        "noisy": noisy,
        "noise": noise,
        "target": clean,
        "mixture": noisy,
        "lengths": torch.tensor([item["length"] for item in items], dtype=torch.long),
        "file_id": [item["file_id"] for item in items],
        "noisy_path": [item["noisy_path"] for item in items],
        "clean_path": [item["clean_path"] for item in items],
    }
    return result


def build_dataloader(
    dataset: SGMSEDataset,
    config: Dict[str, Any],
    training: bool,
    seed: int,
    distributed: bool = False,
) -> DataLoader:
    """Construct deterministic single-process or distributed DataLoader."""
    sampler = (
        DistributedSampler(dataset, shuffle=training, seed=seed, drop_last=False)
        if distributed
        else None
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    workers = int(config["data"].get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=workers,
        collate_fn=collate_batch,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=False,
    )
