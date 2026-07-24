"""Rank-zero generative evaluation and required artifact export."""

import csv
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

from sgmse.data.audio_io import save_audio
from sgmse.diffusion.sampler import PredictorCorrectorSampler
from sgmse.metrics.audio_metrics import (
    compute_audio_metrics,
    summarize_metric_rows,
)


def _git_commit(workspace: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(workspace),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    sampler: PredictorCorrectorSampler,
    loader: Iterable[Dict[str, Any]],
    device: torch.device,
    config: Dict[str, Any],
    checkpoint: str,
    output_dir: str,
    seed: int,
    generator: torch.Generator,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Enhance each sample exactly once and write CSV/JSON/WAV artifacts."""
    directory = Path(output_dir)
    enhanced_directory = directory / "enhanced_wavs"
    directory.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    model.eval()
    sample_rate = int(config["data"]["sample_rate"])
    for batch in loader:
        noisy = batch["noisy"].to(device)
        clean = batch["clean"].to(device)
        lengths = batch["lengths"].to(device)
        result = sampler.sample_waveform(
            model, noisy, lengths, sample_rate, generator
        )
        metric_rows = compute_audio_metrics(
            result.waveform,
            clean,
            noisy,
            sample_rate,
            config["metrics"].get("alignment_policy", "crop"),
            float(config["metrics"].get("eps", 1.0e-8)),
            config["metrics"].get("min_db"),
            config["metrics"].get("max_db"),
        )
        per_item_time = result.inference_time / noisy.shape[0]
        for index, metric_row in enumerate(metric_rows):
            length = int(lengths[index].item())
            duration = length / float(sample_rate)
            row = {
                "file_id": batch["file_id"][index],
                **metric_row,
                "duration": duration,
                "inference_time": per_item_time,
                "rtf": per_item_time / max(duration, 1.0e-12),
                "nfe": result.nfe,
            }
            rows.append(row)
            save_audio(
                str(enhanced_directory / (batch["file_id"][index] + ".wav")),
                result.waveform[index, :length],
                sample_rate,
            )
            logger.info(
                "file=%s valid=%s SI-SNRi=%s SDRi=%s RTF=%.3f",
                row["file_id"],
                row["valid"],
                row.get("si_snri", "n/a"),
                row.get("sdri", "n/a"),
                row["rtf"],
            )
    fields = [
        "file_id",
        "input_sdr",
        "output_sdr",
        "sdri",
        "input_si_snr",
        "output_si_snr",
        "si_snri",
        "duration",
        "inference_time",
        "rtf",
        "nfe",
        "valid",
        "error",
    ]
    with (directory / "per_file_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_metric_rows(rows)
    summary.update(
        {
            "NFE": int(rows[0]["nfe"]) if rows else 0,
            "mean_inference_time": (
                sum(row["inference_time"] for row in rows) / max(1, len(rows))
            ),
            "mean_rtf": sum(row["rtf"] for row in rows) / max(1, len(rows)),
            "seed": int(seed),
            "checkpoint": str(Path(checkpoint).resolve()),
            "git_commit": _git_commit(Path(__file__).resolve().parents[3]),
            "config": {
                key: value
                for key, value in config.items()
                if not key.startswith("_")
            },
        }
    )
    with (directory / "summary_metrics.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    return summary

