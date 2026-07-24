"""Scale-dependent SDR and SI-SNR with explicit invalid-sample handling."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def _as_batch(value: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    if value.ndim == 1:
        return value[None], True
    if value.ndim == 2:
        return value, False
    if value.ndim == 3:
        return value.flatten(1), False
    raise ValueError("Audio metric input must be [T], [B,T], or [B,C,T]")


def _align(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    noisy: torch.Tensor,
    policy: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = (estimate.shape[-1], reference.shape[-1], noisy.shape[-1])
    if len(set(lengths)) == 1:
        return estimate, reference, noisy
    if policy == "error":
        raise ValueError("Waveform length mismatch: {}".format(lengths))
    if policy != "crop":
        raise ValueError("alignment_policy must be crop or error")
    length = min(lengths)
    return estimate[..., :length], reference[..., :length], noisy[..., :length]


def scale_dependent_sdr(
    estimate: torch.Tensor, reference: torch.Tensor, eps: float = 1.0e-8
) -> torch.Tensor:
    """Return signal-to-error SDR, not BSS-Eval filtered SDR."""
    signal = reference.square().sum(dim=-1)
    error = (estimate - reference).square().sum(dim=-1)
    return 10.0 * torch.log10((signal + eps) / (error + eps))


def si_snr(
    estimate: torch.Tensor, reference: torch.Tensor, eps: float = 1.0e-8
) -> torch.Tensor:
    """Return scale-invariant SNR after independent mean removal."""
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)
    reference_energy = reference.square().sum(dim=-1, keepdim=True)
    scale = (estimate * reference).sum(dim=-1, keepdim=True) / reference_energy.clamp_min(eps)
    target = scale * reference
    residual = estimate - target
    target_energy = target.square().sum(dim=-1).clamp_min(eps)
    residual_energy = residual.square().sum(dim=-1)
    # Relative numerical floor preserves scale invariance for exact estimates.
    ratio = target_energy / (residual_energy + eps * target_energy)
    return 10.0 * torch.log10(ratio.clamp_min(eps))


def compute_audio_metrics(
    enhanced: torch.Tensor,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    sample_rate: int = 16000,
    alignment_policy: str = "crop",
    eps: float = 1.0e-8,
    min_db: Optional[float] = None,
    max_db: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Compute independent rows and mark silence/non-finite samples invalid."""
    enhanced, single = _as_batch(enhanced)
    clean, _ = _as_batch(clean)
    noisy, _ = _as_batch(noisy)
    if not enhanced.shape[0] == clean.shape[0] == noisy.shape[0]:
        raise ValueError("Metric batch sizes must match")
    enhanced, clean, noisy = _align(enhanced, clean, noisy, alignment_policy)
    rows: List[Dict[str, Any]] = []
    for index in range(clean.shape[0]):
        values = (enhanced[index], clean[index], noisy[index])
        if not all(torch.isfinite(value).all() for value in values):
            rows.append({"valid": False, "error": "non_finite"})
            continue
        if clean[index].square().sum() <= eps:
            rows.append({"valid": False, "error": "silent_reference"})
            continue
        input_sdr = scale_dependent_sdr(noisy[index : index + 1], clean[index : index + 1], eps)[0]
        output_sdr = scale_dependent_sdr(enhanced[index : index + 1], clean[index : index + 1], eps)[0]
        input_si = si_snr(noisy[index : index + 1], clean[index : index + 1], eps)[0]
        output_si = si_snr(enhanced[index : index + 1], clean[index : index + 1], eps)[0]
        metric_values = [input_sdr, output_sdr, input_si, output_si]
        if min_db is not None or max_db is not None:
            low = -float("inf") if min_db is None else float(min_db)
            high = float("inf") if max_db is None else float(max_db)
            metric_values = [value.clamp(low, high) for value in metric_values]
        input_sdr, output_sdr, input_si, output_si = metric_values
        rows.append(
            {
                "input_sdr": float(input_sdr),
                "output_sdr": float(output_sdr),
                "sdri": float(output_sdr - input_sdr),
                "input_si_snr": float(input_si),
                "output_si_snr": float(output_si),
                "si_snri": float(output_si - input_si),
                "duration": clean.shape[-1] / float(sample_rate),
                "valid": True,
                "error": "",
            }
        )
    return rows


def summarize_metric_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate valid metric rows with quartiles and invalid count."""
    valid = [row for row in rows if row.get("valid", False)]
    names = (
        "input_sdr",
        "output_sdr",
        "sdri",
        "input_si_snr",
        "output_si_snr",
        "si_snri",
        "inference_time",
        "rtf",
    )
    summary: Dict[str, Any] = {
        "count": len(rows),
        "valid_count": len(valid),
        "invalid_count": len(rows) - len(valid),
    }
    for name in names:
        values = np.asarray(
            [row[name] for row in valid if name in row], dtype=np.float64
        )
        if values.size:
            summary[name] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "median": float(np.median(values)),
                "p25": float(np.percentile(values, 25)),
                "p75": float(np.percentile(values, 75)),
            }
    return summary

