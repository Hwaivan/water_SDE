"""Build shared SGMSE mathematical and neural components from YAML."""

from typing import Any, Dict, Tuple

from sgmse.diffusion.losses import ScoreMatchingObjective
from sgmse.diffusion.ouve import OUVESDE
from sgmse.diffusion.sampler import PredictorCorrectorSampler
from sgmse.models import NCSNpp, build_score_model
from sgmse.utils.stft import ComplexSTFTTransform


def build_transform(config: Dict[str, Any]) -> ComplexSTFTTransform:
    values = dict(config["stft"])
    values.update(config["compression"])
    values.pop("crop_frames", None)
    return ComplexSTFTTransform(**values)


def build_sde(config: Dict[str, Any]) -> OUVESDE:
    return OUVESDE(**config["sde"])


def build_components(
    config: Dict[str, Any],
) -> Tuple[NCSNpp, ComplexSTFTTransform, OUVESDE, ScoreMatchingObjective, PredictorCorrectorSampler]:
    transform = build_transform(config)
    sde = build_sde(config)
    model = build_score_model(config)
    objective = ScoreMatchingObjective(
        transform,
        sde,
        crop_frames=config["compression"].get("crop_frames"),
        auxiliary_waveform_weight=float(
            config["training"].get("auxiliary_waveform_weight", 0.0)
        ),
    )
    sampler_values = dict(config["sampler"])
    sampler_values.pop("use_ema", None)
    sampler = PredictorCorrectorSampler(sde, transform, **sampler_values)
    return model, transform, sde, objective, sampler

