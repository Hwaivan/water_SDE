"""Enhance one mono file with full-spectrum or Hann overlap-add sampling."""

import argparse
import json

import torch

from sgmse.data.audio_io import load_audio, save_audio
from sgmse.diffusion.sampler import enhance_long_waveform
from sgmse.factory import build_components
from sgmse.utils.checkpoint import load_model_for_inference
from sgmse.utils.config import load_config
from sgmse.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="SGMSE+ single-file inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--chunk-seconds", type=float, default=None)
    parser.add_argument("--overlap-seconds", type=float, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    seed = int(config.get("inference", {}).get("seed", 1234))
    seed_everything(seed, True)
    model, _, _, _, sampler = build_components(config)
    model = model.to(device).eval()
    load_model_for_inference(
        args.checkpoint,
        model,
        str(device),
        bool(config["sampler"].get("use_ema", True)),
    )
    sample_rate = int(config["data"]["sample_rate"])
    waveform = load_audio(args.input, sample_rate, bool(config["data"]["mono"]))
    inference = config.get("inference", {})
    chunk_seconds = (
        args.chunk_seconds
        if args.chunk_seconds is not None
        else inference.get("chunk_seconds")
    )
    overlap_seconds = (
        args.overlap_seconds
        if args.overlap_seconds is not None
        else float(inference.get("overlap_seconds", 0.5))
    )
    try:
        generator = torch.Generator(device=device)
    except TypeError:
        generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)
    result = enhance_long_waveform(
        sampler,
        model,
        waveform,
        sample_rate,
        generator,
        None if chunk_seconds is None else int(round(chunk_seconds * sample_rate)),
        int(round(overlap_seconds * sample_rate)),
        bool(inference.get("fallback_on_oom", True)),
    )
    enhanced = result.waveform[0, : waveform.numel()].cpu()
    save_audio(args.output, enhanced, sample_rate)
    print(
        json.dumps(
            {
                "output": args.output,
                "samples": enhanced.numel(),
                "nfe": result.nfe,
                "inference_time": result.inference_time,
                "rtf": result.rtf,
                "seed": seed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

