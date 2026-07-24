"""Generate enhanced test WAVs and explicit SDR/SI-SNR reports."""

import argparse
import json
from pathlib import Path

import torch

from sgmse.data import SGMSEDataset, build_dataloader
from sgmse.engine.evaluator import evaluate
from sgmse.factory import build_components
from sgmse.utils.checkpoint import load_model_for_inference
from sgmse.utils.config import load_config
from sgmse.utils.distributed import (
    barrier,
    cleanup_distributed,
    initialize_distributed,
)
from sgmse.utils.logging import create_logger
from sgmse.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SGMSE+ checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="")
    args = parser.parse_args()
    config = load_config(args.config)
    context = initialize_distributed(args.device)
    try:
        if not context.is_main:
            barrier(context)
            return
        seed = int(config["validation"].get("seed", 1234))
        seed_everything(seed, True)
        model, _, _, _, sampler = build_components(config)
        model = model.to(context.device).eval()
        load_model_for_inference(
            args.checkpoint,
            model,
            str(context.device),
            bool(config["sampler"].get("use_ema", True)),
        )
        dataset = SGMSEDataset(config["data"], args.split, seed)
        loader = build_dataloader(dataset, config, False, seed, False)
        logger = create_logger(
            "sgmse_evaluation",
            args.output_dir,
            True,
            filename="evaluation.log",
        )
        try:
            generator = torch.Generator(device=context.device)
        except TypeError:
            generator = torch.Generator(device=context.device.type)
        generator.manual_seed(seed)
        summary = evaluate(
            model,
            sampler,
            loader,
            context.device,
            config,
            args.checkpoint,
            args.output_dir,
            seed,
            generator,
            logger,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        barrier(context)
    finally:
        cleanup_distributed(context)


if __name__ == "__main__":
    main()

