"""Train SGMSE+ score model with single GPU, DataParallel, or torchrun DDP."""

import argparse
from pathlib import Path

import torch

from sgmse.data import SGMSEDataset, build_dataloader
from sgmse.engine import SGMSETrainer
from sgmse.factory import build_components
from sgmse.utils.config import load_config
from sgmse.utils.distributed import (
    cleanup_distributed,
    initialize_distributed,
    unwrap_model,
    wrap_model,
)
from sgmse.utils.ema import ExponentialMovingAverage
from sgmse.utils.logging import JsonlLogger, create_logger, create_summary_writer
from sgmse.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train complex-STFT SGMSE+")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    context = initialize_distributed(args.device)
    try:
        seed = int(config["experiment"]["seed"])
        seed_everything(
            seed + context.rank,
            bool(config["experiment"].get("deterministic", True)),
        )
        output_dir = config["experiment"]["output_dir"]
        logger = create_logger(
            config["experiment"]["name"], output_dir, context.is_main
        )
        train_dataset = SGMSEDataset(config["data"], "train", seed)
        valid_dataset = SGMSEDataset(config["data"], "valid", seed + 1)
        train_loader = build_dataloader(
            train_dataset, config, True, seed, context.enabled
        )
        valid_loader = build_dataloader(
            valid_dataset, config, False, seed + 1, context.enabled
        )
        sampling_loader = (
            build_dataloader(valid_dataset, config, False, seed + 1, False)
            if context.is_main
            else None
        )
        model, _, _, objective, sampler = build_components(config)
        model = model.to(context.device)
        ema = ExponentialMovingAverage(
            model, float(config["training"]["ema_decay"])
        )
        model = wrap_model(
            model,
            context,
            bool(config["distributed"].get("data_parallel", False)),
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"].get("weight_decay", 0.0)),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(config["training"].get("scheduler_factor", 0.5)),
            patience=int(config["training"].get("scheduler_patience", 5)),
            min_lr=float(config["training"].get("min_lr", 1.0e-7)),
        )
        logger.info(
            "rank=%d/%d device=%s train=%d valid=%d parameters=%d",
            context.rank,
            context.world_size,
            context.device,
            len(train_dataset),
            len(valid_dataset),
            sum(p.numel() for p in unwrap_model(model).parameters()),
        )
        trainer = SGMSETrainer(
            model,
            objective,
            sampler,
            optimizer,
            scheduler,
            ema,
            context,
            config,
            logger,
            JsonlLogger(
                str(Path(output_dir) / "metrics.jsonl"),
                context.is_main and bool(config["logging"].get("jsonl", True)),
            ),
            create_summary_writer(
                output_dir,
                context.is_main
                and bool(config["logging"].get("tensorboard", True)),
            ),
        )
        resume = args.resume or config["training"].get("resume")
        if resume:
            trainer.resume(resume)
        trainer.fit(train_loader, valid_loader, sampling_loader)
    finally:
        cleanup_distributed(context)


if __name__ == "__main__":
    main()

