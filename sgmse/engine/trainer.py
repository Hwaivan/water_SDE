"""AMP/DDP/EMA SGMSE trainer with deterministic generative validation."""

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from torch import nn

from sgmse.diffusion.losses import ScoreMatchingObjective
from sgmse.diffusion.sampler import PredictorCorrectorSampler
from sgmse.metrics.audio_metrics import compute_audio_metrics
from sgmse.utils.checkpoint import (
    restore_training_state,
    save_checkpoint,
    training_state,
)
from sgmse.utils.distributed import (
    DistributedContext,
    barrier,
    reduce_mean,
    unwrap_model,
)
from sgmse.utils.ema import ExponentialMovingAverage
from sgmse.utils.logging import JsonlLogger


def _device_generator(device: torch.device, seed: int) -> torch.Generator:
    """Create a generator on the same device as random tensors."""
    try:
        generator = torch.Generator(device=device)
    except TypeError:
        generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))
    return generator


class SGMSETrainer:
    """Complete score-model training state machine."""

    def __init__(
        self,
        model: nn.Module,
        objective: ScoreMatchingObjective,
        sampler: PredictorCorrectorSampler,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        ema: ExponentialMovingAverage,
        context: DistributedContext,
        config: Dict[str, Any],
        logger: logging.Logger,
        jsonl_logger: JsonlLogger,
        writer: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.objective = objective.to(context.device)
        self.sampler = sampler
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema = ema
        self.context = context
        self.config = config
        self.logger = logger
        self.jsonl = jsonl_logger
        self.writer = writer
        training = config["training"]
        self.amp_enabled = bool(
            training.get("amp", True) and context.device.type == "cuda"
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.gradient_clip = float(training.get("gradient_clip_norm", 1.0))
        self.output_dir = Path(config["experiment"]["output_dir"])
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.start_epoch = 0
        self.global_step = 0
        self.best_metric = -math.inf

    def resume(self, path: str) -> None:
        """Restore full training state and resume at the following epoch."""
        checkpoint = restore_training_state(
            path,
            unwrap_model(self.model),
            self.ema,
            self.optimizer,
            self.scheduler,
            self.scaler,
            str(self.context.device),
            restore_rng=True,
        )
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.global_step = int(checkpoint.get("global_step", 0))
        self.best_metric = float(checkpoint.get("best_metric", -math.inf))
        self.logger.info(
            "Resumed %s at epoch=%d global_step=%d",
            path,
            self.start_epoch,
            self.global_step,
        )

    def _move(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return {
            name: batch[name].to(self.context.device, non_blocking=True)
            for name in ("clean", "noisy", "lengths")
        }

    def train_one_epoch(
        self, loader: Iterable[Dict[str, Any]], epoch: int
    ) -> Dict[str, float]:
        """Run one epoch and report loss, grad norm, LR, sigma and time means."""
        self.model.train()
        totals = {
            "loss": 0.0,
            "grad_norm": 0.0,
            "sigma_mean": 0.0,
            "t_mean": 0.0,
            "batches": 0.0,
        }
        for batch_index, raw_batch in enumerate(loader):
            batch = self._move(raw_batch)
            self.optimizer.zero_grad(set_to_none=True)
            try:
                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    losses = self.objective(
                        self.model, batch["clean"], batch["noisy"]
                    )
            except RuntimeError as error:
                if "out of memory" in str(error).lower():
                    raise RuntimeError(
                        "CUDA OOM at epoch {}, batch {}; reduce batch_size, "
                        "crop_frames, base_channels, or channel_multipliers".format(
                            epoch, batch_index
                        )
                    ) from error
                raise
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(
                    "Non-finite score loss at epoch {}, batch {}".format(
                        epoch, batch_index
                    )
                )
            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite gradient norm")
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.ema.update(unwrap_model(self.model))
            self.global_step += 1
            totals["loss"] += float(losses["total"].detach())
            totals["grad_norm"] += float(gradient_norm)
            totals["sigma_mean"] += float(losses["sigma_mean"])
            totals["t_mean"] += float(losses["t_mean"])
            totals["batches"] += 1.0
        count = max(1.0, totals.pop("batches"))
        local = {
            name: torch.tensor(value / count, device=self.context.device)
            for name, value in totals.items()
        }
        reduced = {
            name: float(reduce_mean(value, self.context).cpu())
            for name, value in local.items()
        }
        reduced["lr"] = float(self.optimizer.param_groups[0]["lr"])
        return reduced

    @torch.inference_mode()
    def validate_score(
        self, loader: Iterable[Dict[str, Any]], seed: int
    ) -> float:
        """Compute deterministic score loss with fixed times and noise stream."""
        self.model.eval()
        generator = _device_generator(self.context.device, seed + self.context.rank)
        total, batches = 0.0, 0
        for raw_batch in loader:
            batch = self._move(raw_batch)
            losses = self.objective(
                self.model,
                batch["clean"],
                batch["noisy"],
                generator=generator,
                deterministic=True,
            )
            total += float(losses["total"])
            batches += 1
        value = torch.tensor(total / max(1, batches), device=self.context.device)
        return float(reduce_mean(value, self.context).cpu())

    @torch.inference_mode()
    def validate_sampling(
        self,
        loader: Optional[Iterable[Dict[str, Any]]],
        seed: int,
        max_batches: int,
    ) -> Optional[Dict[str, float]]:
        """Rank-0 fixed-subset reverse diffusion validation using EMA weights."""
        barrier(self.context)
        if not self.context.is_main:
            barrier(self.context)
            return None
        if loader is None:
            raise ValueError("Main rank requires a non-distributed sampling loader")
        base_model = unwrap_model(self.model)
        base_model.eval()
        generator = _device_generator(self.context.device, seed)
        si_values, sdr_values = [], []
        with self.ema.average_parameters(base_model):
            for batch_index, raw_batch in enumerate(loader):
                if batch_index >= max_batches:
                    break
                batch = self._move(raw_batch)
                result = self.sampler.sample_waveform(
                    base_model,
                    batch["noisy"],
                    batch["lengths"],
                    int(self.config["data"]["sample_rate"]),
                    generator,
                )
                for item_index, row in enumerate(
                    compute_audio_metrics(
                        result.waveform,
                        batch["clean"],
                        batch["noisy"],
                        int(self.config["data"]["sample_rate"]),
                        self.config["metrics"].get("alignment_policy", "crop"),
                    )
                ):
                    if row["valid"]:
                        si_values.append(row["si_snri"])
                        sdr_values.append(row["sdri"])
        metrics = {
            "si_snri": float(sum(si_values) / max(1, len(si_values))),
            "sdri": float(sum(sdr_values) / max(1, len(sdr_values))),
            "valid_count": float(len(si_values)),
        }
        barrier(self.context)
        return metrics

    def _save(self, epoch: int, name: str) -> None:
        if not self.context.is_main:
            return
        state = training_state(
            unwrap_model(self.model),
            self.ema,
            self.optimizer,
            self.scheduler,
            self.scaler,
            epoch,
            self.global_step,
            self.best_metric,
            self.config,
        )
        save_checkpoint(state, str(self.checkpoint_dir / name))

    def fit(
        self,
        train_loader: Any,
        valid_loader: Any,
        sampling_valid_loader: Optional[Any],
    ) -> None:
        """Train, validate score every epoch, and sample at fixed intervals."""
        training = self.config["training"]
        validation = self.config["validation"]
        epochs = int(training["epochs"])
        eval_interval = int(validation.get("eval_interval", 5))
        validation_seed = int(validation.get("seed", 1234))
        max_batches = int(validation.get("max_sampling_batches", 1))
        for epoch in range(self.start_epoch, epochs):
            started = time.time()
            if hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(epoch)
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            train_stats = self.train_one_epoch(train_loader, epoch)
            val_score_loss = self.validate_score(valid_loader, validation_seed)
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_score_loss)
                else:
                    self.scheduler.step()
            sample_metrics = None
            if (epoch + 1) % eval_interval == 0 or epoch + 1 == epochs:
                sample_metrics = self.validate_sampling(
                    sampling_valid_loader, validation_seed, max_batches
                )
            improved = False
            if self.context.is_main and sample_metrics is not None:
                improved = sample_metrics["si_snri"] > self.best_metric
                if improved:
                    self.best_metric = sample_metrics["si_snri"]
            record: Dict[str, Any] = {
                "epoch": epoch + 1,
                "global_step": self.global_step,
                "train/loss": train_stats["loss"],
                "train/grad_norm": train_stats["grad_norm"],
                "train/lr": train_stats["lr"],
                "train/sigma_mean": train_stats["sigma_mean"],
                "train/t_mean": train_stats["t_mean"],
                "val/score_loss": val_score_loss,
                "epoch_seconds": time.time() - started,
            }
            if sample_metrics is not None:
                record["val/SI-SNRi"] = sample_metrics["si_snri"]
                record["val/SDRi"] = sample_metrics["sdri"]
            if self.context.is_main:
                self.jsonl.write(record)
                self.logger.info(
                    "epoch=%d loss=%.6f val_score=%.6f grad=%.3f "
                    "lr=%.3e sigma=%.4f t=%.4f SI-SNRi=%s SDRi=%s time=%.1fs",
                    epoch + 1,
                    train_stats["loss"],
                    val_score_loss,
                    train_stats["grad_norm"],
                    train_stats["lr"],
                    train_stats["sigma_mean"],
                    train_stats["t_mean"],
                    "n/a" if sample_metrics is None else "{:.3f}".format(sample_metrics["si_snri"]),
                    "n/a" if sample_metrics is None else "{:.3f}".format(sample_metrics["sdri"]),
                    record["epoch_seconds"],
                )
                if self.writer is not None:
                    for key, value in record.items():
                        if isinstance(value, (int, float)):
                            self.writer.add_scalar(key, value, epoch + 1)
            self._save(epoch, "last.pt")
            if improved:
                self._save(epoch, "best.pt")
            save_every = int(training.get("save_every", 10))
            if save_every > 0 and (epoch + 1) % save_every == 0:
                self._save(epoch, "epoch_{:04d}.pt".format(epoch + 1))
            barrier(self.context)
        if self.context.is_main and self.writer is not None:
            self.writer.close()

