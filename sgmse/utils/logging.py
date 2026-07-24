"""Rank-aware console/file, JSONL, and TensorBoard logging."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional


def create_logger(
    name: str,
    output_dir: str,
    enabled: bool = True,
    filename: str = "training.log",
) -> logging.Logger:
    """Create an idempotent logger; non-main ranks get a null handler."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    if not enabled:
        logger.addHandler(logging.NullHandler())
        return logger
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(directory / filename, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


class JsonlLogger:
    """Append structured epoch/step records."""

    def __init__(self, path: str, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = bool(enabled)
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, values: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(values, ensure_ascii=False) + "\n")


def create_summary_writer(output_dir: str, enabled: bool = True) -> Optional[Any]:
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except (ImportError, ModuleNotFoundError):
        return None
    return SummaryWriter(log_dir=str(Path(output_dir) / "tensorboard"))
