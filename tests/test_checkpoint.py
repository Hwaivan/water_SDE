"""Complete model/EMA/optimizer/scaler checkpoint round trip."""

from pathlib import Path

import torch

from sgmse.utils.checkpoint import (
    restore_training_state,
    save_checkpoint,
    training_state,
)
from sgmse.utils.ema import ExponentialMovingAverage


def test_checkpoint_save_restore(tmp_path: Path) -> None:
    model = torch.nn.Linear(4, 2)
    ema = ExponentialMovingAverage(model, 0.9)
    optimizer = torch.optim.Adam(model.parameters())
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    original = {name: value.clone() for name, value in model.state_dict().items()}
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        training_state(
            model, ema, optimizer, None, scaler, 3, 17, 2.5, {"test": True}
        ),
        str(path),
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    checkpoint = restore_training_state(
        str(path),
        model,
        ema,
        optimizer,
        None,
        scaler,
        "cpu",
        restore_rng=False,
    )
    assert checkpoint["epoch"] == 3
    assert checkpoint["global_step"] == 17
    assert all(
        torch.equal(model.state_dict()[name], value)
        for name, value in original.items()
    )

