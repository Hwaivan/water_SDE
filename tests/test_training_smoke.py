"""One CPU score-training optimization smoke test."""

import torch

from sgmse.diffusion.losses import ScoreMatchingObjective
from sgmse.diffusion.ouve import OUVESDE
from sgmse.models.ncsnpp import NCSNpp
from sgmse.utils.stft import ComplexSTFTTransform


def test_cpu_smoke_training_step() -> None:
    torch.manual_seed(10)
    transform = ComplexSTFTTransform(32, 32, 8, alpha=0.5, beta=0.15)
    objective = ScoreMatchingObjective(transform, OUVESDE(), crop_frames=16)
    model = NCSNpp(
        base_channels=8,
        channel_multipliers=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(),
        image_size=16,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    clean = torch.randn(2, 256) * 0.05
    noisy = clean + 0.02 * torch.randn_like(clean)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        losses = objective(model, clean, noisy)
        assert torch.isfinite(losses["total"])
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

