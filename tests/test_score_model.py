"""Score loss and NCSN++ shape tests."""

import torch

from sgmse.diffusion.losses import complex_score_mse
from sgmse.models.ncsnpp import NCSNpp


def test_score_loss_is_zero_for_exact_target() -> None:
    target = torch.complex(torch.randn(2, 17, 13), torch.randn(2, 17, 13))
    loss = complex_score_mse(target.clone(), target)
    assert float(loss) < 1.0e-12


def test_ncsnpp_forward_shape_and_parameter_count() -> None:
    model = NCSNpp(
        base_channels=8,
        channel_multipliers=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(16,),
        image_size=32,
    )
    inputs = torch.randn(2, 4, 33, 31)
    output = model(inputs, torch.tensor([0.1, 0.9]))
    assert output.shape == (2, 2, 33, 31)
    assert torch.isfinite(output).all()
    assert model.parameter_count() > 0

