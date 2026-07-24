"""OUVE broadcasting, marginal statistics, and complex-noise tests."""

import torch

from sgmse.diffusion.ouve import OUVESDE, complex_gaussian


def test_ouve_mean_std_broadcast_and_range() -> None:
    sde = OUVESDE()
    x0 = torch.ones(3, 4, 5, dtype=torch.complex64)
    y = torch.zeros_like(x0)
    t = torch.tensor([0.03, 0.5, 1.0])
    mean, std = sde.marginal_prob(x0, y, t)
    assert mean.shape == x0.shape
    assert std.shape == (3,)
    assert torch.isfinite(mean.real).all()
    assert torch.all(std > 0)
    assert torch.all(std[1:] > std[:-1])


def test_ouve_monte_carlo_matches_closed_marginal() -> None:
    torch.manual_seed(4)
    count = 30000
    sde = OUVESDE()
    x0 = torch.full((count, 1, 1), 0.7 + 0.2j, dtype=torch.complex64)
    y = torch.full_like(x0, -0.1 + 0.4j)
    t = torch.full((count,), 0.6)
    mean, std = sde.marginal_prob(x0, y, t)
    samples = mean + std[:, None, None] * complex_gaussian(
        x0.shape, x0.device
    )
    empirical_mean = samples.mean()
    expected_mean = mean[0, 0, 0]
    empirical_variance = (samples - mean).abs().square().mean()
    expected_variance = sde.variance(t[:1])[0]
    assert torch.allclose(empirical_mean, expected_mean, atol=8.0e-3, rtol=0.0)
    assert torch.allclose(
        empirical_variance, expected_variance, atol=2.0e-3, rtol=0.04
    )


def test_complex_gaussian_has_unit_expected_energy() -> None:
    torch.manual_seed(5)
    noise = complex_gaussian((200000,), torch.device("cpu"))
    energy = noise.abs().square().mean()
    assert abs(float(energy) - 1.0) < 0.015

