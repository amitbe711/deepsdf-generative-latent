"""DDPM sanity: forward noising shapes, loss decreases, sampling returns codes."""

import torch

from src.models.ddpm import LatentDDPM
from src.train import train_ddpm
from src.utils import AttrDict


def _ddpm_cfg(num_iters: int) -> AttrDict:
    return AttrDict(
        {
            "seed": 0,
            "ddpm": {
                "num_timesteps": 100,
                "beta_start": 1.0e-4,
                "beta_end": 0.02,
                "schedule": "cosine",
                "hidden_dim": 128,
                "num_layers": 3,
                "time_embed_dim": 32,
                "lr": 1.0e-3,
                "num_iters": num_iters,
                "batch_size": 64,
            },
        }
    )


def test_q_sample_shapes_and_schedule():
    model = LatentDDPM(latent_dim=16, num_timesteps=100)
    z0 = torch.randn(8, 16)
    t = torch.randint(0, 100, (8,))
    noise = torch.randn_like(z0)
    zt = model.q_sample(z0, t, noise)
    assert zt.shape == z0.shape
    # alphas_cumprod is monotonically non-increasing in a valid schedule.
    ac = model.alphas_cumprod
    assert torch.all(ac[1:] <= ac[:-1] + 1e-6)


def test_ddpm_training_reduces_loss_and_samples():
    torch.manual_seed(0)
    # Structured codes: a tight cluster the model should learn.
    codes = torch.randn(64, 16) * 0.3 + 2.0
    cfg = _ddpm_cfg(num_iters=600)
    out = train_ddpm(cfg, codes, device="cpu")
    hist = out["history"]
    assert hist[-1]["loss"] < hist[0]["loss"]

    samples = out["model"].sample(10, device="cpu")
    assert samples.shape == (10, 16)
    assert torch.isfinite(samples).all()
