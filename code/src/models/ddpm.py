"""Stage-2 generator #2: a DDPM over the frozen DeepSDF latent codes.

This is the *expressive* prior. It uses the exact noise-prediction formulation
from HW3 (Ho et al., 2020), applied to D-dimensional vectors instead of images:
train ``eps_phi(z_t, t)`` to predict the noise added to a code, then generate by
ancestral sampling from Gaussian noise. Implemented from scratch.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def sinusoidal_time_embedding(timesteps: Tensor, dim: int) -> Tensor:
    """Standard sinusoidal embedding of integer timesteps -> [B, dim]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:  # zero-pad if dim is odd
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class MLPDenoiser(nn.Module):
    """Predicts the noise in a noisy latent code, conditioned on the timestep."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        time_embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.act = nn.SiLU()
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_t: Tensor, t: Tensor) -> Tensor:
        temb = self.time_mlp(sinusoidal_time_embedding(t, self.time_embed_dim))
        h = self.input_proj(z_t)
        for block in self.blocks:
            h = h + self.act(block(h + temb))  # residual block with time conditioning
        return self.output_proj(h)


class LatentDDPM(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_timesteps: int = 200,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "cosine",
        hidden_dim: int = 256,
        num_layers: int = 4,
        time_embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_timesteps = num_timesteps
        self.model = MLPDenoiser(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            time_embed_dim=time_embed_dim,
        )

        betas = self._make_beta_schedule(schedule, num_timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1), alphas_cumprod[:-1]], dim=0
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_var)

    @staticmethod
    def _make_beta_schedule(
        schedule: str, num_timesteps: int, beta_start: float, beta_end: float
    ) -> Tensor:
        if schedule == "linear":
            return torch.linspace(beta_start, beta_end, num_timesteps)
        if schedule == "cosine":
            # Nichol & Dhariwal (2021) cosine schedule.
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            s = 0.008
            f = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi / 2) ** 2
            alphas_cumprod = f / f[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return betas.clamp(1e-5, 0.999)
        raise ValueError(f"Unknown schedule: {schedule!r}")

    def q_sample(self, z_0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Forward diffusion: add noise to a clean code at timestep ``t``."""
        sqrt_ac = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return sqrt_ac * z_0 + sqrt_om * noise

    def loss(self, z_0: Tensor) -> Tensor:
        """MSE between predicted and true noise (the standard DDPM objective)."""
        batch = z_0.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch,), device=z_0.device)
        noise = torch.randn_like(z_0)
        z_t = self.q_sample(z_0, t, noise)
        noise_pred = self.model(z_t, t)
        return torch.mean((noise_pred - noise) ** 2)

    @torch.no_grad()
    def sample(
        self, num_samples: int, device: str = "cpu", generator: torch.Generator | None = None
    ) -> Tensor:
        """Ancestral sampling from N(0, I) back to a clean code."""
        z = torch.randn(num_samples, self.latent_dim, device=device, generator=generator)
        for step in reversed(range(self.num_timesteps)):
            t = torch.full((num_samples,), step, device=device, dtype=torch.long)
            noise_pred = self.model(z, t)

            beta_t = self.betas[step]
            alpha_t = self.alphas[step]
            sqrt_om = self.sqrt_one_minus_alphas_cumprod[step]

            mean = (z - beta_t / sqrt_om * noise_pred) / torch.sqrt(alpha_t)
            if step > 0:
                var = self.posterior_variance[step]
                noise = torch.randn(
                    num_samples, self.latent_dim, device=device, generator=generator
                )
                z = mean + torch.sqrt(var) * noise
            else:
                z = mean
        return z
